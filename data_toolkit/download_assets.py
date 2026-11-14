"""
Download Objaverse-XL objects with "real" PBR materials into the dataset folder.

Traverses the Objaverse-XL glb annotations in random order and keeps only objects with
  - non-constant base color, roughness and metallic textures,
  - clean UVs: present, inside [0, 1], non-degenerate and non-overlapping within each material.

Accepted objects are saved as <root>/<sha256>/pbr.glb and listed in <root>/metadata.csv (source, license,
check statistics). Objects already in the dataset are skipped. Raw downloads go to objaverse_cache/.

Run from the repo root (pip install objaverse trimesh pandas pyarrow numpy pillow tqdm):
    python data_toolkit/download.py baked_env --num_objects 10
"""
import os
import shutil
import argparse
from types import SimpleNamespace
import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm
import objaverse.xl as oxl

CACHE_DIR = 'objaverse_cache'
CHECKS = SimpleNamespace(
    min_std=0.02,          # minimum texture std for a channel to count as non-constant
    max_overlap=0.01,      # maximum fraction of texels covered by more than one face
    max_degenerate=0.01,   # maximum fraction of faces with zero UV area
    max_faces=200000,      # skip larger objects (the UV check gets slow)
    uv_resolution=512,     # resolution of the UV overlap rasterization
)


def texture_std(image, channels, max_size=256):
    """Max per-channel std of a PIL texture, after downsampling for speed."""
    image = image.convert('RGBA')
    image.thumbnail((max_size, max_size))
    pixels = np.asarray(image, dtype=np.float32)[..., channels] / 255.0
    return float(pixels.reshape(-1, len(channels)).std(axis=0).max())


def uv_stats(uv, faces, resolution=512):
    """
    Rasterize UV triangles at pixel centers and count how often each texel is covered.

    Returns:
        out_of_range: fraction of faces with UVs outside [0, 1]
        degenerate: fraction of faces with (near) zero UV area
        overlap: fraction of covered texels covered by more than one face
    """
    tri = uv[faces]                                                     # (F, 3, 2)
    eps = 1e-3
    out_of_range = np.any((tri < -eps) | (tri > 1 + eps), axis=(1, 2))
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    degenerate = area < 1e-12

    coverage = np.zeros((resolution, resolution), dtype=np.int32)
    tri_px = np.clip(tri, 0, 1) * resolution - 0.5                      # pixel-center coordinates
    for t in tri_px[~(out_of_range | degenerate)]:
        x0, y0 = np.floor(t.min(axis=0)).astype(int)
        x1, y1 = np.ceil(t.max(axis=0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, resolution - 1), min(y1, resolution - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        p = np.stack([xs.ravel(), ys.ravel()], axis=-1).astype(np.float64)
        # barycentric inside test
        a, b, c = t
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        w0 = ((b[1] - c[1]) * (p[:, 0] - c[0]) + (c[0] - b[0]) * (p[:, 1] - c[1])) / d
        w1 = ((c[1] - a[1]) * (p[:, 0] - c[0]) + (a[0] - c[0]) * (p[:, 1] - c[1])) / d
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        p = p[inside].astype(int)
        np.add.at(coverage, (p[:, 1], p[:, 0]), 1)

    covered = (coverage > 0).sum()
    overlap = (coverage > 1).sum() / covered if covered > 0 else 1.0
    return float(out_of_range.mean()), float(degenerate.mean()), float(overlap)


def check_object(path, opt):
    """Returns (passed, record) for a downloaded object."""
    record = {'reason': None}
    try:
        scene = trimesh.load(path, force='scene', process=False)
    except Exception as e:
        record['reason'] = f'load failed: {e}'
        return False, record

    meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
    if len(meshes) == 0:
        record['reason'] = 'no meshes'
        return False, record
    num_faces = sum(len(m.faces) for m in meshes)
    record['num_faces'] = num_faces
    if num_faces > opt.max_faces:
        record['reason'] = 'too many faces'
        return False, record

    # Group meshes by material: UVs only need to be non-overlapping within one texture.
    groups = {}
    for mesh in meshes:
        visual = mesh.visual
        if not isinstance(visual, trimesh.visual.TextureVisuals) or visual.uv is None \
                or len(visual.uv) != len(mesh.vertices):
            record['reason'] = 'mesh without UVs'
            return False, record
        material = visual.material
        if not isinstance(material, trimesh.visual.material.PBRMaterial):
            record['reason'] = 'non-PBR material'
            return False, record
        groups.setdefault(id(material), (material, []))[1].append(mesh)

    albedo_std, roughness_std, metallic_std = 0.0, 0.0, 0.0
    worst_overlap, worst_oor, worst_degenerate = 0.0, 0.0, 0.0
    for material, group in groups.values():
        # Material channels: albedo = baseColor RGB, roughness = mR.G, metallic = mR.B (glTF spec)
        if material.baseColorTexture is not None:
            albedo_std = max(albedo_std, texture_std(material.baseColorTexture, [0, 1, 2]))
        if material.metallicRoughnessTexture is not None:
            mr = material.metallicRoughnessTexture
            roughness_factor = 1.0 if material.roughnessFactor is None else material.roughnessFactor
            metallic_factor = 1.0 if material.metallicFactor is None else material.metallicFactor
            roughness_std = max(roughness_std, roughness_factor * texture_std(mr, [1]))
            metallic_std = max(metallic_std, metallic_factor * texture_std(mr, [2]))

        # UV layout of all meshes sharing this material, merged into one face list
        uvs, faces, offset = [], [], 0
        for mesh in group:
            uvs.append(np.asarray(mesh.visual.uv, dtype=np.float64))
            faces.append(np.asarray(mesh.faces) + offset)
            offset += len(mesh.visual.uv)
        oor, degenerate, overlap = uv_stats(np.concatenate(uvs), np.concatenate(faces), opt.uv_resolution)
        worst_oor = max(worst_oor, oor)
        worst_degenerate = max(worst_degenerate, degenerate)
        worst_overlap = max(worst_overlap, overlap)

    record.update({
        'num_materials': len(groups),
        'albedo_std': albedo_std,
        'roughness_std': roughness_std,
        'metallic_std': metallic_std,
        'uv_out_of_range': worst_oor,
        'uv_degenerate': worst_degenerate,
        'uv_overlap': worst_overlap,
    })

    if albedo_std < opt.min_std:
        record['reason'] = 'constant albedo'
    elif roughness_std < opt.min_std:
        record['reason'] = 'constant roughness'
    elif metallic_std < opt.min_std:
        record['reason'] = 'constant metallic'
    elif worst_oor > 0:
        record['reason'] = 'UVs outside [0, 1]'
    elif worst_degenerate > opt.max_degenerate:
        record['reason'] = 'degenerate UVs'
    elif worst_overlap > opt.max_overlap:
        record['reason'] = 'overlapping UVs'
    return record['reason'] is None, record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', nargs='?', default='baked_env', help='dataset folder')
    parser.add_argument('--num_objects', type=int, default=10, help='total number of objects in the dataset')
    parser.add_argument('--source', default='sketchfab', help='sketchfab, github, thingiverse or smithsonian')
    opt = parser.parse_args()
    os.makedirs(opt.root, exist_ok=True)

    # Only glTF binaries carry the PBR textures we can check
    annotations = oxl.get_annotations(download_dir=CACHE_DIR)
    annotations = annotations[(annotations['source'] == opt.source) & (annotations['fileType'] == 'glb')]
    annotations = annotations.drop_duplicates('sha256').sample(frac=1, random_state=0)

    metadata_path = os.path.join(opt.root, 'metadata.csv')
    accepted = pd.read_csv(metadata_path).to_dict('records') if os.path.exists(metadata_path) else []
    done = {r['sha256'] for r in accepted} | set(os.listdir(opt.root))
    annotations = annotations[~annotations['sha256'].isin(done)]

    rejected, num_checked = {}, 0
    pbar = tqdm(total=opt.num_objects, initial=len(accepted), desc='Accepted')
    for start in range(0, len(annotations), 32):
        if len(accepted) >= opt.num_objects:
            break
        batch = annotations.iloc[start:start + 32]
        downloaded = oxl.download_objects(objects=batch, download_dir=CACHE_DIR, processes=8)
        for _, row in batch.iterrows():
            path = downloaded.get(row['fileIdentifier'])
            if path is None or not os.path.exists(path):
                continue
            num_checked += 1
            try:
                passed, record = check_object(path, CHECKS)
            except Exception as e:
                passed, record = False, {'reason': f'check failed: {e}'}
            if passed and len(accepted) < opt.num_objects:
                os.makedirs(os.path.join(opt.root, row['sha256']), exist_ok=True)
                shutil.move(path, os.path.join(opt.root, row['sha256'], 'pbr.glb'))
                record.pop('reason')
                accepted.append({'sha256': row['sha256'], 'file_identifier': row['fileIdentifier'],
                                 'source': row['source'], 'license': row['license'], **record})
                pd.DataFrame(accepted).to_csv(metadata_path, index=False)
                pbar.update()
            else:
                reason = (record['reason'] or 'enough objects').split(':')[0]
                rejected[reason] = rejected.get(reason, 0) + 1
                os.remove(path)
        pbar.set_postfix(checked=num_checked)
    pbar.close()

    print(f'{len(accepted)} objects in {opt.root} ({num_checked} checked this run). Rejections:')
    for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f'  {reason}: {count}')
