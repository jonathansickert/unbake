"""
Find Objaverse-XL objects with "real" PBR materials and add them to data_toolkit/selected_assets.json.

Downloads candidates in random order to a temporary folder and keeps only those with
  - a single PBR material (one texture set and UV layout),
  - non-constant base color, roughness and metallic textures,
  - clean UVs: present, with no face outside [0, 1], degenerate or overlapping another face,
  - a license that allows commercial use and derivatives (no NC / ND).
Accepted objects (sha256, source, license, check statistics) are appended to the list; the meshes are
deleted. Every checked object is recorded in data_toolkit/checked_assets.json (sha256 -> rejection reason,
null if accepted), so later runs skip it. Download the accepted objects with download_assets.py afterwards.

Run from the repo root (pip install objaverse trimesh pandas pyarrow numpy pillow tqdm):
    python data_toolkit/select_assets.py      # runs over all candidates; stop any time with Ctrl+C
"""
import os
import json
import argparse
import shutil
import tempfile
from types import SimpleNamespace
import numpy as np
import trimesh
from tqdm import tqdm
import objaverse.xl as oxl

LIST_PATH = 'data_toolkit/selected_assets.json'
CHECKED_PATH = 'data_toolkit/checked_assets.json'
ANNOTATIONS_DIR = 'objaverse_cache'  # Objaverse-XL annotation tables (meshes go to temporary folders)
CHECKS = SimpleNamespace(
    max_materials=1,       # maximum number of materials (texture sets)
    min_std=0.02,          # minimum texture std for a channel to count as non-constant
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
        # strictly inside: a texel center on an edge shared by two faces must not count as overlap
        inside = (w0 > 1e-9) & (w1 > 1e-9) & (w0 + w1 < 1 - 1e-9)
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
    record['num_materials'] = len(groups)
    if len(groups) > opt.max_materials:
        record['reason'] = 'too many materials'
        return False, record

    albedo_std, roughness_std, metallic_std = 0.0, 0.0, 0.0
    for material, _ in groups.values():
        # Material channels: albedo = baseColor RGB, roughness = mR.G, metallic = mR.B (glTF spec)
        if material.baseColorTexture is not None:
            albedo_std = max(albedo_std, texture_std(material.baseColorTexture, [0, 1, 2]))
        if material.metallicRoughnessTexture is not None:
            mr = material.metallicRoughnessTexture
            roughness_factor = 1.0 if material.roughnessFactor is None else material.roughnessFactor
            metallic_factor = 1.0 if material.metallicFactor is None else material.metallicFactor
            roughness_std = max(roughness_std, roughness_factor * texture_std(mr, [1]))
            metallic_std = max(metallic_std, metallic_factor * texture_std(mr, [2]))
    record.update({'albedo_std': albedo_std, 'roughness_std': roughness_std, 'metallic_std': metallic_std})
    if albedo_std < opt.min_std:
        record['reason'] = 'constant albedo'
    elif roughness_std < opt.min_std:
        record['reason'] = 'constant roughness'
    elif metallic_std < opt.min_std:
        record['reason'] = 'constant metallic'
    if record['reason'] is not None:
        return False, record   # skip the slow UV rasterization

    worst_overlap, worst_oor, worst_degenerate = 0.0, 0.0, 0.0
    for _, group in groups.values():
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

    if worst_oor > 0:
        record['reason'] = 'UVs outside [0, 1]'
    elif worst_degenerate > 0:
        record['reason'] = 'degenerate UVs'
    elif worst_overlap > 0:
        record['reason'] = 'overlapping UVs'
    return record['reason'] is None, record


def save(obj, path):
    """Write JSON via a temp file, so Ctrl+C mid-write never leaves a broken file."""
    with open(path + '.tmp', 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(path + '.tmp', path)


def download(batch, download_dir):
    """fileIdentifier -> local path. One failed download aborts objaverse's whole batch, so on failure the
    batch is retried object by object and the objects that still fail are left out (retried in a later run)."""
    try:
        return oxl.download_objects(objects=batch, download_dir=download_dir, processes=8)
    except Exception:
        downloaded = {}
        for i in range(len(batch)):
            try:
                downloaded.update(oxl.download_objects(objects=batch.iloc[i:i + 1], download_dir=download_dir,
                                                       processes=1))
            except Exception as e:
                print(f'Download failed: {batch.iloc[i]["fileIdentifier"]} ({type(e).__name__})')
        return downloaded


def filter_objects(opt):
    accepted = json.load(open(opt.list)) if os.path.exists(opt.list) else []
    checked = json.load(open(CHECKED_PATH)) if os.path.exists(CHECKED_PATH) else {}
    annotations = oxl.get_annotations(download_dir=ANNOTATIONS_DIR)
    # Only glTF binaries carry the PBR textures we can check
    annotations = annotations[(annotations['source'] == opt.source) & (annotations['fileType'] == 'glb')]
    annotations = annotations[~annotations['license'].str.contains('Non-Commercial|No Derivatives', na=False)]
    annotations = annotations.drop_duplicates('sha256').sample(frac=1, random_state=0)
    annotations = annotations[~annotations['sha256'].isin({r['sha256'] for r in accepted} | set(checked))]

    rejected = {}
    pbar = tqdm(total=len(annotations), desc='Checked')
    tmp = tempfile.TemporaryDirectory()
    # objaverse looks up its Sketchfab index (object-paths.json.gz) in the download folder: reuse the cached one
    index = os.path.join(ANNOTATIONS_DIR, 'hf-objaverse-v1', 'object-paths.json.gz')
    if os.path.exists(index):
        os.makedirs(os.path.join(tmp.name, 'hf-objaverse-v1'))
        shutil.copy(index, os.path.join(tmp.name, 'hf-objaverse-v1'))
    try:
        for start in range(0, len(annotations), 32):
            batch = annotations.iloc[start:start + 32]
            downloaded = download(batch, tmp.name)
            for _, row in batch.iterrows():
                pbar.update()
                path = downloaded.get(row['fileIdentifier'])
                if path is None or not os.path.exists(path):
                    continue  # failed download: retried in a later run
                try:
                    passed, record = check_object(path, CHECKS)
                except Exception as e:
                    passed, record = False, {'reason': f'check failed: {e}'}
                os.remove(path)
                checked[row['sha256']] = record.pop('reason')
                if passed:
                    accepted.append({'sha256': row['sha256'], 'file_identifier': row['fileIdentifier'],
                                     'source': row['source'], 'license': row['license'], **record})
                else:
                    reason = checked[row['sha256']].split(':')[0]
                    rejected[reason] = rejected.get(reason, 0) + 1
            save(accepted, opt.list)   # accepted first: a crash in between only re-checks objects
            save(checked, CHECKED_PATH)
            pbar.set_postfix(accepted=len(accepted))
    except KeyboardInterrupt:
        print('Stopped.')
    finally:
        pbar.close()
        tmp.cleanup()
        save(accepted, opt.list)
        save(checked, CHECKED_PATH)

    print(f'{len(accepted)} objects in {opt.list}. Rejections this run:')
    for reason, count in sorted(rejected.items(), key=lambda x: -x[1]):
        print(f'  {reason}: {count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='sketchfab', help='sketchfab, github, thingiverse or smithsonian')
    parser.add_argument('--list', default=LIST_PATH, help='JSON list of accepted objects')
    filter_objects(parser.parse_args())
