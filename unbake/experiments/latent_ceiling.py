"""Upper bound of any latent-space method: encode the ground-truth PBR materials with the TRELLIS.2
texture encoder, decode them again and bake into the original UV layout.

The result folders use the baseline output format, so evaluate_uv.py / evaluate_relit.py /
compare.py apply directly. The error of this reconstruction is what guidance in the texture
latent can at best reach.

    python experiments/latent_ceiling.py --data data/unbake_sub10 --out results/dataset/sub10_latent_ceiling
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))

import numpy as np
import torch
import trimesh
from PIL import Image

import o_voxel
from trellis2.modules.sparse import SparseTensor
from trellis2 import models
from unguided_trellis2 import load_parts, normalization, normalize, bake_to_uv, to_u8, dilate

LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder with <object>/<hdr>_rot<deg>.glb + pbr.glb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()

    root = "microsoft/TRELLIS.2-4B"
    from huggingface_hub import snapshot_download
    snap = snapshot_download(root, allow_patterns=["ckpts/tex_enc_*", "ckpts/tex_dec_*"])
    enc = models.from_pretrained(f"{snap}/ckpts/tex_enc_next_dc_f16c32_fp16").cuda().eval()
    dec = models.from_pretrained(f"{snap}/ckpts/tex_dec_next_dc_f16c32_fp16").cuda().eval()

    for baked in sorted(Path(args.data).glob("*/*_rot*.glb")):
        t0 = time.time()
        name = f"{baked.parent.name}__{baked.stem}"
        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        gt_path = baked.parent / "pbr.glb"
        parts = load_parts(gt_path)
        center, scale = normalization(np.concatenate([p["mesh"].vertices for p in parts]))

        # ground truth in TRELLIS space -> PBR voxels (same voxelizer as TRELLIS.2's data pipeline)
        scene = trimesh.load(gt_path, force="scene", process=False)
        m = np.eye(4)
        m[:3, 3] = -center
        s = np.diag([scale, scale, scale, 1.0])
        r = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])  # (x, y, z) -> (x, -z, y)
        scene.apply_transform(r @ s @ m)
        coords, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
            scene, grid_size=args.resolution, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        feats = torch.cat([attr[k] for k in LAYOUT], -1).float() / 255 * 2 - 1
        x = SparseTensor(feats.cuda(), torch.cat([torch.zeros_like(coords[:, :1]), coords], -1).int().cuda())

        with torch.no_grad():
            z = enc(x)
            rec = dec(z) * 0.5 + 0.5
        n_lat, n_vox = z.coords.shape[0], rec.coords.shape[0]

        # bake into the baked input's UV layout (identical to the ground truth's) and write baseline outputs
        bparts = load_parts(baked)
        atlases = {}
        for k, p in enumerate(bparts):
            atlases.setdefault(p["tex_key"], []).append(k)
        atlas_meta = []
        for a, (key, idx) in enumerate(atlases.items()):
            h, w = bparts[idx[0]]["texture"].shape[:2]
            bc = np.zeros((h, w, 3), np.float32); ro = np.zeros((h, w), np.float32); me = np.zeros((h, w), np.float32)
            cover = np.zeros((h, w), bool)
            for k in idx:
                q = bparts[k]
                v = normalize(np.asarray(q["mesh"].vertices), center, scale)
                maps = bake_to_uv(rec, args.resolution, v, np.asarray(q["mesh"].faces), q["uv"], h, w, LAYOUT)
                mk = maps["mask"]
                bc[mk], ro[mk], me[mk] = maps["basecolor"][mk], maps["roughness"][mk], maps["metallic"][mk]
                cover |= mk
            Image.fromarray(to_u8(bc)).save(out_dir / f"tex{a}_basecolor.png")
            Image.fromarray(to_u8(ro)).save(out_dir / f"tex{a}_roughness.png")
            Image.fromarray(to_u8(me)).save(out_dir / f"tex{a}_metallic.png")
            Image.fromarray(to_u8(cover.astype(np.float32))).save(out_dir / f"tex{a}_mask.png")
            atlas_meta.append({"index": a, "size": [h, w], "parts": [bparts[k]["name"] for k in idx]})
            bparts_mat = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.fromarray(dilate(to_u8(bc), cover)),
                metallicRoughnessTexture=Image.fromarray(np.stack(
                    [np.zeros((h, w), np.uint8), dilate(to_u8(ro), cover), dilate(to_u8(me), cover)], -1)),
                metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
            for k in idx:
                bparts[k]["material"] = bparts_mat
        glb = []
        for p in bparts:
            mesh = p["mesh"].copy()
            mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=p["material"])
            glb.append(mesh)
        trimesh.Scene(glb).export(out_dir / "textured.glb")
        meta = {"input": str(baked), "method": "latent_ceiling", "parts": [p["name"] for p in bparts],
                "atlases": atlas_meta, "n_pbr_voxels": int(feats.shape[0]), "n_latent": int(n_lat),
                "n_decoded_voxels": int(n_vox), "total_s": round(time.time() - t0, 1)}
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{name[:8]}…{baked.stem}: {feats.shape[0]} PBR voxels -> {n_lat} latents -> {n_vox} decoded, "
              f"{meta['total_s']} s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
