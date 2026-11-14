"""Helpers shared by latent_opt.py and guided_sampling.py."""
import json

import numpy as np
import torch
import trimesh
from PIL import Image

from unguided_trellis2 import bake_to_uv, to_u8, dilate


def set_block_checkpointing(dec, on):
    """Recompute every decoder block in the backward pass instead of storing its activations."""
    from torch.utils.checkpoint import checkpoint
    for res in dec.blocks:
        for block in res:
            if not hasattr(block, "_orig_forward"):
                block._orig_forward = block.forward
            if on:
                block.forward = (lambda f: lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k))(block._orig_forward)
            else:
                block.forward = block._orig_forward


def voxel_index(coords, pos, res):
    """Index of the decoded voxel containing each point (-1 if none). coords: [M, 4] (batch, x, y, z)."""
    c = coords[:, 1:].long()
    key = (c[:, 0] * res + c[:, 1]) * res + c[:, 2]
    order = torch.argsort(key)
    skey = key[order]
    q = ((pos + 0.5) * res).floor().long().clamp(0, res - 1)
    qk = (q[:, 0] * res + q[:, 1]) * res + q[:, 2]
    i = torch.searchsorted(skey, qk).clamp(max=len(skey) - 1)
    return torch.where(skey[i] == qk, order[i], torch.full_like(i, -1))


def write_result(asset, pbr_voxel, res, layout, out_dir, baked, lam, args):
    atlas_meta, mats = [], {}
    for a, (key, idx) in enumerate(asset.atlases.items()):
        h, w = asset.parts[idx[0]]["texture"].shape[:2]
        bc = np.zeros((h, w, 3), np.float32); ro = np.zeros((h, w), np.float32); me = np.zeros((h, w), np.float32)
        cover = np.zeros((h, w), bool)
        for k in idx:
            m = bake_to_uv(pbr_voxel, res, asset.v[k], asset.f[k], asset.parts[k]["uv"], h, w, layout)
            mk = m["mask"]
            bc[mk], ro[mk], me[mk] = m["basecolor"][mk], m["roughness"][mk], m["metallic"][mk]
            cover |= mk
        for n, x in [("basecolor", bc), ("roughness", ro), ("metallic", me)]:
            Image.fromarray(to_u8(x)).save(out_dir / f"tex{a}_{n}.png")
        Image.fromarray(to_u8(cover.astype(np.float32))).save(out_dir / f"tex{a}_mask.png")
        mats[key] = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(dilate(to_u8(bc), cover)),
            metallicRoughnessTexture=Image.fromarray(np.stack([np.zeros((h, w), np.uint8), dilate(to_u8(ro), cover),
                                                               dilate(to_u8(me), cover)], -1)),
            metallicFactor=1.0, roughnessFactor=1.0, doubleSided=True)
        atlas_meta.append({"index": a, "size": [h, w], "parts": [asset.parts[k]["name"] for k in idx]})
    glb = []
    for p in asset.parts:
        mesh = p["mesh"].copy()
        mesh.visual = trimesh.visual.TextureVisuals(uv=p["uv"], material=mats[p["tex_key"]])
        glb.append(mesh)
    trimesh.Scene(glb).export(out_dir / "textured.glb")
    (out_dir / "meta.json").write_text(json.dumps({"input": str(baked), "method": f"latent_opt_lambda{lam}",
                                                   "atlases": atlas_meta, "steps": args.steps, "lr": args.lr,
                                                   "anchor": getattr(args, "anchor", 0.0)}, indent=2))
