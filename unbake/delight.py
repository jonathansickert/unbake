"""
De-lighting: the base color at the texture's own resolution as the baked color divided by the shading, with
the lighting fitted on a prior prediction.

The rebake model (guided_trellis2.py) explains the bakes well once the base color is right, but it cannot fix
the base color's overall brightness and tint, since the lighting is refit. A prior prediction, e.g. an
ensemble of guided samples (ensemble.py), provides them: the lighting L (order-2 SH, 9 x 3) is fitted with the
prior's base color c_p at every covered texel, and then, in linear color,

    c = o / (t^T L),    t: SH transfer with visibility (guided_trellis2.visibility_transfer)

per texel, so details smaller than a voxel come back. Where the shading is weak (deep shadow) the division is
unreliable and the prior takes over: c <- w c + (1 - w) c_p with w = clamp(e / (shadow * median e), 0, 1),
e = mean shading. With --split, low frequencies come from the prior and only details from the division:
c <- blur(c_p) * c / blur(c), a Gaussian of --split texels (at 1024 px) inside the UV islands.
Roughness and metallic are set to 0.5 / 0, or kept from the prior with --keep-rm.

    python -m unbake.delight --prior output/predictions/guided_ensemble --output output/predictions/guided_delight
"""
import argparse
import json
import os
import time
from pathlib import Path

import nvdiffrast.torch as dr
import numpy as np
import torch
import torch.nn.functional as F

from unbake.evaluation.metrics import material_maps
from unbake.gltf_io import Glb
from unbake.guided_trellis2 import GAMMA, visibility_transfer
from unbake.predictions import INPUT_AS_ALBEDO_CONSTANTS, PREDICTION_FILE
from unbake.uv_bake import TrellisFrame, _rasterize_uv, merge, to_gltf_images


@torch.no_grad()
def texels(prims, frame, size):
    """Covered texels of a material's primitives at its texture size: mask (H, W), surface point and face
    normal (N, 3) in TRELLIS space, in mask order."""
    positions, uvs, faces = merge(prims, frame)
    rast, faces_t = _rasterize_uv(uvs, faces, size)
    tri = rast[0, ..., 3].long() - 1
    mask = tri >= 0
    pos_t = torch.from_numpy(positions).cuda()
    p0, p1, p2 = (pos_t[faces_t[:, k].long()] for k in range(3))
    face_n = F.normalize(torch.cross(p1 - p0, p2 - p0, dim=-1), dim=-1)
    pos = dr.interpolate(pos_t[None], rast, faces_t)[0][0][mask]
    return mask, pos, face_n[tri[mask]]


def masked_blur(img, mask, sigma):
    """Gaussian blur of (H, W, C) inside mask (normalized convolution, nothing bleeds in from outside)."""
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=img.device, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())[None, None]

    def conv(x):  # (C, H, W), separable
        x = F.conv2d(x[:, None], k[..., None], padding=(r, 0))
        return F.conv2d(x, k[:, :, None, :], padding=(0, r))[:, 0]
    m = mask[None].float()
    return (conv(img.permute(2, 0, 1) * m) / conv(m).clamp(min=1e-6)).permute(1, 2, 0)


@torch.no_grad()
def delight(baked: Glb, prior: Glb, args):
    """Base color maps {material: (H, W, 3) sRGB} and the mean division weight."""
    prims = baked.world_primitives()
    frame = TrellisFrame(np.concatenate([p.positions for p in prims]))
    materials = sorted({p.material for p in prims if p.material is not None})
    per_mat, pos, nrm, obs, kd = {}, [], [], [], []
    for material in materials:
        img = baked.material_base_color_image(material)
        size = baked.image(img).size if img is not None else (1024, 1024)
        mask, p, n = texels([q for q in prims if q.material == material], frame, size)
        o = torch.from_numpy(material_maps(baked, material, size)["base_color"]).cuda()
        c = torch.from_numpy(material_maps(prior, material, size)["base_color"]).cuda()
        per_mat[material] = (mask, c, len(p))
        pos.append(p); nrm.append(n)
        obs.append(o[mask].clamp(0, 1) ** GAMMA); kd.append(c[mask].clamp(1e-4, 1) ** GAMMA)
    pos, nrm, obs, kd = torch.cat(pos), torch.cat(nrm), torch.cat(obs), torch.cat(kd)
    positions, _, faces = merge(prims, frame)
    T, _ = visibility_transfer(pos, nrm, positions, faces, args.resolution, args.dirs, args.map_res)

    light = []  # per channel: min || (c_p,k * T) L_k - o_k ||^2, as in guided_trellis2.Rebake
    for ch in range(3):
        A = kd[:, ch:ch + 1] * T
        light.append(torch.linalg.solve(A.T @ A + args.ridge * len(A) * torch.eye(9, device=A.device), A.T @ obs[:, ch]))
    E = (T @ torch.stack(light, -1)).clamp(min=1e-4)
    e = E.mean(1)
    w = (e / (args.shadow * e.median().clamp(min=1e-6))).clamp(0, 1)[:, None]
    div = (obs / E).clamp(0, 1)

    out, start = {}, 0
    for material in materials:
        mask, c, n = per_mat[material]
        d, k, wm = div[start:start + n], kd[start:start + n], w[start:start + n]
        start += n
        if args.split > 0:
            h, wd = mask.shape
            s = args.split * max(h, wd) / 1024
            D = torch.zeros(h, wd, 3, device="cuda"); D[mask] = d
            P = torch.zeros(h, wd, 3, device="cuda"); P[mask] = k
            d = (masked_blur(P, mask, s) * D / masked_blur(D, mask, s).clamp(min=1e-4))[mask].clamp(0, 1)
        bc = c.clone()  # outside the UV islands: the prior (keeps its padding)
        bc[mask] = (wm * d + (1 - wm) * k) ** (1 / GAMMA)
        out[material] = bc.cpu().numpy()
    return out, float(w.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prior", required=True, help="prediction folder with the prior base color per input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", default=None, help="comma-separated item id prefixes")
    parser.add_argument("--shadow", type=float, default=0.3, help="shading (relative to the median) below which the prior takes over")
    parser.add_argument("--split", type=float, default=4.0, help="Gaussian sigma in texels at 1024 px; 0 = off")
    parser.add_argument("--dirs", type=int, default=256, help="visibility directions")
    parser.add_argument("--map-res", type=int, default=2048, help="shadow map resolution")
    parser.add_argument("--resolution", type=int, default=1024, help="voxel size for the normal offset of the visibility test")
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--keep-rm", action="store_true", help="keep the prior's roughness / metallic")
    args = parser.parse_args()

    for meta_path in sorted(Path(args.prior).glob("*/meta.json")):
        name = meta_path.parent.name
        if args.only and not any(name.startswith(o) for o in args.only.split(",")):
            continue
        out = Path(args.output) / name
        if (out / "meta.json").exists():
            continue
        t0 = time.time()
        meta = json.loads(meta_path.read_text())
        prior = Glb(str(meta_path.parent / PREDICTION_FILE))
        glb = Glb(meta["input"])
        maps, wmean = delight(glb, prior, args)
        for material, bc in maps.items():
            h, w = bc.shape[:2]
            pm = material_maps(prior, material, (w, h))
            rm = {k: pm[k] if args.keep_rm else np.full((h, w), INPUT_AS_ALBEDO_CONSTANTS[k], np.float32)
                  for k in ("roughness", "metallic")}
            glb.set_pbr_textures(material, *to_gltf_images({"base_color": bc, **rm}))
        os.makedirs(out, exist_ok=True)
        glb.save(str(out / PREDICTION_FILE))
        (out / "meta.json").write_text(json.dumps({
            "input": meta["input"], "method": "delight", "prior": str(meta_path.parent), "shadow": args.shadow,
            "split": args.split, "dirs": args.dirs, "keep_rm": args.keep_rm, "division_weight_mean": round(wmean, 4),
            "time_s": round(time.time() - t0, 1)}, indent=2))
        print(f"{name[:8]}…{name.split('__')[-1][:20]}: division weight {wmean:.2f}, {time.time() - t0:.0f} s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
