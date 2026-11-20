"""
Detail recovery: put the input texture's texel-level detail back into a predicted base color.

The voxel material prior (TRELLIS.2 at 512^3 / 1024^3) blurs details smaller than a voxel, while the baked input
texture has them, multiplied by baked shading. Per texel of the input texture:

  1. surface point, smooth normal and the order-2 SH transfer of the cosine-weighted visibility (shadow maps, as
     the rebake loss: unbake/guided_trellis2.visibility_transfer),
  2. lighting L fitted to the observed texels with the prediction's base color c_pred (robust least squares),
  3. shading-normalized albedo D = o / (t . L)   (linear color),
  4. only D's high frequencies are kept, since its low frequencies contain whatever shading the model misses:
        c = c_pred * clamp(D / blur_sigma(D), 1 / f_max, f_max)
     with a blur that stays inside the UV islands (normalized convolution with the island mask).

mode="full" returns D itself (scaled per channel to c_pred's mean) for comparison.

    python -m unbake.detail --input examples/cauldron_baked.glb --pred <prediction.glb> --out <out.glb> --sigma 4
"""
import argparse

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

import unbake.guided_trellis2 as G
from unbake.evaluation.metrics import material_maps, match_materials
from unbake.gltf_io import Glb
from unbake.uv_bake import TrellisFrame, _rasterize_uv, merge, to_gltf_images

GAMMA = 2.2


def masked_blur(img: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur restricted to the mask (normalized convolution); img (H, W, C), mask (H, W) bool."""
    m = mask.astype(np.float32)
    k = int(np.ceil(3 * sigma)) * 2 + 1
    num = cv2.GaussianBlur(img * m[..., None], (k, k), sigma, borderType=cv2.BORDER_CONSTANT)
    den = cv2.GaussianBlur(m, (k, k), sigma, borderType=cv2.BORDER_CONSTANT)
    return num / np.maximum(den, 1e-6)[..., None]


@torch.no_grad()
def texel_transfer(glb: Glb, frame: TrellisFrame, material: int, size, vis_dirs=256, vis_res=2048, resolution=1024):
    """Per texel of `material` (texture size `size`): mask (H, W) and SH9 transfer with visibility (H*W covered, 9)."""
    prims = glb.world_primitives()
    group = [p for p in prims if p.material == material]
    pos, uvs, faces = merge(group, frame)
    rast, ft = _rasterize_uv(uvs, faces, size)
    mask = rast[0, ..., 3] > 0
    tp = dr.interpolate(torch.from_numpy(pos).cuda()[None], rast, ft)[0][0][mask]
    if all(p.normals is not None for p in group):
        vn = torch.from_numpy(np.concatenate([G.frame_normals(p.normals) for p in group])).float().cuda()
        n = F.normalize(dr.interpolate(vn[None], rast, ft)[0][0][mask], dim=-1)
    else:
        p0, p1, p2 = (torch.from_numpy(pos).cuda()[ft[:, k].long()] for k in range(3))
        n = F.normalize(torch.cross(p1 - p0, p2 - p0, dim=-1), dim=-1)[(rast[0, ..., 3].long() - 1)[mask]]
    all_pos, _, all_faces = merge(prims, frame)  # occluders: the whole mesh
    T, _ = G.visibility_transfer(tp, n, all_pos, all_faces, resolution, vis_dirs, vis_res)
    return mask.cpu().numpy(), T


def recover(inp: Glb, pred: Glb, mode="highpass", sigma=4.0, f_max=2.0, robust_iters=4):
    """Returns {pred material: new base color (H, W, 3) sRGB in [0, 1]}."""
    frame = TrellisFrame(np.concatenate([p.positions for p in inp.world_primitives()]))
    match = match_materials(inp, pred)  # input material -> prediction material
    out = {}
    for mi, mp in match.items():
        img = inp.material_base_color_image(mi)
        if img is None:
            continue
        size = inp.image(img).size
        o = material_maps(inp, mi, size)["base_color"]  # (H, W, 3) sRGB, baked
        c = material_maps(pred, mp, size)["base_color"]  # (H, W, 3) sRGB, prediction in the same UV layout
        mask, T = texel_transfer(inp, frame, mi, size)
        o_t = torch.from_numpy(o[mask] ** GAMMA).cuda()
        c_t = torch.from_numpy(np.clip(c[mask], 1e-4, 1) ** GAMMA).cuda()
        # robust per-channel least squares for L (9 x 3): o = c * (T . L)
        w = torch.ones(len(o_t), device="cuda")
        for it in range(robust_iters + 1):
            L, res = [], []
            for ch in range(3):
                A = c_t[:, ch:ch + 1] * T
                Aw = A * w[:, None]
                l = torch.linalg.solve(Aw.T @ A + 1e-4 * w.sum() * torch.eye(9, device="cuda"), Aw.T @ o_t[:, ch])
                L.append(l)
                res.append(A @ l - o_t[:, ch])
            r = torch.stack(res, -1).abs().mean(-1)
            s = (2.5 * 1.4826 * r.median()).clamp(min=1e-6)
            w = (s * s / (s * s + r * r)) ** 2
        S = (T @ torch.stack(L, -1)).clamp(min=1e-3)  # (N, 3) shading
        D = np.zeros_like(o)
        D[mask] = (o_t / S).cpu().numpy()
        c_lin = np.clip(c, 1e-4, 1) ** GAMMA
        if mode == "full":
            scale = c_lin[mask].mean(0) / np.maximum(D[mask].mean(0), 1e-6)
            new = D * scale
        else:
            factor = D / np.maximum(masked_blur(D, mask, sigma), 1e-6)
            new = c_lin * np.clip(factor, 1 / f_max, f_max)
        new = np.where(mask[..., None], np.clip(new, 0, 1) ** (1 / GAMMA), c)
        out[mp] = new.astype(np.float32)
    return out


def write(pred_path: str, new_base: dict, out_path: str):
    """The prediction with its base color textures replaced (metallic / roughness unchanged)."""
    from PIL import Image
    glb = Glb(pred_path)
    for mp, bc in new_base.items():
        pbr = glb.json["materials"][mp]["pbrMetallicRoughness"]
        mr = glb.image(glb.json["textures"][pbr["metallicRoughnessTexture"]["index"]]["source"]).convert("RGB")
        glb.set_pbr_textures(mp, Image.fromarray((bc * 255 + 0.5).astype(np.uint8), "RGB"), mr)
    glb.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="baked-lighting .glb")
    ap.add_argument("--pred", required=True, help="prediction .glb (input's UV layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="highpass", choices=("highpass", "full"))
    ap.add_argument("--sigma", type=float, default=4.0, help="high-pass blur in texels")
    ap.add_argument("--f-max", type=float, default=2.0, help="clamp of the detail factor")
    args = ap.parse_args()
    write(args.pred, recover(Glb(args.input), Glb(args.pred), args.mode, args.sigma, args.f_max), args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
