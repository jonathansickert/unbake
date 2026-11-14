"""De-lighting: base color = baked color / estimated shading, with the scale and tint from a prior.

The rebake model (Lambert, SH lighting, visibility) explains the bakes well when the base color is right,
but the rebake loss cannot fix the overall brightness and tint of the base color (the lighting is refit).
Here a prior result (e.g. an ensemble of optimized TRELLIS.2 samples) provides exactly that: the lighting
is fitted with the prior base color at every covered texel, and the base color is then recovered at full
texture resolution as baked / (T @ L). Where the shading is weak (deep shadow) the division is unreliable,
so the result falls back to the prior: weight w = clamp(E / (--shadow * median E), 0, 1) per texel.

    python method/delight.py --prior results/opt_ensemble --out results/delight

Roughness / metallic are set to 0.5 / 0. Output in the baseline format (with textured.glb), so
baselines/evaluate_uv.py and baselines/evaluate_relit.py apply.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))

import numpy as np
import torch
from PIL import Image

from rebake import Asset, srgb_to_linear, linear_to_srgb, fit_lighting, sample_maps, load_result_maps, cosine_rays, sh_basis_order
from ensemble import write_glb
from unguided_trellis2 import to_u8


def masked_blur(img, mask, sigma):
    """Gaussian blur of [H, W, C] inside mask (normalized convolution, so nothing bleeds in from outside)."""
    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=img.device, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    k = (k / k.sum())[None, None]
    def conv(x):  # [C, H, W], separable
        c = x.shape[0]
        x = torch.nn.functional.conv2d(x[:, None], k[..., None], padding=(r, 0))
        return torch.nn.functional.conv2d(x, k[:, :, None, :], padding=(0, r))[:, 0]
    m = mask[None].float()
    num = conv(img.permute(2, 0, 1) * m)
    den = conv(m).clamp(min=1e-6)
    return (num / den).permute(1, 2, 0)


@torch.no_grad()
def split_frequencies(asset, tx, prior, div, sigma):
    """Low frequencies from the prior, details from the division: blur(prior) * div / blur(div), per atlas in UV."""
    out = prior.clone()
    for a, (key, idx) in enumerate(asset.atlases.items()):
        h, w = asset.parts[idx[0]]["texture"].shape[:2]
        sel = tx["atlas"] == a
        y, x = tx["yx"][sel, 0], tx["yx"][sel, 1]
        mask = torch.zeros(h, w, dtype=torch.bool, device="cuda"); mask[y, x] = True
        P = torch.zeros(h, w, 3, device="cuda"); P[y, x] = prior[sel]
        D = torch.zeros(h, w, 3, device="cuda"); D[y, x] = div[sel]
        s = sigma * max(h, w) / 1024
        out[sel] = (masked_blur(P, mask, s) * D / masked_blur(D, mask, s).clamp(min=1e-4))[y, x]
    return out.clamp(0, 1)


@torch.no_grad()
def transfer_order(asset, pos, nrm, rays, lmax, chunk=4096):
    """SH transfer with visibility up to degree lmax, in chunks of points (cosine_rays is not chunked)."""
    out = []
    for s in range(0, len(pos), chunk):
        d, vis = cosine_rays(asset, pos[s:s + chunk], nrm[s:s + chunk], rays, seed=s)
        out.append((sh_basis_order(d, lmax) * vis[..., None]).mean(1))
    return torch.cat(out)


@torch.no_grad()
def delight(asset, prior_dir, rays, shadow, split=0.0, sh_order=2):
    tx = asset.texels(max_points=1 << 40)
    T = asset.transfer(tx["pos"], tx["nrm"], rays) if sh_order == 2 else transfer_order(asset, tx["pos"], tx["nrm"], rays, sh_order)
    bc, _, _ = sample_maps(load_result_maps(prior_dir, asset), tx)
    kd = srgb_to_linear(bc.clamp(0, 1))
    c = srgb_to_linear(tx["baked"])
    L = fit_lighting(kd, c, T)
    E = T @ L                                            # [N, 3] linear shading
    e = E.mean(1)
    w = (e / (shadow * e.median().clamp(min=1e-6))).clamp(0, 1)[:, None]
    div = (c / E.clamp(min=1e-4)).clamp(0, 1)
    if split > 0:
        div = split_frequencies(asset, tx, kd, div, split)
    out = linear_to_srgb(w * div + (1 - w) * kd)
    return tx, out, float(w.mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prior", required=True, help="results folder with the prior base color per input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rays", type=int, default=64)
    ap.add_argument("--shadow", type=float, default=0.3, help="shading (relative to the median) below which the prior takes over")
    ap.add_argument("--split", type=float, default=0.0, help="Gaussian sigma in texels (at 1024): low frequencies from the prior, details from the division; 0 = off")
    ap.add_argument("--sh-order", type=int, default=2, help="SH degree of the lighting")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    prior = Path(args.prior)
    for meta_path in sorted(prior.glob("*/meta.json")):
        name = meta_path.parent.name
        if args.only and not any(name.startswith(o) for o in args.only.split(",")):
            continue
        out = Path(args.out) / name
        if (out / "meta.json").exists():
            continue
        t0 = time.time()
        meta = json.loads(meta_path.read_text())
        asset = Asset(meta["input"])
        tx, bc, wmean = delight(asset, meta_path.parent, args.rays, args.shadow, args.split, args.sh_order)
        out.mkdir(parents=True, exist_ok=True)
        atlas_meta = []
        for a, (key, idx) in enumerate(asset.atlases.items()):
            h, w = asset.parts[idx[0]]["texture"].shape[:2]
            img = np.zeros((h, w, 3), np.float32)
            cover = np.zeros((h, w), bool)
            sel = (tx["atlas"] == a).cpu().numpy()
            yx = tx["yx"].cpu().numpy()[sel]
            img[yx[:, 0], yx[:, 1]] = bc.cpu().numpy()[sel]
            cover[yx[:, 0], yx[:, 1]] = True
            Image.fromarray(to_u8(img)).save(out / f"tex{a}_basecolor.png")
            Image.fromarray(np.full((h, w), 128, np.uint8)).save(out / f"tex{a}_roughness.png")
            Image.fromarray(np.zeros((h, w), np.uint8)).save(out / f"tex{a}_metallic.png")
            Image.fromarray(to_u8(cover.astype(np.float32))).save(out / f"tex{a}_mask.png")
            atlas_meta.append({"index": a, "size": [h, w], "parts": [asset.parts[k]["name"] for k in idx]})
        m = {"input": meta["input"], "method": "delight", "prior": str(meta_path.parent), "atlases": atlas_meta,
             "rays": args.rays, "shadow": args.shadow, "split": args.split, "sh_order": args.sh_order, "division_weight_mean": round(wmean, 4),
             "total_s": round(time.time() - t0, 1)}
        (out / "meta.json").write_text(json.dumps(m, indent=2))
        write_glb(m, out)
        print(f"{name[:8]}…{name.split('__')[-1][:20]}: division weight {wmean:.2f}, {m['total_s']} s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
