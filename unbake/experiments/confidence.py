"""Feasibility test 5: does the spread of per-view 2D predictions predict their error?

Uses SuperMat's per-view predictions (results/<...>/supermat_pred, written by baselines/supermat.py),
projects them onto the texels of the baked input (depth-tested, as in the back-projection), and
per texel computes the cosine-weighted mean over the views that see it and the weighted standard
deviation sigma. The proposal's confidence weight is w = 1 / (1 + (sigma / tau)^2), tau = median sigma.
Reports per channel: Spearman(sigma, |mean - GT|) and the error of the confident vs. unconfident half.

    python -m unbake.experiments.confidence --pred results/dataset/sub10_supermat --out results/feasibility/confidence.json
"""
import argparse
import json
import sys
from pathlib import Path


import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unbake.method.rebake import Asset, sample_maps, load_gt_maps
import unbake.baselines.supermat as S


def spearman(a, b):
    a, b = a.cpu().numpy(), b.cpu().numpy()
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=60000)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    res = {}
    for meta_path in sorted(Path(args.pred).glob("*/meta.json")):
        d = meta_path.parent
        baked = Path(json.loads(meta_path.read_text())["input"])
        asset = Asset(baked)
        tx = asset.texels(args.points)
        gt_bc, gt_ro, gt_me = sample_maps(load_gt_maps(asset, baked.parent / "pbr.glb"), tx)
        # same view geometry as baselines/supermat.py (positions scaled to the orthographic frame)
        pv = [{"v": v * S.EXTENT, "f": f, "uv": p["uv"], "texture": p["texture"]} for v, f, p in zip(asset.v, asset.f, asset.parts)]
        _, depths = S.render_views(pv, 1024)
        pos, nrm = tx["pos"] * S.EXTENT, tx["nrm"]
        load = lambda n, k: torch.tensor(np.asarray(Image.open(d / "supermat_pred" / f"{n}_{k:04d}.png").convert(
            "RGB" if n == "albedo" else "L").resize((1024, 1024), Image.BILINEAR)), dtype=torch.float32, device="cuda") / 255
        vals, ws = [], []
        for k, (cam, up) in enumerate(S.VIEWS):
            c2w = S.c2w_matrix(cam, up)
            x, y, dep = S.to_view(pos, c2w)
            to_cam = torch.tensor(cam, dtype=torch.float32, device="cuda")
            cos = (nrm @ (to_cam / to_cam.norm())).clamp(min=0)
            px = ((x / 0.5 + 1) / 2 * 1024).long().clamp(0, 1023)
            py = ((1 - (y / 0.5 + 1) / 2) * 1024).long().clamp(0, 1023)
            vis = dep <= depths[k][py, px] + 0.01
            grid = torch.stack([x / 0.5, -(y / 0.5)], -1)[None, None]
            alb = load("albedo", k).mean(-1)  # luminance-like
            m = torch.stack([alb, load("roughness", k), load("metallic", k)], -1)
            vals.append(F.grid_sample(m.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T)
            ws.append(cos ** 2 * vis)
        V, W = torch.stack(vals, 1), torch.stack(ws, 1)  # [N, 6, 3], [N, 6]
        seen2 = (W > 1e-6).sum(1) >= 2
        wn = W / W.sum(1, keepdim=True).clamp(min=1e-8)
        mean = (V * wn[..., None]).sum(1)
        sigma = torch.sqrt(((V - mean[:, None]) ** 2 * wn[..., None]).sum(1))
        gt = torch.stack([gt_bc.mean(-1), gt_ro, gt_me], -1)
        err = (mean - gt).abs()
        r = {"seen_by_2_views": float(seen2.float().mean())}
        for c, ch in enumerate(["albedo", "roughness", "metallic"]):
            s, e = sigma[seen2, c], err[seen2, c]
            conf = s <= s.median()
            r[ch] = {"spearman_sigma_error": spearman(s, e), "err_confident_half": float(e[conf].mean()),
                     "err_unconfident_half": float(e[~conf].mean()), "err_all": float(e.mean())}
        res[d.name] = r
        print(f"{d.name[:8]} seen>=2: {r['seen_by_2_views']:.2f} | " + " | ".join(
            f"{ch}: rho {r[ch]['spearman_sigma_error']:+.2f}, err conf {r[ch]['err_confident_half']:.3f} vs "
            f"unconf {r[ch]['err_unconfident_half']:.3f}" for ch in ["albedo", "roughness", "metallic"]), flush=True)
        torch.cuda.empty_cache()
    summ = {ch: {k: float(np.mean([r[ch][k] for r in res.values()])) for k in res[next(iter(res))][ch]}
            for ch in ["albedo", "roughness", "metallic"]}
    Path(args.out).write_text(json.dumps({"summary": summ, "assets": res}, indent=2))
    print("\nmean over inputs:")
    for ch, s in summ.items():
        print(f"  {ch:9s} Spearman(sigma, error) {s['spearman_sigma_error']:+.2f} | error confident half "
              f"{s['err_confident_half']:.3f} vs unconfident half {s['err_unconfident_half']:.3f}")


if __name__ == "__main__":
    main()
