"""Feasibility tests 2 and 3 for the rebake loss.

Test 2 (model fit): with the ground-truth materials, how well does Lambertian shading under
fitted second-order SH lighting (with / without visibility) reproduce the baked texture?
Test 3 (discrimination): rebake error of several candidate material sets (each with its own
fitted lighting). The ground truth should have the smallest error, and the rebake error should
track the albedo error w.r.t. the ground truth.

    python -m unbake.experiments.rebake_discrimination --data data/unbake_sub10 --out results/feasibility/rebake.json \
        --method trellis=results/dataset/all80_trellis_s0 --method supermat=results/dataset/sub10_supermat ...
"""
import argparse
import json
import sys
from pathlib import Path


import numpy as np
import torch

from unbake.method.rebake import Asset, srgb_to_linear, rebake_error, sample_maps, load_result_maps, load_gt_maps


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", action="append", default=[], help="name=results_folder")
    ap.add_argument("--points", type=int, default=60000)
    ap.add_argument("--rays", type=int, default=128)
    ap.add_argument("--kd", choices=["basecolor", "diffuse"], default="basecolor",
                    help="basecolor: kd = base color (the dataset bakes ignore metallic); diffuse: base color * (1 - metallic)")
    args = ap.parse_args()
    methods = dict(m.split("=", 1) for m in args.method)
    torch.set_grad_enabled(False)

    rows = {}
    for baked in sorted(Path(args.data).glob("*/*_rot*.glb")):
        name = f"{baked.parent.name}__{baked.stem}"
        asset = Asset(baked)
        tx = asset.texels(args.points)
        T_vis = asset.transfer(tx["pos"], tx["nrm"], args.rays, visibility=True)
        T_nov = asset.transfer(tx["pos"], tx["nrm"], args.rays, visibility=False)
        gt_bc, gt_ro, gt_me = sample_maps(load_gt_maps(asset, baked.parent / "pbr.glb"), tx)

        cands = {"ground truth": (gt_bc, gt_me),
                 "input as albedo": (tx["baked"], torch.zeros_like(gt_me)),
                 "constant gray": (torch.full_like(gt_bc, 0.5), torch.zeros_like(gt_me))}
        for m, folder in methods.items():
            d = Path(folder) / name
            if (d / "meta.json").exists():
                bc, ro, me = sample_maps(load_result_maps(d, asset), tx)
                cands[m] = (bc, me)
        r = {}
        for c, (bc, me) in cands.items():
            kd = srgb_to_linear(bc.clamp(0, 1)) * ((1 - me[:, None]) if args.kd == "diffuse" else 1)
            err, _ = rebake_error(kd, tx["baked"], T_vis)
            r[c] = {"rebake_mae": err["mae"], "rebake_psnr": err["psnr"],
                    "albedo_mae": float((bc - gt_bc).abs().mean()),
                    "metallic_mae": float((me - gt_me).abs().mean())}
        # model fit without visibility (ground truth only)
        kd_gt = srgb_to_linear(gt_bc.clamp(0, 1)) * ((1 - gt_me[:, None]) if args.kd == "diffuse" else 1)
        r["ground truth"]["rebake_mae_no_visibility"] = rebake_error(kd_gt, tx["baked"], T_nov)[0]["mae"]
        rows[name] = r
        order = sorted(r, key=lambda c: r[c]["rebake_mae"])
        print(f"{name[:8]}…{baked.stem[:18]:18s} GT rebake {r['ground truth']['rebake_mae']:.4f} "
              f"(no vis {r['ground truth']['rebake_mae_no_visibility']:.4f}) | ranking: " +
              " < ".join(f"{c}({r[c]['rebake_mae']:.3f})" for c in order), flush=True)
        torch.cuda.empty_cache()

    # summary
    cand_names = sorted({c for r in rows.values() for c in r})
    summary = {}
    for c in cand_names:
        vals = [r[c] for r in rows.values() if c in r]
        summary[c] = {k: float(np.mean([v[k] for v in vals])) for k in ["rebake_mae", "albedo_mae", "metallic_mae"]}
        summary[c]["n"] = len(vals)
    summary["ground truth"]["rebake_mae_no_visibility"] = float(np.mean(
        [r["ground truth"]["rebake_mae_no_visibility"] for r in rows.values()]))
    gt_best = [min(r, key=lambda c: r[c]["rebake_mae"]) == "ground truth" for r in rows.values()]
    rho = [spearman([r[c]["rebake_mae"] for c in r], [r[c]["albedo_mae"] for c in r]) for r in rows.values()]
    summary["_ground_truth_lowest_fraction"] = float(np.mean(gt_best))
    summary["_spearman_rebake_vs_albedo_error"] = float(np.mean(rho))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "assets": rows}, indent=2))

    print(f"\n| candidate | rebake MAE (sRGB) | albedo MAE vs GT | metallic MAE vs GT |")
    print("|---|---|---|---|")
    for c in sorted(cand_names, key=lambda c: summary[c]["rebake_mae"]):
        s = summary[c]
        print(f"| {c} | {s['rebake_mae']:.4f} | {s['albedo_mae']:.4f} | {s['metallic_mae']:.4f} |")
    print(f"\nGT rebake MAE without visibility: {summary['ground truth']['rebake_mae_no_visibility']:.4f}")
    print(f"GT has the lowest rebake error on {100 * np.mean(gt_best):.0f}% of inputs; "
          f"mean Spearman(rebake error, albedo error) over candidates: {np.mean(rho):.2f}")


if __name__ == "__main__":
    main()
