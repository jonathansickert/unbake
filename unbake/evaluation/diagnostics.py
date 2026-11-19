"""
Diagnostics of base color errors and per-object statistics, from the UV evaluation.

  * Scale-invariant base color MAE: the prediction is scaled before the MAE by the area-weighted least-squares
    factor against the ground truth, either one factor for all channels (si1) or one per channel (si3). The rebake
    model cannot fix brightness or tint (the lighting is refit), so MAE - si3 MAE is the part of the error from
    the overall brightness and tint and si3 MAE the part from shading left in the albedo and wrong details.
    Also the log ratio of predicted to true mean brightness.
  * Per object (the 8 lightings of an item averaged, n = 10 objects): the difference of every method to a
    reference method, with a 95% bootstrap interval over objects, the exact Wilcoxon signed-rank p-value and the
    number of objects where the method is better.

    python -m unbake.evaluation.diagnostics --data data --predictions output/predictions \
        --uv output/eval/metrics_uv.json --relit output/eval/metrics_relit.json --reference trellis_ensemble_delight
"""
import argparse
import json
import os

import numpy as np

from unbake.evaluation.metrics import match_materials, material_maps
from unbake.evaluation.summary import add_eval_args, evaluation_inputs
from unbake.gltf_io import Glb
from unbake.uv_bake import uv_surface_weights


def scale_invariant(gt: Glb, pred: Glb, constants=None) -> dict:
    gts, preds, ws = [], [], []
    gt_prims = gt.world_primitives()
    for material, pred_material in match_materials(gt, pred).items():
        group = [p for p in gt_prims if p.material == material]
        img = gt.material_base_color_image(material)
        size = gt.image(img).size if img is not None else (1024, 1024)
        w = uv_surface_weights(group, size)
        m = w > 0
        gts.append(material_maps(gt, material, size)["base_color"][m])
        preds.append(material_maps(pred, pred_material, size, constants)["base_color"][m])
        ws.append(w[m])
    g, p, w = np.concatenate(gts), np.concatenate(preds), np.concatenate(ws)[:, None]
    mae = lambda x: float((np.abs(x - g) * w).sum() / (w.sum() * 3))
    s1 = (w * g * p).sum() / max((w * p * p).sum(), 1e-12)
    s3 = (w * g * p).sum(0) / np.maximum((w * p * p).sum(0), 1e-12)
    mean_g, mean_p = (w * g).sum() / w.sum() / 3, (w * p).sum() / w.sum() / 3
    return {"MAE": mae(p), "si1_MAE": mae(np.clip(s1 * p, 0, 1)), "si3_MAE": mae(np.clip(s3 * p, 0, 1)),
            "log_brightness_ratio": float(np.log(max(mean_p, 1e-6) / max(mean_g, 1e-6))),
            "gt_brightness": float(mean_g)}


def per_object(records, key):
    """{method: {item: mean over its lightings}} for a function key(record) -> float."""
    out = {}
    for r in records:
        out.setdefault(r["method"], {}).setdefault(r["item"], []).append(key(r))
    return {m: {i: float(np.mean(v)) for i, v in d.items()} for m, d in out.items()}


def paired(per_obj, method, reference, lower_better=True, boot=10000, seed=0):
    from scipy.stats import wilcoxon
    items = sorted(set(per_obj[method]) & set(per_obj[reference]))
    d = np.array([per_obj[method][i] - per_obj[reference][i] for i in items])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (boot, len(d)))].mean(1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    wins = int((d < 0).sum() if lower_better else (d > 0).sum())
    return {"n": len(d), "diff": float(d.mean()), "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "p": p, "wins": wins}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_eval_args(parser)
    parser.add_argument("--uv", required=True, help="metrics_uv.json")
    parser.add_argument("--relit", default=None, help="metrics_relit.json")
    parser.add_argument("--reference", default="trellis_ensemble_delight")
    args = parser.parse_args()

    out_path = os.path.join(args.output, "diagnostics.json")
    si = json.load(open(out_path)).get("scale_invariant", {}) if os.path.isfile(out_path) else {}
    dataset, items, methods = evaluation_inputs(args)
    for index in items:
        item = dataset[index]
        gt = None
        for lighting in item.lightings:
            for name, locate in methods.items():
                path, constants = locate(item, lighting)
                k = f"{name}|{item.id}|{lighting.name}"
                if k in si or not os.path.isfile(path):
                    continue
                gt = gt or item.ground_truth()
                si[k] = scale_invariant(gt, Glb(path), constants)
        print(f"{item.id[:12]} done", flush=True)
    os.makedirs(args.output, exist_ok=True)

    rows = [{"method": k.split("|")[0], "item": k.split("|")[1], **v} for k, v in si.items()]
    names = [n for n in methods if any(r["method"] == n for r in rows)]
    print("\nBase color, mean over inputs: MAE, scale-invariant MAE (one scale / per channel), brightness log ratio")
    print(f"{'method':<34}{'MAE':>8}{'si1':>8}{'si3':>8}{'log b':>8}")
    for n in names:
        rr = [r for r in rows if r["method"] == n]
        print(f"{n:<34}" + "".join(f"{np.mean([r[c] for r in rr]):8.4f}" for c in ("MAE", "si1_MAE", "si3_MAE", "log_brightness_ratio")))
    for n in names:  # does the prediction get the brightness of dark / bright objects wrong?
        rr = [r for r in rows if r["method"] == n]
        x, y = np.log([r["gt_brightness"] for r in rr]), [r["log_brightness_ratio"] for r in rr]
        rows_slope = np.polyfit(x, y, 1)[0]
        print(f"  {n}: slope of log(pred/gt brightness) over log(gt brightness): {rows_slope:+.2f}")

    stats = {}
    uv = json.load(open(args.uv))["records"]
    metrics = {f"uv {c}": (uv, lambda r, c=c: r["uv"][c]["MAE"], True) for c in ("base_color", "roughness", "metallic")}
    metrics["uv base_color si3"] = (rows, lambda r: r["si3_MAE"], True)
    if args.relit:
        relit = json.load(open(args.relit))["records"]
        for k, lower in (("LPIPS", True), ("PSNR", False), ("SSIM", False)):
            metrics[f"relit {k}"] = (relit, lambda r, k=k: float(np.mean([e[k] for e in r["relit"].values()])), lower)
    print(f"\nPer object (n = objects), difference to {args.reference}: mean [95% bootstrap CI], Wilcoxon p, objects better")
    for label, (recs, key, lower) in metrics.items():
        po = per_object(recs, key)
        if args.reference not in po:
            continue
        print(f"\n{label}")
        for n in names:
            if n == args.reference or n not in po:
                continue
            s = paired(po, n, args.reference, lower)
            stats.setdefault(label, {})[n] = s
            print(f"  {n:<32}{s['diff']:+9.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]  p={s['p']:.3f}  {s['wins']}/{s['n']}")
    json.dump({"reference": args.reference, "scale_invariant": si, "paired": stats}, open(out_path, "w"), indent=1)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
