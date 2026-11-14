"""Per-input comparison of several prediction folders, with the scale-invariant base color error.

For every input that all given methods have: base color MAE and relit PSNR from the evaluation
(output/eval/metrics_uv.json and metrics_relit.json of unbake.evaluation, if present), and the
scale-invariant base color MAE computed here: per atlas and color channel the prediction is scaled by the
least-squares factor to the ground truth in linear space before comparing in sRGB, i.e. the error left once
the albedo brightness/tint ambiguity of the rebake loss is factored out (texel-weighted).

    python -m unbake.method.analyze --methods trellis_unguided_1024 rebake_opt rebake_delight [--md out.md]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from unbake.dataset import UnbakeDataset
from unbake.predictions import METHOD_NAMES, PREDICTIONS, PREDICTION_FILE
from unbake.baselines.common import load_parts
from unbake.method.rebake import pbr_maps, match_parts


def s2l(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def l2s(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def scale_invariant_mae(pred_glb, gt_path, cache):
    if gt_path not in cache:
        cache[gt_path] = pbr_maps(gt_path), load_parts(gt_path)
    gts, gt_parts = cache[gt_path]
    names = match_parts(load_parts(pred_glb), gt_parts)
    err, n = 0.0, 0
    for parts, pred in pbr_maps(pred_glb).items():
        gt = gts[frozenset(names.get(x, x) for x in parts)]
        m = gt["mask"]
        h, w = m.shape
        bc = pred["basecolor"]
        if bc.shape[:2] != (h, w):
            from PIL import Image
            bc = np.asarray(Image.fromarray((bc * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), np.float32) / 255
        p, g = s2l(bc[m]), s2l(gt["basecolor"][m])
        s = (p * g).sum(0) / np.maximum((p * p).sum(0), 1e-8)
        err += float(np.abs(l2s(p * s) - l2s(g)).mean()) * m.sum()
        n += m.sum()
    return err / n


def eval_records(path):
    """{(method display name, item id, lighting name): record} of an unbake.evaluation output file."""
    if not os.path.isfile(path):
        return {}
    return {(r["method"], r["item"], r["lighting"]): r for r in json.load(open(path))["records"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+", required=True, help="prediction folder names; the first is the reference")
    ap.add_argument("--data", default="data", help="dataset root")
    ap.add_argument("--predictions", default=PREDICTIONS)
    ap.add_argument("--eval", default="output/eval", help="output folder of unbake.evaluation")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    names = [METHOD_NAMES.get(m, m) for m in args.methods]
    dataset = UnbakeDataset(args.data)
    gt_of = {dataset[i].id: dataset[i].ground_truth_path for i in range(len(dataset))}
    folders = [Path(args.predictions) / m for m in args.methods]
    assets = sorted(set.intersection(*[{p.parent.name for p in f.glob(f"*/{PREDICTION_FILE}")} for f in folders]))
    uv, rl = eval_records(os.path.join(args.eval, "metrics_uv.json")), eval_records(os.path.join(args.eval, "metrics_relit.json"))
    cache, rows = {}, {}
    for a in assets:
        item, lighting = a.split("__", 1)
        rows[a] = []
        for f, name in zip(folders, names):
            try:
                si = scale_invariant_mae(str(f / a / PREDICTION_FILE), gt_of[item], cache)
            except Exception as e:
                print(f"scale-invariant failed {f.name}/{a[:8]}: {e}", file=sys.stderr)
                si = float("nan")
            u, r = uv.get((name, item, lighting)), rl.get((name, item, lighting))
            mae = u["uv"]["base_color"]["MAE"] if u else float("nan")
            psnr = float(np.mean([e["PSNR"] for e in r["relit"].values()])) if r else float("nan")
            rows[a].append((mae, si, psnr))
    out = [f"{len(assets)} assets\n", "### Mean\n", "| method | basecolor MAE | scale-inv. basecolor MAE | relit PSNR |",
           "|---|---|---|---|"]
    allv = np.array([[rows[a][i] for a in assets] for i in range(len(names))])  # [method, asset, metric]
    common = ~np.isnan(allv).any(0)  # [asset, metric]: every method has a value
    for i, n in enumerate(names):
        out.append(f"| {n} | " + " | ".join(f"{allv[i, common[:, k], k].mean():{f}}" if common[:, k].any() else "–"
                                            for k, f in enumerate([".3f", ".3f", ".2f"])) + " |")
    out.append("\nobjects per column: " + ", ".join(str(int(common[:, k].sum())) for k in range(3)))
    if len(names) > 1:
        out += ["", f"### Per object: basecolor MAE / scale-inv. MAE / relit PSNR (better than {names[0]} in bold)\n",
                "| asset | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        for a in assets:
            base = rows[a][0]
            cells = []
            for j, r in enumerate(rows[a]):
                f = [f"{r[0]:.3f}", f"{r[1]:.3f}", f"{r[2]:.1f}"]
                if j:
                    f = [f"**{x}**" if better else x for x, better in
                         zip(f, [r[0] < base[0], r[1] < base[1], r[2] > base[2]])]
                cells.append(" / ".join(f))
            out.append(f"| {a[:8]}…{a.split('__')[-1][:14]} | " + " | ".join(cells) + " |")
        for j in range(1, len(names)):
            w = sum(rows[a][j][0] < rows[a][0][0] for a in assets)
            out.append(f"\n{names[j]}: basecolor MAE better than {names[0]} on {w}/{len(assets)} objects")
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
