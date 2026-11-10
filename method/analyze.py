"""Per-object comparison and scale-invariant albedo error for several results folders.

For every asset that all given folders share: base color MAE as evaluated (metrics_uv.json), the
scale-invariant base color MAE (per atlas and color channel the prediction is scaled by the least-squares
factor to the ground truth in linear space before comparing in sRGB, i.e. the error left once the albedo
brightness/tint ambiguity of the rebake loss is factored out) and relit PSNR (metrics_relit.json).

    python method/analyze.py results/dataset/all80_trellis_s0 results/feasibility/latent_opt/lambda0.1 \
        --names TRELLIS "opt 0.1" [--md out.md]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import numpy as np

from evaluate_uv import locate, gt_maps, load_rgb, load_parts, match_parts


def s2l(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def l2s(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def scale_invariant_mae(pred_dir, cache):
    meta_path = pred_dir / "meta.json"
    inp, gt_path, _ = locate(meta_path)
    if gt_path not in cache:
        cache[gt_path] = (gt_maps(gt_path), match_parts(load_parts(inp), load_parts(gt_path)))
    gts, names = cache[gt_path]
    err, n = 0.0, 0
    for atlas in json.loads(meta_path.read_text())["atlases"]:
        gt = gts[frozenset(names.get(x, x) for x in atlas["parts"])]
        h, w = gt["mask"].shape
        m = gt["mask"]
        p = s2l(load_rgb(pred_dir / f"tex{atlas['index']}_basecolor.png", h, w)[m])
        g = s2l(gt["basecolor"][m])
        s = (p * g).sum(0) / np.maximum((p * p).sum(0), 1e-8)
        err += float(np.abs(l2s(p * s) - l2s(g)).mean()) * m.sum()
        n += m.sum()
    return err / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    dirs = [Path(d) for d in args.dirs]
    names = args.names or [d.name for d in dirs]
    uv = [json.loads((d / "metrics_uv.json").read_text())["assets"] for d in dirs]
    rl = [json.loads((d / "metrics_relit.json").read_text())["assets"] if (d / "metrics_relit.json").exists() else {}
          for d in dirs]
    assets = sorted(set.intersection(*[set(u) for u in uv]))
    cache, rows = {}, {}
    for a in assets:
        rows[a] = []
        for d, u, r in zip(dirs, uv, rl):
            try:
                si = scale_invariant_mae(d / a, cache)
            except Exception as e:
                print(f"scale-invariant failed {d.name}/{a[:8]}: {e}", file=sys.stderr)
                si = float("nan")
            rows[a].append((u[a]["pred"]["basecolor"]["mae"], si, r.get(a, {}).get("pred", {}).get("psnr", float("nan"))))
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
