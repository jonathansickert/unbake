"""
Relighting evaluation for explicit GLBs (assets outside the dataset, e.g. the examples/ cauldron): the
ground truth, input as albedo and every given prediction rendered exactly as in evaluate_relit.py (TRELLIS.2 PBR
renderer with base color / roughness / metallic, the 8 held-out views, the held-out env maps of env_maps/eval/,
seeded SSAO) and scored with the same metrics (LPIPS on the full frame, PSNR / SSIM / MAE on object pixels), plus
the UV metrics of evaluate_uv.py. Also writes a figure: one row per env map (view --view), columns input (baked,
unlit) | ground truth | methods, with each method's LPIPS under its tile (mean over the 8 views of that env map;
best per row in bold).

    python -m unbake.evaluation.relit_glbs --gt examples/cauldron_pbr.glb --input examples/cauldron_baked.glb \
        --glb "Unguided=<out>/unguided.glb" "Guided=<out>/rebake.glb" --out <out>/relit

Writes <out>/metrics.json, <out>/metrics.md and <out>/relit.pdf + .png.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unbake.evaluation.figure import COLUMN_TITLES, on_white
from unbake.evaluation.metrics import RenderMetrics, uv_metrics
from unbake.evaluation.render import (ENV_MAP_MEAN_LUMINANCE, EVAL_ENV_MAPS, NUM_VIEWS, RESOLUTION, held_out_cameras,
                                      load_envmap, render_mesh, render_views)
from unbake.gltf_io import Glb
from unbake.predictions import INPUT_AS_ALBEDO, INPUT_AS_ALBEDO_CONSTANTS
from unbake.reference import render_unlit
from unbake.uv_bake import TrellisFrame

METRICS = ("LPIPS", "PSNR", "SSIM", "MAE")
LOWER_IS_BETTER = {"LPIPS": True, "PSNR": False, "SSIM": False, "MAE": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", required=True, help="ground-truth PBR .glb")
    ap.add_argument("--input", required=True, help="baked-lighting .glb (input as albedo, and the input column)")
    ap.add_argument("--glb", nargs="+", required=True, help="label=path of predictions (same geometry / UVs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", type=int, default=1, help=f"held-out view shown in the figure, 0..{NUM_VIEWS - 1}")
    ap.add_argument("--fig-res", type=int, default=1024, help="render resolution of the figure tiles")
    ap.add_argument("--width", type=float, default=6.5, help="figure width in inches")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gt, baked = Glb(args.gt), Glb(args.input)
    methods = [(COLUMN_TITLES[INPUT_AS_ALBEDO], baked, INPUT_AS_ALBEDO_CONSTANTS, args.input)]
    for spec in args.glb:
        label, path = spec.split("=", 1)
        methods.append((label.replace("\\n", "\n"), Glb(path), None, path))

    frame = TrellisFrame(np.concatenate([p.positions for p in gt.world_primitives()]))  # same placement for all
    envs = {os.path.basename(p)[:-4]: load_envmap(p, ENV_MAP_MEAN_LUMINANCE) for p in EVAL_ENV_MAPS}
    cameras = held_out_cameras(NUM_VIEWS)
    scorer = RenderMetrics()

    # ---- metrics, as evaluate_relit.py / evaluate_uv.py
    gt_renders, masks = render_views(render_mesh(gt, frame), envs, cameras, RESOLUTION)
    results = {}
    for title, glb, constants, path in methods:
        renders, _ = render_views(render_mesh(glb, frame, constants), envs, cameras, RESOLUTION)
        per_env = {env: scorer(renders[env], gt_renders[env], masks) for env in envs}
        mean = {k: float(np.mean([per_env[e][k] for e in envs])) for k in METRICS}
        results[title.replace("\n", " ")] = {"path": path, "relit": per_env, "relit_mean": mean,
                                              "uv": uv_metrics(gt, glb, constants)}
        del renders
        torch.cuda.empty_cache()
    del gt_renders, masks

    names = list(results)
    best = {k: (min if LOWER_IS_BETTER[k] else max)(results[n]["relit_mean"][k] for n in names) for k in METRICS}
    uv_best = {c: min(results[n]["uv"][c]["MAE"] for n in names) for c in ("base_color", "roughness", "metallic")}
    fmt = lambda v, b, d: f"**{v:.{d}f}**" if abs(v - b) < 1e-12 else f"{v:.{d}f}"
    lines = ["| method | base color MAE | roughness MAE | metallic MAE | relit LPIPS ↓ | relit PSNR ↑ | relit SSIM ↑ | relit MAE ↓ |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        u, r = results[n]["uv"], results[n]["relit_mean"]
        lines.append(f"| {n} | " + " | ".join(fmt(u[c]["MAE"], uv_best[c], 4) for c in ("base_color", "roughness", "metallic"))
                     + " | " + " | ".join(fmt(r[k], best[k], 2 if k == "PSNR" else 3) for k in METRICS) + " |")
    table = "\n".join(lines)
    with open(os.path.join(args.out, "metrics.md"), "w") as f:
        f.write(f"Ground truth {args.gt}, input {args.input}; relighting: {NUM_VIEWS} held-out views x "
                f"{len(envs)} env maps ({', '.join(envs)}), mean over env maps.\n\n{table}\n")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"gt": args.gt, "input": args.input, "env_maps": EVAL_ENV_MAPS, "num_views": NUM_VIEWS,
                   "resolution": RESOLUTION, "results": results}, f, indent=2)
    print(table)

    # ---- figure: rows = env maps, one view
    res = args.fig_res
    extr, intr = cameras
    cam = ([extr[args.view]], [intr[args.view]])
    yaw_deg, pitch_deg = (args.view + 0.5) * 360 / NUM_VIEWS, 15 if args.view % 2 == 0 else 35  # as held_out_cameras
    inp = np.asarray(render_unlit(baked, frame, yaw_deg, pitch_deg, resolution=res), np.float32) / 255

    def relit(glb, constants=None):
        r, m = render_views(render_mesh(glb, frame, constants), envs, cam, res, seed=args.view)
        return {e: r[e][0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() for e in envs}, m[0].clamp(0, 1).cpu().numpy()

    cols = [("Input\n(baked)", {e: inp[..., :3] for e in envs}, inp[..., 3], None), ("Ground\ntruth", *relit(gt), None)]
    for title, glb, constants, _ in methods:
        cols.append((title, *relit(glb, constants), results[title.replace("\n", " ")]["relit"]))
    ys, xs = np.nonzero(np.max([a for _, _, a, _ in cols], 0) > 0.01)
    mg = round(0.025 * res)
    y0, y1, x0, x1 = max(ys.min() - mg, 0), min(ys.max() + mg, res), max(xs.min() - mg, 0), min(xs.max() + mg, res)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = int(np.clip(cy - side // 2, 0, res - side)), int(np.clip(cx - side // 2, 0, res - side))

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8})
    n_rows, n_cols = len(envs), len(cols)
    tile = (args.width - 0.35) / n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.width, (tile + 0.13) * n_rows + 0.3),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.16}, squeeze=False)
    for r, env in enumerate(envs):
        row_best = min(m[env]["LPIPS"] for _, _, _, m in cols if m is not None)
        for c, (title, imgs, alpha, metr) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(on_white(imgs[env], alpha)[y0:y0 + side, x0:x0 + side], interpolation="antialiased")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 1:
                ax.spines["right"].set_visible(True), ax.spines["right"].set_linestyle(":"), ax.spines["right"].set_color("0.6")
            if r == 0:
                ax.set_title(title, fontsize=7, pad=4)
            if c == 0:
                ax.set_ylabel(env.replace("_2k", "").replace("_", " "), fontsize=6, labelpad=3)
            if metr is not None:
                v = metr[env]["LPIPS"]
                ax.set_xlabel(f"{v:.3f}", fontsize=7, labelpad=1, fontweight="bold" if v == row_best else "normal")
    pdf = os.path.join(args.out, "relit.pdf")
    fig.savefig(pdf, dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(pdf)[0] + ".png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"-> {args.out}/metrics.md, metrics.json, relit.pdf/.png")


if __name__ == "__main__":
    main()
