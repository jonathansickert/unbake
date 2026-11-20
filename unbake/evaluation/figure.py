"""
Qualitative figure for the paper: one row per object, relit under a held-out env map from a
held-out view. Columns: the input (baked asset, unlit) | ground truth | every method. Tiles are
rendered as in evaluate_relit.py (same seeded renderer, view and env map) but at --render-res,
composited on white and cropped to the object. With --metric, each method's relighting LPIPS for
that input is printed under its tile (mean over the views of that env map, from
metrics_relit.json; best per row in bold).

    python -m unbake.evaluation.figure                                  # output/figure.pdf + .png
    python -m unbake.evaluation.figure --items 006d 733b --env 1 --view 3

Each row uses the item's first lighting condition for which every method has a prediction (with
complete methods simply the first one, as items/<item>/comparison.png), so the examples are not
picked for any method; --lighting N forces the N-th one.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unbake.evaluation.render import (ENV_MAP_MEAN_LUMINANCE, EVAL_ENV_MAPS, NUM_VIEWS, RESOLUTION,
                                      held_out_cameras, load_envmap, render_mesh, render_views)
from unbake.evaluation.summary import add_eval_args, evaluation_inputs
from unbake.gltf_io import Glb
from unbake.paths import OUTPUT
from unbake.predictions import INPUT_AS_ALBEDO, first_complete_lighting
from unbake.reference import render_unlit
from unbake.uv_bake import TrellisFrame

DEFAULT_ITEMS = ["006d3159", "733b4416", "85e39950", "9fcdd9db"]
COLUMN_TITLES = {  # two lines each, so that narrow columns don't overlap
    INPUT_AS_ALBEDO: "Input as\nalbedo",
    "TRELLIS.2 (unguided, 512)": "TRELLIS.2\n($512^3$)",
    "TRELLIS.2 (unguided, 1024)": "TRELLIS.2\n($1024^3$)",
    "SuperMat": "SuperMat\n",
    "Material Anything": "Material\nAnything",
    "RGB<->X": "RGB$\\leftrightarrow$X\n",
    "Hybrid (TRELLIS.2 base color + SuperMat r/m)": "TRELLIS.2 +\nSuperMat r/m",
    "Hybrid + detail recovery": "Hybrid +\ndetail",
    "UnBake 1024 + detail recovery": "Guided + vis.\n+ detail",
    "UnBake basic (rebake + SuperMat 16v, beta 1)": "Guided\n($512^3$)",
    "UnBake basic 1024 (rebake + SuperMat 16v, beta 1)": "Guided\n($1024^3$)",
    "UnBake 1024 (rebake + visibility + SuperMat 16v, beta 1)": "Guided + vis.\n($1024^3$)",
}


def on_white(rgb, alpha):
    """(H, W, 3) and (H, W) in [0, 1] -> RGB on a white background."""
    return rgb * alpha[..., None] + (1 - alpha[..., None])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_eval_args(ap)
    ap.add_argument("--items", nargs="+", default=DEFAULT_ITEMS, help="item ids (or unique prefixes), one row each")
    ap.add_argument("--lighting", type=int, default=None,
                    help="index of the lighting condition (input) per item; default: the first one every method has")
    ap.add_argument("--env", type=int, default=1, help=f"index of the eval env map: {[os.path.basename(p) for p in EVAL_ENV_MAPS]}")
    ap.add_argument("--view", type=int, default=1, help=f"held-out view, 0..{NUM_VIEWS - 1}")
    ap.add_argument("--metric", action="store_true", help="print each method's LPIPS under its tile")
    ap.add_argument("--render-res", type=int, default=1536,
                    help=f"render resolution of the tiles before cropping (evaluation: {RESOLUTION})")
    ap.add_argument("--width", type=float, default=5.5, help="figure width in inches (text width of the paper)")
    ap.add_argument("--out", default=os.path.join(OUTPUT, "figure.pdf"), help="also written as .png next to it")
    args = ap.parse_args()

    dataset, _, methods = evaluation_inputs(args)
    res = args.render_res
    env_name = os.path.basename(EVAL_ENV_MAPS[args.env])[:-4]
    envs = {env_name: load_envmap(EVAL_ENV_MAPS[args.env], ENV_MAP_MEAN_LUMINANCE)}
    extr, intr = held_out_cameras(NUM_VIEWS)
    camera = ([extr[args.view]], [intr[args.view]])
    yaw_deg, pitch_deg = (args.view + 0.5) * 360 / NUM_VIEWS, 15 if args.view % 2 == 0 else 35  # as held_out_cameras
    lpips = {}
    if args.metric:
        with open(os.path.join(args.output, "metrics_relit.json")) as f:
            lpips = {(r["item"], r["lighting"], r["method"]): r["relit"][env_name]["LPIPS"] for r in json.load(f)["records"]}

    rows = []
    for prefix in args.items:
        hits = [i for i in dataset.ids if i.startswith(prefix)]
        assert len(hits) == 1, f"--items {prefix}: matches {len(hits)} items"
        item = dataset[hits[0]]
        lighting = item.lightings[args.lighting] if args.lighting is not None else first_complete_lighting(item, methods)
        gt = item.ground_truth()
        frame = TrellisFrame(np.concatenate([p.positions for p in gt.world_primitives()]))

        def relit(glb, constants=None):  # same seed per view as evaluate_relit.py
            renders, masks = render_views(render_mesh(glb, frame, constants), envs, camera, res, seed=args.view)
            return renders[env_name][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(), masks[0].cpu().numpy()

        inp = np.asarray(render_unlit(lighting.baked(), frame, yaw_deg, pitch_deg, resolution=res), np.float32) / 255
        tiles = [("Input\n(baked)", inp[..., :3], inp[..., 3], None), ("Ground\ntruth", *relit(gt), None)]
        for name, locate in methods.items():
            path, constants = locate(item, lighting)
            if os.path.isfile(path):
                tiles.append((COLUMN_TITLES.get(name, name), *relit(Glb(path), constants),
                              lpips.get((item.id, lighting.name, name))))
        # crop all tiles of the row to the union of their object masks (+ margin)
        ys, xs = np.nonzero(np.max([a for _, _, a, _ in tiles], 0) > 0.01)
        m = round(0.025 * res)
        y0, y1, x0, x1 = max(ys.min() - m, 0), min(ys.max() + m, res), max(xs.min() - m, 0), min(xs.max() + m, res)
        side = max(y1 - y0, x1 - x0)  # square crop around the object's center
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        y0, x0 = np.clip(cy - side // 2, 0, res - side), np.clip(cx - side // 2, 0, res - side)
        rows.append([(title, on_white(rgb, a)[y0:y0 + side, x0:x0 + side], metric) for title, rgb, a, metric in tiles])
        print(f"{item.id[:8]}: {lighting.name}, {len(tiles) - 2} methods")

    titles = [t for t, _, _ in rows[0]]
    assert all([t for t, _, _ in r] == titles for r in rows), "every row needs the same methods"
    n_rows, n_cols = len(rows), len(titles)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8})
    tile = args.width / n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(args.width, tile * n_rows + (0.11 * n_rows if lpips else 0) + 0.3),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.14 if lpips else 0.03}, squeeze=False)
    for r, row in enumerate(rows):
        values = [m for _, _, m in row if m is not None]
        for c, (title, img, metric) in enumerate(row):
            ax = axes[r, c]
            ax.imshow(img, interpolation="antialiased")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 1:  # separate the reference columns from the methods
                ax.spines["right"].set_visible(True), ax.spines["right"].set_linestyle(":"), ax.spines["right"].set_color("0.6")
            if r == 0:
                ax.set_title(title, fontsize=7, pad=4)
            if metric is not None:
                ax.set_xlabel(f"{metric:.3f}", fontsize=7, labelpad=1, fontweight="bold" if metric == min(values) else "normal")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # dpi also applies to the images in the PDF: matplotlib resamples every tile once, anti-aliased, to
    # 600 dpi at print size (raw renders would be shrunk by the PDF viewer, often without smoothing)
    fig.savefig(args.out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"{n_rows} rows x {n_cols} columns, env map {env_name}, view {args.view} -> {args.out}, {png}")


if __name__ == "__main__":
    main()
