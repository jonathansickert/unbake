"""
UV-map figure for the paper: the material maps of one asset side by side, as evaluated by
evaluate_uv.py. Rows: base color, roughness, metallic. Columns: the input (its baked texture) |
ground truth | every method. All maps are shown in the ground truth's UV layout (predictions share
it; unbake.evaluation.metrics.match_materials checks this). Roughness and metallic are grayscale
(black = 0, white = 1); texels outside the UV islands are pale blue, so they are not confused with 1.

    python -m unbake.evaluation.uv_figure                         # output/uv_figure.pdf + .png
    python -m unbake.evaluation.uv_figure --item 796f --lighting 2

Uses the item's material that covers the largest surface area, and as the input its first lighting
condition for which every method has a prediction (as figure.py; --lighting N forces the N-th one).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from unbake.evaluation.figure import COLUMN_TITLES
from unbake.evaluation.metrics import match_materials, material_maps
from unbake.evaluation.summary import add_eval_args, evaluation_inputs
from unbake.gltf_io import Glb
from unbake.paths import OUTPUT
from unbake.predictions import first_complete_lighting
from unbake.uv_bake import uv_surface_weights

ROWS = [("base_color", "Base color"), ("roughness", "Roughness"), ("metallic", "Metallic")]
BACKGROUND = np.array([0.72, 0.83, 0.97])  # outside the UV islands: pale blue, outside the grayscale range


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_eval_args(ap)
    ap.add_argument("--item", default="85e39950", help="item id (or unique prefix)")
    ap.add_argument("--lighting", type=int, default=None,
                    help="index of the lighting condition (input) per item; default: the first one every method has")
    ap.add_argument("--width", type=float, default=5.5, help="figure width in inches (text width of the paper)")
    ap.add_argument("--out", default=os.path.join(OUTPUT, "uv_figure.pdf"), help="also written as .png next to it")
    args = ap.parse_args()

    dataset, _, methods = evaluation_inputs(args)
    hits = [i for i in dataset.ids if i.startswith(args.item)]
    assert len(hits) == 1, f"--item {args.item}: matches {len(hits)} items"
    item = dataset[hits[0]]
    lighting = item.lightings[args.lighting] if args.lighting is not None else first_complete_lighting(item, methods)
    gt = item.ground_truth()

    # the ground-truth material covering the largest surface area, at its texture size
    prims = gt.world_primitives()
    candidates = []
    for material in sorted({p.material for p in prims if p.material is not None}):
        img = gt.material_base_color_image(material)
        size = gt.image(img).size if img is not None else (1024, 1024)
        weights = uv_surface_weights([p for p in prims if p.material == material], size)
        candidates.append((weights.sum(), material, size, weights > 0))
    _, material, size, mask = max(candidates, key=lambda c: c[0])
    ys, xs = np.nonzero(mask)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))  # bounding box of the UV islands

    def maps_of(glb, constants=None):
        return material_maps(glb, match_materials(gt, glb)[material], size, constants)

    baked = lighting.baked()
    columns = [("Input\n(baked)", {"base_color": maps_of(baked)["base_color"]}), ("Ground\ntruth", maps_of(gt))]
    for name, locate in methods.items():
        path, constants = locate(item, lighting)
        if os.path.isfile(path):
            columns.append((COLUMN_TITLES.get(name, name), maps_of(Glb(path), constants)))

    n_cols = len(columns)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8})
    tile = (args.width - 0.35) / n_cols  # room for the row labels
    fig = plt.figure(figsize=(args.width, tile * len(ROWS) * crop_aspect(crop) + 0.35))
    grid = fig.add_gridspec(len(ROWS), n_cols, wspace=0.04, hspace=0.04)
    axes = np.array([[fig.add_subplot(grid[r, c]) for c in range(n_cols)] for r in range(len(ROWS))])
    for r, (channel, label) in enumerate(ROWS):
        for c, (title, maps) in enumerate(columns):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=7, pad=4)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, labelpad=4)
            if channel not in maps:  # the input has no roughness / metallic
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, color="0.5", transform=ax.transAxes)
                continue
            img = np.clip(maps[channel], 0, 1)
            if img.ndim == 2:  # roughness / metallic: grayscale
                img = np.repeat(img[..., None], 3, -1)
            ax.imshow(np.where(mask[..., None], img, BACKGROUND)[crop], interpolation="antialiased")
            ax.spines["left"].set_visible(c == 1), ax.spines["left"].set_color("0.6")  # outline the reference
            for s in ("left", "right", "top", "bottom"):
                ax.spines[s].set_visible(True), ax.spines[s].set_linewidth(0.4), ax.spines[s].set_color("0.75")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=600, bbox_inches="tight", pad_inches=0.02)  # images resampled to 600 dpi, anti-aliased
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"{item.id[:8]} ({lighting.name}), material {material} at {size[0]}x{size[1]}, "
          f"{n_cols - 2} methods -> {args.out}, {png}")


def crop_aspect(crop):
    """height / width of the cropped atlas."""
    return (crop[0].stop - crop[0].start) / (crop[1].stop - crop[1].start)


if __name__ == "__main__":
    main()
