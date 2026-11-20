"""
Material maps in image space: base color, roughness and metallic of the input, the ground truth and every
method, rendered (unshaded) from one held-out view with the evaluation renderer (unbake.evaluation.render,
same cameras as evaluate_relit.py). Rows: base color, roughness, metallic. Columns: input (baked texture,
base color only) | ground truth | methods. Roughness and metallic are grayscale (black = 0, white = 1), on a
white background. Under each method tile: its MAE against the ground truth on the object pixels of that view
(best per row in bold).

Dataset item, methods from a predictions folder (as figure.py / uv_figure.py):

    python -m unbake.evaluation.material_figure --items 85e39950 --predictions output/guided_eval/predictions \
        --out output/guided_eval/materials_cauldron.pdf

Any GLBs with the ground truth's geometry, e.g. the example cauldron runs (label=path):

    python -m unbake.evaluation.material_figure --gt examples/cauldron_pbr.glb --input examples/cauldron_baked.glb \
        --glb "Unguided=output/guided/cauldron_vis_ss_1024/unguided.glb" \
              "Guided=output/guided/cauldron_vis_ss_1024/vis1_ss4_rebake+supermat_sv_b1.glb" \
        --out output/guided/materials_cauldron.pdf

Input as albedo is included in both modes (baked texture as base color, roughness 0.5, metallic 0).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from trellis2.renderers import EnvMap, PbrMeshRenderer

from unbake.evaluation.figure import COLUMN_TITLES
from unbake.evaluation.render import NUM_VIEWS, held_out_cameras, render_mesh
from unbake.evaluation.summary import add_eval_args, evaluation_inputs
from unbake.gltf_io import Glb
from unbake.paths import OUTPUT
from unbake.predictions import INPUT_AS_ALBEDO, INPUT_AS_ALBEDO_CONSTANTS, first_complete_lighting
from unbake.uv_bake import TrellisFrame

ROWS = [("base_color", "Base color"), ("roughness", "Roughness"), ("metallic", "Metallic")]


@torch.no_grad()
def render_materials(glb, frame, extr, intr, resolution, constants=None):
    """{channel: (H, W, 3) in [0, 1]} and the coverage mask (H, W), unshaded."""
    renderer = PbrMeshRenderer({"resolution": resolution, "near": 1, "far": 100, "ssaa": 2, "peel_layers": 1})
    unused_light = EnvMap(torch.ones(64, 128, 3, device="cuda"))  # required by the renderer; channels ignore it
    r = renderer.render(render_mesh(glb, frame, constants), extr, intr, envmap=unused_light)
    mask = r.mask.reshape(resolution, resolution).clamp(0, 1).cpu().numpy()
    out = {"base_color": r.base_color.permute(1, 2, 0).clamp(0, 1).cpu().numpy()}
    for ch in ("roughness", "metallic"):
        x = r[ch].reshape(resolution, resolution).clamp(0, 1).cpu().numpy()
        out[ch] = np.repeat(x[..., None], 3, -1)
    return out, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_eval_args(ap)
    ap.add_argument("--items", default="85e39950", help="dataset item id (or unique prefix), dataset mode")
    ap.add_argument("--lighting", type=int, default=None, help="input index per item (default: first complete one)")
    ap.add_argument("--gt", default=None, help="ground-truth .glb: explicit mode (with --input and --glb)")
    ap.add_argument("--input", default=None, help="baked-lighting .glb (explicit mode)")
    ap.add_argument("--glb", nargs="*", default=[], help="label=path of predictions (explicit mode)")
    ap.add_argument("--view", type=int, default=1, help=f"held-out view, 0..{NUM_VIEWS - 1}")
    ap.add_argument("--render-res", type=int, default=1024)
    ap.add_argument("--width", type=float, default=5.5, help="figure width in inches")
    ap.add_argument("--out", default=os.path.join(OUTPUT, "material_figure.pdf"), help="also written as .png")
    args = ap.parse_args()

    columns = []  # (title, glb, constants)
    if args.gt:
        assert args.input, "explicit mode needs --input"
        gt, baked = Glb(args.gt), Glb(args.input)
        columns.append((COLUMN_TITLES[INPUT_AS_ALBEDO], baked, INPUT_AS_ALBEDO_CONSTANTS))
        for spec in args.glb:
            label, path = spec.split("=", 1)
            columns.append((label.replace("\\n", "\n"), Glb(path), None))
        source = os.path.basename(args.gt)
    else:
        dataset, _, methods = evaluation_inputs(args)
        hits = [i for i in dataset.ids if i.startswith(args.items)]
        assert len(hits) == 1, f"--items {args.items}: matches {len(hits)} items"
        item = dataset[hits[0]]
        lighting = item.lightings[args.lighting] if args.lighting is not None else first_complete_lighting(item, methods)
        gt, baked = item.ground_truth(), lighting.baked()
        for name, locate in methods.items():
            path, constants = locate(item, lighting)
            if os.path.isfile(path):
                columns.append((COLUMN_TITLES.get(name, name), Glb(path), constants))
        source = f"{item.id[:8]} ({lighting.name})"

    frame = TrellisFrame(np.concatenate([p.positions for p in gt.world_primitives()]))
    extr, intr = held_out_cameras(NUM_VIEWS)
    e, i, res = extr[args.view], intr[args.view], args.render_res
    ref, ref_mask = render_materials(gt, frame, e, i, res)
    inp, inp_mask = render_materials(baked, frame, e, i, res)
    tiles = [("Input\n(baked)", {"base_color": inp["base_color"]}, inp_mask, None), ("Ground\ntruth", ref, ref_mask, None)]
    for title, glb, constants in columns:
        maps, mask = render_materials(glb, frame, e, i, res, constants)
        valid = (mask > 0.99) & (ref_mask > 0.99)
        err = {ch: float(np.abs(maps[ch][valid] - ref[ch][valid]).mean()) for ch, _ in ROWS}
        tiles.append((title, maps, mask, err))

    # square crop around the union of the object masks
    ys, xs = np.nonzero(np.max([m for _, _, m, _ in tiles], 0) > 0.01)
    margin = round(0.025 * res)
    y0, y1, x0, x1 = max(ys.min() - margin, 0), min(ys.max() + margin, res), max(xs.min() - margin, 0), min(xs.max() + margin, res)
    side = max(y1 - y0, x1 - x0)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0, x0 = int(np.clip(cy - side // 2, 0, res - side)), int(np.clip(cx - side // 2, 0, res - side))
    crop = (slice(y0, y0 + side), slice(x0, x0 + side))

    n_cols = len(tiles)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8})
    tile = (args.width - 0.35) / n_cols
    fig, axes = plt.subplots(len(ROWS), n_cols, figsize=(args.width, (tile + 0.13) * len(ROWS) + 0.3),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.16}, squeeze=False)
    for r, (channel, label) in enumerate(ROWS):
        best = min(err[channel] for _, _, _, err in tiles if err is not None)
        for c, (title, maps, mask, err) in enumerate(tiles):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=7, pad=4)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, labelpad=4)
            if c == 1:  # separate the references from the methods
                ax.spines["right"].set_visible(True), ax.spines["right"].set_linestyle(":"), ax.spines["right"].set_color("0.6")
            if channel not in maps:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8, color="0.5", transform=ax.transAxes)
                continue
            img = maps[channel] * mask[..., None] + (1 - mask[..., None])  # on white
            ax.imshow(img[crop], interpolation="antialiased")
            if err is not None:
                ax.set_xlabel(f"{err[channel]:.3f}", fontsize=7, labelpad=1,
                              fontweight="bold" if err[channel] == best else "normal")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=600, bbox_inches="tight", pad_inches=0.02)
    png = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.02)
    print(f"{source}: {n_cols - 2} methods, view {args.view} -> {args.out}, {png}")
    for title, _, _, err in tiles[2:]:
        print(f"  {title.replace(chr(10), ' '):34s} " + "  ".join(f"{ch} {err[ch]:.3f}" for ch, _ in ROWS))


if __name__ == "__main__":
    main()
