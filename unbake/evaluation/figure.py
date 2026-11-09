"""
Qualitative comparison figure from the renders of evaluate_relit.py.

    python -m unbake.evaluation.figure --out output/figure.png --rows 5

Each row is one dataset item, for its first lighting condition (--lighting to pick another),
under one eval env map from one view: input (the unlit baked asset) | ground truth | every
method that has renders for that input.
"""
import argparse
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw

from unbake.evaluation.summary import add_eval_args, evaluation_inputs
from unbake.reference import render_unlit
from unbake.uv_bake import TrellisFrame


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_eval_args(ap)
    ap.add_argument("--rows", type=int, default=5, help="number of items")
    ap.add_argument("--lighting", type=int, default=0, help="index of the lighting condition per item")
    ap.add_argument("--env", type=int, default=0, help="index of the eval env map")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.max_items = args.rows

    with open(os.path.join(args.output, "metrics_relit.json")) as f:
        relit = json.load(f)
    res, views = relit["resolution"], relit["num_views"]
    dataset, items, methods = evaluation_inputs(args)
    slug = lambda name: re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    t = args.tile
    crop = lambda path: np.asarray(Image.open(path).convert("RGB").crop(
        (args.view * res, args.env * res, (args.view + 1) * res, (args.env + 1) * res)).resize((t, t)))
    assert args.view < views, f"--view must be < {views}"

    header = ["input", "ground truth"] + list(methods)
    rows = []
    for index in items:
        item = dataset[index]
        lighting = item.lightings[args.lighting]
        render_dir = os.path.join(args.output, "renders", item.id)
        glb = lighting.baked()
        inp = render_unlit(glb, TrellisFrame(np.concatenate([p.positions for p in glb.world_primitives()])), resolution=t)
        bg = Image.new("RGB", (t, t), (0, 0, 0))
        bg.paste(inp, mask=inp.split()[3])
        cells = [np.asarray(bg), crop(os.path.join(render_dir, "ground_truth.png"))]
        for name in methods:
            path = os.path.join(render_dir, slug(name), f"{lighting.name}.png")
            cells.append(crop(path) if os.path.isfile(path) else np.full((t, t, 3), 255, np.uint8))
        rows.append(np.concatenate(cells, 1))

    grid = Image.fromarray(np.concatenate(rows, 0))
    canvas = Image.new("RGB", (grid.width, grid.height + 24), (255, 255, 255))
    canvas.paste(grid, (0, 24))
    draw = ImageDraw.Draw(canvas)
    for i, h in enumerate(header):
        draw.text((i * t + 6, 5), h, fill=(0, 0, 0))
    canvas.save(args.out)
    print(f"{len(rows)} rows, env map {os.path.basename(relit['env_maps'][args.env])}, view {args.view} -> {args.out}")


if __name__ == "__main__":
    main()
