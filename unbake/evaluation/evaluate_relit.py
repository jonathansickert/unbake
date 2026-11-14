"""
Relighting: every method's prediction and the ground truth rendered with the TRELLIS.2 PBR
renderer (base color / roughness / metallic only) from held-out views under the env maps in
env_maps/eval/, none of which is used to create the dataset. LPIPS on the full frame, PSNR /
SSIM / MAE on object pixels. Every method is scored once per dataset input (item x lighting).

    python -m unbake.evaluation.evaluate_relit                  # every method in output/predictions
    python -m unbake.evaluation.evaluate_relit --max-items 2 --no-renders

Writes to --output:
    metrics_relit.json                            summary + one record per item/lighting/method
    items/<item>/comparison.png                   ground truth vs. methods, first lighting condition
    renders/<item>/ground_truth.png               every relit render at full resolution:
    renders/<item>/<method>/<lighting>.png        rows = env maps, columns = views
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image

from unbake.evaluation.metrics import RenderMetrics
from unbake.evaluation.render import held_out_cameras, load_envmap, render_mesh, render_views
from unbake.evaluation.summary import add_eval_args, evaluation_inputs, report_all
from unbake.gltf_io import Glb
from unbake.uv_bake import TrellisFrame

EVAL_ENV_MAPS = sorted(glob.glob("env_maps/eval/*.exr"))
# Every eval env map is scaled to this mean luminance, so a dim map (e.g. a night sky) does not
# render nearly black and hide material errors. None keeps each map's own brightness.
ENV_MAP_MEAN_LUMINANCE = 1.0
NUM_VIEWS = 8
RESOLUTION = 512

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
add_eval_args(parser)
parser.add_argument("--no-renders", action="store_true", help="don't save the full-resolution relit renders")
args = parser.parse_args()
dataset, items, methods = evaluation_inputs(args)
envs = {os.path.basename(p)[:-4]: load_envmap(p, ENV_MAP_MEAN_LUMINANCE) for p in EVAL_ENV_MAPS}
assert envs, "no eval env maps found in env_maps/eval/"
cameras = held_out_cameras(NUM_VIEWS)
render_metrics = RenderMetrics()
print(f"{len(items)} items, methods: {', '.join(methods)}")
print(f"Relighting: {NUM_VIEWS} held-out views under {len(envs)} eval env maps: {', '.join(envs)}")

slug = lambda name: re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
thumb = 256
to_img = lambda t: np.asarray(Image.fromarray((t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize((thumb, thumb)))


def save_comparison(item, lighting, renders: dict[str, dict[str, torch.Tensor]]):
    """One block per env map; rows = ground truth, then each method; first 4 views."""
    views = min(4, NUM_VIEWS)
    rows = [np.concatenate([to_img(r[env][v]) for v in range(views)], 1) for env in envs for r in renders.values()]
    out = os.path.join(args.output, "items", item.id)
    os.makedirs(out, exist_ok=True)
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(out, "comparison.png"))
    with open(os.path.join(out, "comparison.txt"), "w") as f:
        f.write(f"lighting of the input: {lighting.name}\nrows per env map ({', '.join(envs)}): {', '.join(renders)}\n")


def save_renders(renders: dict[str, torch.Tensor], path: str):
    """Full-resolution grid: rows = env maps, columns = views."""
    if args.no_renders:
        return
    full = lambda t: (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    grid = np.concatenate([np.concatenate([full(renders[env][v]) for v in range(NUM_VIEWS)], 1) for env in envs], 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(grid).save(path)


records, missing = [], {name: [] for name in methods}
start = time.time()
for n, index in enumerate(items):
    item = dataset[index]
    gt = item.ground_truth()
    frame = TrellisFrame(np.concatenate([p.positions for p in gt.world_primitives()]))  # same placement for all
    gt_renders, masks = render_views(render_mesh(gt, frame), envs, cameras, RESOLUTION)
    render_dir = os.path.join(args.output, "renders", item.id)
    save_renders(gt_renders, os.path.join(render_dir, "ground_truth.png"))
    print(f"[{n + 1}/{len(items)}] {item.id[:12]}  ({len(item.lightings)} lighting conditions, {time.time() - start:.0f}s elapsed)")

    for li, lighting in enumerate(item.lightings):
        comparison = {"ground truth": gt_renders}
        for name, locate in methods.items():
            path, constants = locate(item, lighting)
            if not os.path.isfile(path):
                missing[name].append(path)
                continue
            renders, _ = render_views(render_mesh(Glb(path), frame, constants), envs, cameras, RESOLUTION)
            records.append({"item": item.id, "lighting": lighting.name, "method": name, "path": path,
                            "relit": {env: render_metrics(renders[env], gt_renders[env], masks) for env in envs}})
            save_renders(renders, os.path.join(render_dir, slug(name), f"{lighting.name}.png"))
            if li == 0:
                comparison[name] = renders
            del renders
        if li == 0:
            save_comparison(item, lighting, comparison)
    del gt_renders, masks
    torch.cuda.empty_cache()

summary, summary_common = report_all(records, list(methods), missing, num_envs=len(envs), num_views=NUM_VIEWS)
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "metrics_relit.json"), "w") as f:
    json.dump({"data": args.data, "items": [dataset.ids[i] for i in items], "methods": list(methods),
               "env_maps": EVAL_ENV_MAPS, "env_map_mean_luminance": ENV_MAP_MEAN_LUMINANCE,
               "num_views": NUM_VIEWS, "resolution": RESOLUTION,
               "summary": summary, "summary_common_pairs": summary_common,
               "missing_predictions": {n: len(p) for n, p in missing.items()}, "records": records}, f, indent=2)
if not args.no_renders:
    with open(os.path.join(args.output, "renders", "README.txt"), "w") as f:
        f.write(f"Rows = env maps ({', '.join(envs)}), columns = {NUM_VIEWS} held-out views at {RESOLUTION}px.\n")
print(f"\nSaved {args.output}/: metrics_relit.json, items/<item>/comparison.png"
      + ("" if args.no_renders else ", renders/<item>/ground_truth.png, renders/<item>/<method>/<lighting>.png"))
