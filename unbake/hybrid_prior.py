"""
Hybrid predictions: base color from a guided prediction, roughness / metallic from the 2D prior fused directly into
the UV layout (experiment 21: the fused 16-view SuperMat estimates beat the constants dataset-wide, while guiding the
latent towards them loses most of their information).

Per input, the saved per-view SuperMat single-view predictions (prior2d, 16 Fibonacci views, from the guided run
that computed them) are fused per texel of every material, with the prior's cos^2 weighting and depth test
(unbake.prior2d.fuse). Texels no view sees take the nearest seen texel of the same texture. The base prediction's
metallicRoughness texture is replaced; base color, geometry and UV layout stay unchanged.

    python -m unbake.hybrid_prior          # both variants below, all dataset inputs
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from unbake.baselines.supermat import render_views
from unbake.dataset import UnbakeDataset
from unbake.evaluation.part_metrics import atlases
from unbake.gltf_io import Glb
from unbake.paths import DATA
from unbake.predictions import PREDICTION_FILE, PREDICTIONS, input_name
from unbake.prior2d import CHANNELS, _view_parts, fibonacci_views, fuse

PRIOR_RUN = os.path.join(PREDICTIONS, "guided_supermat_sv_b1_1024")  # holds <input>/supermat_sv_pred (experiment 10)
VARIANTS = {  # output folder -> base color prediction folder
    "guided_detail+sm16_rm": "guided_supermat_sv_b1_1024_vis_ss4_detail",
    "guided+sm16_rm": "guided_supermat_sv_b1_1024_vis_ss4",
}


@torch.no_grad()
def fused_maps(pred: Glb, baked_path: str, pred_dir: str, num_views=16, size=512):
    """{material: {"roughness", "metallic"}} (H, W) float maps in the prediction's own UV layout."""
    views = fibonacci_views(num_views)
    _, depths = render_views(_view_parts(baked_path), size, views)
    load = lambda n, k: torch.tensor(np.asarray(Image.open(os.path.join(pred_dir, f"{k:04d}", f"{n}.png")).convert("L")
                                                .resize((size, size), Image.BILINEAR)),
                                     dtype=torch.float32, device="cuda")[..., None] / 255
    preds = {c: [load(c, k) for k in range(num_views)] for c in CHANNELS}
    out = {}
    for a in atlases(pred):
        val, _, nv = fuse(views, depths, preds, a["pos"][a["mask_t"]], a["nrm"][a["mask_t"]])
        seen = nv.cpu().numpy() > 0
        maps = {}
        for c in CHANNELS:
            grid = np.full(a["mask"].shape, np.nan, np.float32)
            grid[a["mask"]] = np.where(seen, val[c][:, 0].cpu().numpy(), np.nan)
            if np.isnan(grid).all():
                grid[:] = 0.5 if c == "roughness" else 0.0
            _, (iy, ix) = ndimage.distance_transform_edt(np.isnan(grid), return_indices=True)
            maps[c] = grid[iy, ix]  # also fills outside the islands (no seams under mip filtering)
        out[a["material"]] = maps
    return out


def write(base_path, maps, out_path):
    g = Glb(base_path)
    u8 = lambda x: (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    for material, m in maps.items():
        pbr = g.json["materials"][material]["pbrMetallicRoughness"]
        base = g.image(g.json["textures"][pbr["baseColorTexture"]["index"]]["source"]).convert("RGB")
        assert base.size == m["roughness"].shape[::-1], "texture size mismatch"
        mr = Image.fromarray(np.stack([np.full_like(u8(m["roughness"]), 255), u8(m["roughness"]), u8(m["metallic"])], -1), "RGB")
        g.set_pbr_textures(material, base, mr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--predictions", default=PREDICTIONS)
    ap.add_argument("--prior-run", default=PRIOR_RUN, help="run folder with <input>/supermat_sv_pred")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="out_folder=base_folder pairs (default: the two guided variants)")
    args = ap.parse_args()
    variants = dict(v.split("=", 1) for v in args.variants) if args.variants else VARIANTS
    dataset = UnbakeDataset(args.data)
    todo = [(item, l) for item in dataset for l in item.lightings]
    start = time.time()
    for n, (item, lighting) in enumerate(todo):
        name = input_name(item, lighting)
        outs = {v: os.path.join(args.predictions, v, name, PREDICTION_FILE) for v in variants}
        if all(os.path.isfile(p) for p in outs.values()):
            continue
        ref = os.path.join(args.predictions, next(iter(variants.values())), name, PREDICTION_FILE)
        maps = fused_maps(Glb(ref), lighting.baked_path, os.path.join(args.prior_run, name, "supermat_sv_pred"))
        for v, base in variants.items():
            write(os.path.join(args.predictions, base, name, PREDICTION_FILE), maps, outs[v])
        print(f"[{n + 1}/{len(todo)}] {name[:8]} {lighting.name} ({time.time() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
