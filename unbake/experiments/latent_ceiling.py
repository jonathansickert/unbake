"""Upper bound of any latent-space method: encode the ground-truth PBR materials with the TRELLIS.2
texture encoder, decode them again and bake into the original UV layout.

Written in the prediction layout of unbake/predictions.py, so unbake.evaluation applies directly. The error
of this reconstruction is what guidance in the texture latent can at best reach.

    python -m unbake.experiments.latent_ceiling                     # -> output/predictions/latent_upper_bound
"""
import os
import argparse
import time

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import trimesh

import o_voxel
from trellis2.modules.sparse import SparseTensor
from trellis2 import models
from unbake.baselines.common import load_parts, normalization
from unbake.dataset import UnbakeDataset
from unbake.predictions import PREDICTIONS, input_name
from unbake.uv_bake import TrellisFrame
from unbake.method.common import REPO, select_inputs, write_prediction

LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}


@torch.no_grad()
def reconstruct(gt_path, enc, dec, resolution):
    """Ground truth in TRELLIS space -> PBR voxels (TRELLIS.2's voxelizer) -> encode -> decode, values in [0, 1]."""
    parts = load_parts(gt_path)
    center, scale = normalization(np.concatenate([p["mesh"].vertices for p in parts]))
    scene = trimesh.load(gt_path, force="scene", process=False)
    m = np.eye(4)
    m[:3, 3] = -center
    s = np.diag([scale, scale, scale, 1.0])
    r = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])  # (x, y, z) -> (x, -z, y)
    scene.apply_transform(r @ s @ m)
    coords, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        scene, grid_size=resolution, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    feats = torch.cat([attr[k] for k in LAYOUT], -1).float() / 255 * 2 - 1
    x = SparseTensor(feats.cuda(), torch.cat([torch.zeros_like(coords[:, :1]), coords], -1).int().cuda())
    z = enc(x)
    return dec(z) * 0.5 + 0.5, int(feats.shape[0]), int(z.coords.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="dataset root")
    ap.add_argument("--output", default=f"{PREDICTIONS}/latent_upper_bound")
    ap.add_argument("--only", default=None, help="comma-separated item id prefixes")
    ap.add_argument("--lightings", default=None, help="comma-separated lighting names")
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    snap = snapshot_download(REPO, allow_patterns=["ckpts/tex_enc_*", "ckpts/tex_dec_*"])
    enc = models.from_pretrained(f"{snap}/ckpts/tex_enc_next_dc_f16c32_fp16").cuda().eval()
    dec = models.from_pretrained(f"{snap}/ckpts/tex_dec_next_dc_f16c32_fp16").cuda().eval()
    cache = {}
    for item, lighting in select_inputs(UnbakeDataset(args.data), args.only, args.lightings):
        out_dir = os.path.join(args.output, input_name(item, lighting))
        if os.path.isfile(os.path.join(out_dir, "meta.json")):
            continue
        t0 = time.time()
        if item.id not in cache:  # the ground truth is shared by all lightings of an item
            cache.clear()
            cache[item.id] = reconstruct(item.ground_truth_path, enc, dec, args.resolution)
        rec, n_vox, n_lat = cache[item.id]
        prims = lighting.baked().world_primitives()
        frame = TrellisFrame(np.concatenate([p.positions for p in prims]))
        write_prediction(lighting, frame, rec, args.resolution, out_dir,
                         meta={"method": "latent_upper_bound", "n_pbr_voxels": n_vox, "n_latent": n_lat,
                               "time_s": round(time.time() - t0, 1)})
        print(f"{item.id[:8]}…{lighting.name}: {n_vox} PBR voxels -> {n_lat} latents, {time.time() - t0:.1f} s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
