"""
Unguided TRELLIS.2 baseline: the TRELLIS.2 material stage on every dataset input (an item's asset
baked under one lighting), conditioned on the geometry and an unlit render of the baked asset
(unbake/reference.py), i.e. only on what a user with the baked mesh has. The per-voxel materials
are baked into the input's original UV layout (unbake/uv_bake.py), so the output has the ground
truth's geometry and UV layout. Output per input (see unbake/predictions.py):

    <output>/<item id>__<lighting name>/textured.glb    prediction
                                       /cond.png        conditioning image
                                       /meta.json       settings (written last: marks the input as done)

    python -m unbake.baselines.unguided_trellis2                       # whole dataset, resolution 1024
    python -m unbake.baselines.unguided_trellis2 --resolution 512 --max-items 2

Stages, to keep at most ~5 GB of models in RAM:
  1. DINOv3 image conditioning for all pending reference images (kept on CPU), then DINOv3 is freed.
  2. Shape encoder + material flow model + material decoder: per item, the shape latent is
     computed once (all lightings share the geometry), then per lighting: sample the material
     latent, decode per-voxel PBR, bake into the UV layout, save.
"""
import os

os.environ.setdefault("HF_HOME", "/tmp/hf_cache/")  # model downloads (wiped on container restart)
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import time

import numpy as np
import torch
import trimesh


# xformers picks its Hopper-only FlashAttention-3 kernel on any GPU >= sm_90, which
# crashes on Blackwell (sm_120) with "invalid argument". Fall back to FlashAttention-2.
from xformers.ops.fmha import dispatch as _xformers_dispatch
_xformers_dispatch._set_use_fa3(False)

from trellis2 import models
from trellis2.pipelines import Trellis2TexturingPipeline, rembg

from unbake.dataset import UnbakeDataset
from unbake.predictions import PREDICTIONS, input_name
from unbake.reference import PITCH_DEG, YAW_DEG, render_unlit
from unbake.uv_bake import TrellisFrame, bake_voxels_to_uv, merge, to_gltf_images

REPO = "microsoft/TRELLIS.2-4B"
SEED = 42

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--data", default="data", help="dataset root")
parser.add_argument("--max-items", type=int, default=None, help="only the first N items")
parser.add_argument("--resolution", type=int, default=1024, choices=(512, 1024),
                    help="voxel resolution of the material stage")
parser.add_argument("--seed", type=int, default=SEED)
parser.add_argument("--output", default=None, help=f"default: {PREDICTIONS}/trellis_unguided_<resolution>"
                    "[_s<seed> unless the seed is the default]")
args = parser.parse_args()
output_dir = args.output or (f"{PREDICTIONS}/trellis_unguided_{args.resolution}"
                             + ("" if args.seed == SEED else f"_s{args.seed}"))

dataset = UnbakeDataset(args.data)
items = [dataset[i] for i in range(len(dataset))[:args.max_items]]
out_dir = lambda item, lighting: os.path.join(output_dir, input_name(item, lighting))
pending = [(item, [l for l in item.lightings if not os.path.isfile(os.path.join(out_dir(item, l), "meta.json"))])
           for item in items]
pending = [(item, lightings) for item, lightings in pending if lightings]
total = sum(len(l) for _, l in pending)
print(f"{total} predictions to make ({len(items)} items, resolution {args.resolution}, seed {args.seed}) "
      f"-> {output_dir}/")
if not total:
    raise SystemExit

# ---------------------------------------------------------------- pipeline (no models yet)
rembg.BiRefNet = lambda **kwargs: None  # reference images have alpha: background removal not needed
Trellis2TexturingPipeline.model_names_to_load = []
pipeline = Trellis2TexturingPipeline.from_pretrained(REPO, config_file="texturing_pipeline.json")
pipeline.low_vram = True  # each model goes to the GPU only while it runs
pipeline.cuda()


def load(name):
    path = pipeline._pretrained_args["models"][name]
    full = f"{REPO}/{path}" if path.startswith("ckpts/") else path
    print(f"  loading {name}")
    pipeline.models[name] = models.from_pretrained(full).eval()
    return pipeline.models[name]


def reference_image(item, lighting):
    glb = lighting.baked()
    image = render_unlit(glb, TrellisFrame(np.concatenate([p.positions for p in glb.world_primitives()])))
    assert image.mode == "RGBA", f"reference image for {input_name(item, lighting)} needs a transparent background"
    os.makedirs(out_dir(item, lighting), exist_ok=True)
    image.save(os.path.join(out_dir(item, lighting), "cond.png"))
    return image


start = time.time()
with torch.no_grad():
    # ------------------------------------------------------------ 1. image conditioning
    print(f"[1/2] Image conditioning (DINOv3) for {total} reference images")
    conds = {}
    for item, lightings in pending:
        for lighting in lightings:
            image = reference_image(item, lighting)
            conds[item.id, lighting.name] = pipeline.get_cond([pipeline.preprocess_image(image)], args.resolution,
                                                              include_neg_cond=False)["cond"].cpu()
    pipeline.image_cond_model = None
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------ 2. material stage per item
    print("[2/2] Material stage")
    load("shape_slat_encoder")
    flow = load(f"tex_slat_flow_model_{args.resolution}")
    load("tex_slat_decoder")
    done = 0
    for n, (item, lightings) in enumerate(pending):
        # All lightings of an item share the geometry (and UV layout): encode the shape once
        prims = lightings[0].baked().world_primitives()
        frame = TrellisFrame(np.concatenate([p.positions for p in prims]))
        positions, _, faces = merge(prims, frame)
        shape_slat = pipeline.encode_shape_slat(trimesh.Trimesh(vertices=positions, faces=faces, process=False), args.resolution)
        print(f"  item {n + 1}/{len(pending)} {item.id[:12]}: {len(faces)} faces, {shape_slat.feats.shape[0]} latent voxels")

        for lighting in lightings:
            t0 = time.time()
            cond = conds.pop((item.id, lighting.name)).cuda()
            torch.manual_seed(args.seed)
            tex_slat = pipeline.sample_tex_slat({"cond": cond, "neg_cond": torch.zeros_like(cond)}, flow, shape_slat)
            pbr_voxel = pipeline.decode_tex_slat(tex_slat)
            del tex_slat, cond

            # Bake into this lighting's baked asset: one texture set per material, at its baked texture size
            glb = lighting.baked()
            prims = glb.world_primitives()
            for material in sorted({p.material for p in prims if p.material is not None}):
                img_index = glb.material_base_color_image(material)
                size = glb.image(img_index).size if img_index is not None else (1024, 1024)
                maps = bake_voxels_to_uv([p for p in prims if p.material == material], frame, pbr_voxel, args.resolution, size)
                glb.set_pbr_textures(material, *to_gltf_images(maps))
            del pbr_voxel
            glb.save(os.path.join(out_dir(item, lighting), "textured.glb"))
            meta = {"input": lighting.baked_path, "method": "trellis_unguided", "resolution": args.resolution,
                    "seed": args.seed,
                    "view_yaw": YAW_DEG, "view_pitch": PITCH_DEG, "time_s": round(time.time() - t0, 1)}
            with open(os.path.join(out_dir(item, lighting), "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            done += 1
            elapsed = time.time() - start
            print(f"    [{done}/{total}] {lighting.name}  ({elapsed / 60:.1f} min, ~{elapsed / done * (total - done) / 60:.0f} min left)")
        del shape_slat
        torch.cuda.empty_cache()

print(f"Saved {done} predictions to {output_dir}/")
