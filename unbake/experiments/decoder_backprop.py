"""Memory/time test: can we backpropagate a loss on the decoded materials to the texture latent?

Encodes a real asset, takes a texture latent of the right shape (random, like an early x0 estimate),
decodes it with gradients enabled, computes a loss on the decoded per-voxel materials and calls
backward(). Reports peak VRAM and time per resolution and variant.

    python -m unbake.experiments.decoder_backprop <asset.glb>
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import trimesh

from unbake.baselines.common import load_parts, normalization, normalize
from unbake.method.common import set_block_checkpointing


VARIANTS = [v for v in os.environ.get("VARIANTS", "plain,block-ckpt").split(",")]


def main():
    path = sys.argv[1]
    from trellis2.pipelines import Trellis2TexturingPipeline
    pipe = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
    pipe.cuda()
    dec = pipe.models["tex_slat_decoder"]
    parts = load_parts(path)
    c, s = normalization(np.concatenate([p["mesh"].vertices for p in parts]))
    verts, faces, off = [], [], 0
    for p in parts:
        verts.append(normalize(np.asarray(p["mesh"].vertices), c, s)); faces.append(np.asarray(p["mesh"].faces) + off)
        off += len(p["mesh"].vertices)
    mesh = trimesh.Trimesh(np.concatenate(verts), np.concatenate(faces), process=False)
    print(f"{Path(path).parent.name[:8]}/{Path(path).name}: {len(mesh.faces)} faces", flush=True)

    for res in [int(r) for r in os.environ.get("RES", "512,1024").split(",")]:
        with torch.no_grad():
            shape_slat = pipe.encode_shape_slat(mesh, res)
        n_lat = shape_slat.coords.shape[0]
        dec.to("cuda")
        for variant in VARIANTS:
            set_block_checkpointing(dec, variant == "block-ckpt")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            feats = torch.randn(n_lat, 32, device="cuda", requires_grad=True)
            lat = shape_slat.replace(feats=feats)
            t0 = time.time()
            try:
                out = dec(lat)
                n_vox = out.feats.shape[0]
                target = torch.rand_like(out.feats)
                loss = torch.nn.functional.huber_loss(out.feats.float(), target.float())
                loss.backward()
                torch.cuda.synchronize()
                ok = feats.grad is not None and torch.isfinite(feats.grad).all().item()
                print(f"res {res:4d} {variant:10s}: OK  latent {n_lat:6d} -> {n_vox:8d} voxels, "
                      f"peak {(torch.cuda.max_memory_allocated() - base) / 2**30:5.2f} GB above base, "
                      f"{time.time() - t0:5.1f} s, grad finite {ok}", flush=True)
            except torch.OutOfMemoryError as e:
                print(f"res {res:4d} {variant:10s}: OOM ({str(e)[:80]})", flush=True)
            except RuntimeError as e:
                print(f"res {res:4d} {variant:10s}: ERROR {type(e).__name__}: {str(e)[:120]}", flush=True)
            del feats, lat
            out = target = loss = None
        print(f"  total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
