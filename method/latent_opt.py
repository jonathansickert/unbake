"""Feasibility test 4: does optimizing the TRELLIS.2 texture latent with the rebake loss improve the materials?

For each input: unguided TRELLIS.2 samples the texture latent z0 (seed 0, same as the baseline), then z is
optimized with Adam on
    L = Huber(rebake(z) - baked) + lambda * ||z - z0||^2      (z normalized with the latent statistics)
where rebake = Lambert under SH lighting with visibility, kd = base color (the dataset bakes ignore
metallic), and the lighting is refit by least squares at every step. The decoder runs with per-block
gradient checkpointing (needed at 1024^3 on 24 GB) and loss scaling (it computes in fp16, unscaled
gradients underflow to zero). Every few steps the decoded materials are compared with the ground truth on
the sampled texels; the final materials are baked into the UV layout in the baseline output format, so
evaluate_uv.py / evaluate_relit.py / compare.py apply.

    python method/latent_opt.py --data data/unbake_sub10 --out results/feasibility/latent_opt
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from rebake import Asset, srgb_to_linear, linear_to_srgb, fit_lighting, sample_maps, load_gt_maps
from unguided_trellis2 import load_pipeline
from common import set_block_checkpointing, voxel_index, write_result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="comma-separated object id prefixes")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lambdas", default="0,0.1,1", help="regularization weights to compare")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--points", type=int, default=100000)
    ap.add_argument("--loss-scale", type=float, default=65536.0)
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (as unguided_trellis2.py --seed)")
    ap.add_argument("--metal-prior", type=float, default=0.0, help="weight of the mean decoded metallic (pushes to 0)")
    ap.add_argument("--anchor", type=float, default=0.0, help="weight of the per-channel mean-albedo anchor to the "
                    "unguided prediction (the refit lighting makes the rebake loss blind to albedo scale)")
    args = ap.parse_args()

    pipe = load_pipeline("microsoft/TRELLIS.2-4B")
    dec = pipe.models["tex_slat_decoder"]
    std = torch.tensor(pipe.tex_slat_normalization["std"], device="cuda")[None]
    mean = torch.tensor(pipe.tex_slat_normalization["mean"], device="cuda")[None]
    layout = pipe.pbr_attr_layout
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    curves_path = out_root / "curves.json"
    curves = json.loads(curves_path.read_text()) if curves_path.exists() else {}
    lambdas = [float(x) for x in args.lambdas.split(",")]
    inputs = sorted(Path(args.data).glob("*/*_rot*.glb"))
    if args.only:
        inputs = [p for p in inputs if any(p.parent.name.startswith(o) for o in args.only.split(","))]

    for baked in inputs:
        name = f"{baked.parent.name}__{baked.stem}"
        if all((out_root / f"lambda{lam}" / name / "meta.json").exists() for lam in lambdas):
            continue
        try:
            t0 = time.time()
            asset = Asset(baked)
            geom = trimesh.Trimesh(np.concatenate(asset.v), np.concatenate(
                [f + o for f, o in zip(asset.f, np.cumsum([0] + [len(v) for v in asset.v]))]), process=False)
            with torch.no_grad():  # unguided sample, as in baselines/unguided_trellis2.py (seed 0, dataset render)
                image = pipe.preprocess_image(Image.open(baked.with_suffix(".png")))
                torch.manual_seed(args.seed)
                cond = pipe.get_cond([image], 1024 if args.resolution == 1024 else 512)
                shape_slat = pipe.encode_shape_slat(geom, args.resolution)
                tex_slat = pipe.sample_tex_slat(cond, pipe.models[f"tex_slat_flow_model_{args.resolution}"], shape_slat, {})
            z0 = ((tex_slat.feats - mean) / std).detach()
            dec.to("cuda")
            set_block_checkpointing(dec, True)

            tx = asset.texels(args.points)
            T = asset.transfer(tx["pos"], tx["nrm"], 128)
            c_srgb = tx["baked"]
            c_lin = srgb_to_linear(c_srgb)
            gt_bc, gt_ro, gt_me = sample_maps(load_gt_maps(asset, baked.parent / "pbr.glb"), tx)
            with torch.no_grad():
                coords = dec(tex_slat.replace(feats=z0 * std + mean)).coords
            vidx = voxel_index(coords, tx["pos"], args.resolution)
            ok = vidx >= 0
            print(f"{name[:8]}…{baked.stem[:18]}: {len(z0)} latents, {len(coords)} voxels, {ok.float().mean():.3f} of texels "
                  f"in a voxel, setup {time.time() - t0:.0f} s", flush=True)

            def decode(z):
                return dec(tex_slat.replace(feats=z * std + mean)).feats.float() * 0.5 + 0.5

            def measure(feats):
                f = feats[vidx[ok]].clamp(0, 1)
                bc, ro, me = f[:, layout["base_color"]], f[:, layout["roughness"]][:, 0], f[:, layout["metallic"]][:, 0]
                return {"albedo_mae": float((bc - gt_bc[ok]).abs().mean()), "roughness_mae": float((ro - gt_ro[ok]).abs().mean()),
                        "metallic_mae": float((me - gt_me[ok]).abs().mean())}

            with torch.no_grad():
                kd0_mean = srgb_to_linear(decode(z0)[vidx[ok]][:, layout["base_color"]].clamp(0, 1)).mean(0)
            curves.setdefault(name, {})
            for lam in lambdas:
                z = z0.clone().requires_grad_(True)
                opt = torch.optim.Adam([z], lr=args.lr)
                curve = []
                for step in range(args.steps + 1):
                    feats = decode(z)
                    bc = feats[vidx[ok]][:, layout["base_color"]].clamp(0, 1)
                    kd = srgb_to_linear(bc)
                    with torch.no_grad():
                        L = fit_lighting(kd, c_lin[ok], T[ok])
                    pred = linear_to_srgb(kd * (T[ok] @ L))
                    l_rebake = F.huber_loss(pred, c_srgb[ok], delta=0.1)
                    l_reg = ((z - z0) ** 2).mean()
                    l_anchor = ((kd.mean(0) / kd0_mean - 1) ** 2).sum()
                    l_metal = feats[vidx[ok]][:, layout["metallic"]].clamp(0, 1).mean()
                    loss = l_rebake + lam * l_reg + args.anchor * l_anchor + args.metal_prior * l_metal
                    if step % args.eval_every == 0 or step == args.steps:
                        m = measure(feats.detach())
                        m.update({"step": step, "rebake": float(l_rebake), "reg": float(l_reg)})
                        curve.append(m)
                        print(f"   lambda {lam:<4} step {step:3d}: rebake {m['rebake']:.5f} | albedo MAE {m['albedo_mae']:.4f} "
                              f"rough {m['roughness_mae']:.4f} metal {m['metallic_mae']:.4f} | drift {m['reg']:.4f}", flush=True)
                    if step == args.steps:
                        break
                    opt.zero_grad()
                    (loss * args.loss_scale).backward()
                    opt.step()
                curves[name][str(lam)] = curve
                out_dir = out_root / f"lambda{lam}" / name
                out_dir.mkdir(parents=True, exist_ok=True)
                with torch.no_grad():
                    out = dec(tex_slat.replace(feats=z.detach() * std + mean))
                    write_result(asset, out.replace(out.feats.float() * 0.5 + 0.5), args.resolution, layout, out_dir, baked, lam, args)
                del z, opt, feats
                torch.cuda.empty_cache()
            curves_path.write_text(json.dumps(curves, indent=2))
            set_block_checkpointing(dec, False)
            print(f"   done in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:  # log and continue with the next input
            import traceback
            print(f"FAILED {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            (Path(args.out) / f"error_{name}.txt").write_text(traceback.format_exc())
            set_block_checkpointing(dec, False)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
