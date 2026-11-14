"""Test-time optimization of the TRELLIS.2 texture latent with the rebake loss.

For each dataset input: the unguided TRELLIS.2 sample z0 (same conditioning, shape latent and seed as
unbake/baselines/unguided_trellis2.py), then z is optimized with Adam on
    L = Huber(rebake(z) - baked) + lambda * ||z - z0||^2 + anchor * L_anchor + metal_prior * mean(metallic)
(z normalized with the latent statistics), where rebake = Lambert under SH lighting with visibility,
kd = base color (the dataset bakes ignore metallic), and the lighting is refit by least squares at every
step. L_anchor keeps the mean albedo per channel at that of z0, since the refit lighting makes the rebake
loss blind to the albedo scale. The decoder runs with per-block gradient checkpointing (needed at 1024^3
on 24 GB) and loss scaling (it computes in fp16, unscaled gradients underflow to zero). Every few steps
the decoded materials are compared with the ground truth on the sampled texels (curves.json); the result
is written in the prediction layout of unbake/predictions.py, so unbake.evaluation applies.

    python -m unbake.method.latent_opt                             # lambda 0.1 + anchor, seed 42, all inputs
    python -m unbake.method.latent_opt --seed 1 --only 006d3159 --lightings golden_bay_2k_rot143
"""
import os
import argparse
import json
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

from unbake.dataset import UnbakeDataset
from unbake.predictions import PREDICTIONS, input_name
from unbake.method.rebake import Asset, srgb_to_linear, linear_to_srgb, fit_lighting, sample_maps, load_gt_maps
from unbake.method.common import (SEED, load_pipeline, prepare, write_prediction, select_inputs,
                                  set_block_checkpointing, voxel_index)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="dataset root")
    ap.add_argument("--output", default=None, help=f"default: {PREDICTIONS}/rebake_opt[_s<seed>][_lambda<lambda>]")
    ap.add_argument("--only", default=None, help="comma-separated item id prefixes")
    ap.add_argument("--lightings", default=None, help="comma-separated lighting names")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--resolution", type=int, default=1024, choices=(512, 1024))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--lambdas", default="0.1", help="regularization weights; several give one output each")
    ap.add_argument("--anchor", type=float, default=1.0, help="weight of the per-channel mean-albedo anchor")
    ap.add_argument("--metal-prior", type=float, default=0.0, help="weight of the mean decoded metallic (pushes to 0)")
    ap.add_argument("--seed", type=int, default=SEED, help="sampling seed (as unguided_trellis2 --seed)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--points", type=int, default=100000)
    ap.add_argument("--loss-scale", type=float, default=65536.0)
    args = ap.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    base = args.output or f"{PREDICTIONS}/rebake_opt" + ("" if args.seed == SEED else f"_s{args.seed}")
    out_root = {lam: base if len(lambdas) == 1 else f"{base}_lambda{lam}" for lam in lambdas}
    inputs = select_inputs(UnbakeDataset(args.data), args.only, args.lightings, args.max_items)
    inputs = [(it, l) for it, l in inputs
              if not all(os.path.isfile(os.path.join(out_root[lam], input_name(it, l), "meta.json")) for lam in lambdas)]
    print(f"{len(inputs)} inputs to optimize -> {', '.join(out_root.values())}")
    if not inputs:
        return
    pipe = load_pipeline()
    dec = pipe.models["tex_slat_decoder"]
    flow = pipe.models[f"tex_slat_flow_model_{args.resolution}"]
    std = torch.tensor(pipe.tex_slat_normalization["std"], device="cuda")[None]
    mean = torch.tensor(pipe.tex_slat_normalization["mean"], device="cuda")[None]
    layout = pipe.pbr_attr_layout
    os.makedirs(base, exist_ok=True)
    curves_path = os.path.join(base, "curves.json")
    curves = json.load(open(curves_path)) if os.path.isfile(curves_path) else {}

    for item, lighting in inputs:
        name = input_name(item, lighting)
        try:
            t0 = time.time()
            asset = Asset(lighting.baked_path)
            with torch.no_grad():  # unguided sample, exactly as in the baseline
                cond, shape_slat, frame, image = prepare(pipe, lighting, args.resolution)
                torch.manual_seed(args.seed)
                tex_slat = pipe.sample_tex_slat(cond, flow, shape_slat)
            z0 = ((tex_slat.feats - mean) / std).detach()
            dec.to("cuda")  # the pipeline moves models back to the CPU after use (low_vram)
            set_block_checkpointing(dec, True)

            tx = asset.texels(args.points)
            T = asset.transfer(tx["pos"], tx["nrm"], 128)
            c_srgb = tx["baked"]
            c_lin = srgb_to_linear(c_srgb)
            gt_bc, gt_ro, gt_me = sample_maps(load_gt_maps(asset, item.ground_truth_path), tx)
            with torch.no_grad():
                coords = dec(tex_slat.replace(feats=z0 * std + mean)).coords
            vidx = voxel_index(coords, tx["pos"], args.resolution)
            ok = vidx >= 0
            print(f"{item.id[:8]}…{lighting.name[:18]}: {len(z0)} latents, {len(coords)} voxels, {ok.float().mean():.3f} of texels "
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
                with torch.no_grad():
                    out = dec(tex_slat.replace(feats=z.detach() * std + mean))
                    write_prediction(lighting, frame, out.replace(out.feats.float() * 0.5 + 0.5), args.resolution,
                                     os.path.join(out_root[lam], name), image=image,
                                     meta={"method": "rebake_opt", "seed": args.seed, "lambda": lam, "anchor": args.anchor,
                                           "metal_prior": args.metal_prior, "steps": args.steps, "lr": args.lr})
                del z, opt, feats
                torch.cuda.empty_cache()
            json.dump(curves, open(curves_path, "w"), indent=2)
            set_block_checkpointing(dec, False)
            print(f"   done in {time.time() - t0:.0f} s", flush=True)
        except Exception as e:  # log and continue with the next input
            import traceback
            print(f"FAILED {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            open(os.path.join(base, f"error_{name}.txt"), "w").write(traceback.format_exc())
            set_block_checkpointing(dec, False)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
