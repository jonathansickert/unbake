"""Guided sampling: rectified-flow sampling of the TRELLIS.2 texture latent where, at the last K sampler
steps, the clean-latent estimate x0 = x_t - t v is corrected by a few gradient steps on the rebake loss
(through the frozen decoder) before sampling continues from the corrected estimate:

    x_{t'} = (1 - t') x0_corrected + (sigma_min + (1 - sigma_min) t') eps_hat

Same sampler settings, conditioning, shape latent and seed as unbake/baselines/unguided_trellis2.py, so
--guided-steps 0 reproduces the baseline exactly. --init input starts from the encoded baked texture,
noised to --t-start (SDEdit-like). Output in the prediction layout of unbake/predictions.py.

    python -m unbake.method.guided_sampling --anchor 1 --skip-last 2
"""
import os
import argparse
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from unbake.baselines.common import load_parts, normalization
from unbake.dataset import UnbakeDataset
from unbake.predictions import PREDICTIONS, input_name
from unbake.method.rebake import Asset, srgb_to_linear, linear_to_srgb, fit_lighting, sample_maps, load_gt_maps
from unbake.method.common import (SEED, load_pipeline, prepare, write_prediction, select_inputs,
                                  set_block_checkpointing, voxel_index)

ENC_LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}


def encode_input(baked, enc, res, roughness=0.5, metallic=0.0):
    """Texture latent of the baked input used as base color (constant roughness / metallic), voxelized exactly
    like experiments/latent_ceiling.py, i.e. in the same TRELLIS space as Asset and the shape latent."""
    import o_voxel
    from trellis2.modules.sparse import SparseTensor
    parts = load_parts(baked)
    center, scale = normalization(np.concatenate([p["mesh"].vertices for p in parts]))
    scene = trimesh.load(baked, force="scene", process=False)
    m = np.eye(4); m[:3, 3] = -center
    s = np.diag([scale, scale, scale, 1.0])
    r = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    scene.apply_transform(r @ s @ m)
    coords, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        scene, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
    feats = torch.cat([attr[k] for k in ENC_LAYOUT], -1).float() / 255
    feats[:, ENC_LAYOUT["metallic"]] = metallic
    feats[:, ENC_LAYOUT["roughness"]] = roughness
    feats[:, ENC_LAYOUT["alpha"]] = 1.0
    x = SparseTensor((feats * 2 - 1).cuda(), torch.cat([torch.zeros_like(coords[:, :1]), coords], -1).int().cuda())
    with torch.no_grad():
        return enc(x)


def align(src, dst_coords):
    """Features of sparse tensor src at dst_coords (rows missing in src: None mask)."""
    def key(c):
        c = c[:, 1:].long()
        return (c[:, 0] * 4096 + c[:, 1]) * 4096 + c[:, 2]
    ks, kd = key(src.coords), key(dst_coords)
    order = torch.argsort(ks)
    i = torch.searchsorted(ks[order], kd).clamp(max=len(ks) - 1)
    hit = ks[order][i] == kd
    return src.feats[order[i]], hit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="dataset root")
    ap.add_argument("--output", default=None, help=f"default: {PREDICTIONS}/rebake_guided[_s<seed>]")
    ap.add_argument("--only", default=None, help="comma-separated item id prefixes")
    ap.add_argument("--lightings", default=None, help="comma-separated lighting names")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED, help="sampling seed (as unguided_trellis2 --seed)")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--guided-steps", type=int, default=6, help="guide the last K of the sampler steps")
    ap.add_argument("--inner", type=int, default=5, help="gradient steps on x0 per guided sampler step")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=0.1, help="weight of ||x0 - x0_pred||^2")
    ap.add_argument("--points", type=int, default=100000)
    ap.add_argument("--loss-scale", type=float, default=65536.0)
    ap.add_argument("--anchor", type=float, default=0.0, help="weight of the per-channel mean-albedo anchor")
    ap.add_argument("--skip-last", type=int, default=0, help="leave the last N sampler steps unguided")
    ap.add_argument("--init", default="noise", choices=["noise", "input"],
                    help="input: start from the encoded baked texture, noised to --t-start (SDEdit-like)")
    ap.add_argument("--t-start", type=float, default=1.0, help="start time for --init input (1 = pure noise)")
    ap.add_argument("--metal-prior", type=float, default=0.0, help="weight of the mean decoded metallic (pushes to 0)")
    args = ap.parse_args()
    out_root = args.output or f"{PREDICTIONS}/rebake_guided" + ("" if args.seed == SEED else f"_s{args.seed}")
    inputs = [(it, l) for it, l in select_inputs(UnbakeDataset(args.data), args.only, args.lightings, args.max_items)
              if not os.path.isfile(os.path.join(out_root, input_name(it, l), "meta.json"))]
    print(f"{len(inputs)} inputs to sample -> {out_root}")
    if not inputs:
        return

    pipe = load_pipeline()
    dec = pipe.models["tex_slat_decoder"]
    flow = pipe.models[f"tex_slat_flow_model_{args.resolution}"]
    sampler, params = pipe.tex_slat_sampler, dict(pipe.tex_slat_sampler_params)
    t_mean = torch.tensor(pipe.tex_slat_normalization["mean"], device="cuda")[None]
    t_std = torch.tensor(pipe.tex_slat_normalization["std"], device="cuda")[None]
    s_mean = torch.tensor(pipe.shape_slat_normalization["mean"], device="cuda")[None]
    s_std = torch.tensor(pipe.shape_slat_normalization["std"], device="cuda")[None]
    layout = pipe.pbr_attr_layout
    enc = None
    if args.init == "input":
        from huggingface_hub import snapshot_download
        from trellis2 import models
        snap = snapshot_download("microsoft/TRELLIS.2-4B", allow_patterns=["ckpts/tex_enc_*"])
        enc = models.from_pretrained(f"{snap}/ckpts/tex_enc_next_dc_f16c32_fp16").cuda().eval()
    for item, lighting in inputs:
        name = input_name(item, lighting)
        baked = lighting.baked_path
        try:
            t0 = time.time()
            asset = Asset(baked)
            with torch.no_grad():
                cond, shape_slat, frame, image = prepare(pipe, lighting, args.resolution)
            shape_norm = (shape_slat - s_mean) / s_std
            torch.manual_seed(args.seed)  # the first draw after seeding, as in pipe.sample_tex_slat (CPU RNG)
            noise = shape_norm.replace(feats=torch.randn(shape_norm.coords.shape[0], flow.in_channels - shape_norm.feats.shape[1]
                                                         ).to("cuda"))
            tx = asset.texels(args.points)
            T = asset.transfer(tx["pos"], tx["nrm"], 128)
            c_srgb, c_lin = tx["baked"], srgb_to_linear(tx["baked"])
            gt_bc, gt_ro, gt_me = sample_maps(load_gt_maps(asset, item.ground_truth_path), tx)
            flow.to("cuda")
            dec.to("cuda")
            set_block_checkpointing(dec, True)
            vidx = ok = None

            def decode(x0n):  # normalized latent features -> per-voxel materials in [0, 1]
                return dec(noise.replace(feats=x0n * t_std + t_mean)).feats.float() * 0.5 + 0.5

            steps, rescale = params["steps"], params["rescale_t"]
            t_seq = np.linspace(1, 0, steps + 1)
            t_seq = (rescale * t_seq / (1 + (rescale - 1) * t_seq)).tolist()
            kw = dict(neg_cond=cond["neg_cond"], guidance_strength=params["guidance_strength"],
                      guidance_interval=params["guidance_interval"], guidance_rescale=params.get("guidance_rescale", 0.0),
                      concat_cond=shape_norm)
            x, hist, i0 = noise, [], 0
            if args.init == "input":
                z_in = encode_input(baked, enc, args.resolution)
                f_in, hit = align(z_in, noise.coords)
                z_in_n = torch.where(hit[:, None], (f_in.float() - t_mean) / t_std, torch.zeros_like(noise.feats))
                i0 = min(i for i in range(steps) if t_seq[i] <= args.t_start + 1e-6)
                t0_ = t_seq[i0]
                x = noise.replace(feats=(1 - t0_) * z_in_n + (sampler.sigma_min + (1 - sampler.sigma_min) * t0_) * noise.feats)
                hist.append({"t": round(t0_, 3), "encoded_fraction": round(float(hit.float().mean()), 4), "albedo_mae": float("nan")})
            for i in range(i0, steps):
                t, t_prev = t_seq[i], t_seq[i + 1]
                with torch.no_grad():
                    x0, eps, v = sampler._get_model_prediction(flow, x, t, cond["cond"], **kw)
                if steps - args.guided_steps <= i < steps - args.skip_last:
                    if vidx is None:
                        with torch.no_grad():
                            coords = dec(noise.replace(feats=x0.feats * t_std + t_mean)).coords
                        vidx = voxel_index(coords, tx["pos"], args.resolution)
                        ok = vidx >= 0
                    z_pred = x0.feats.detach()
                    z = z_pred.clone().requires_grad_(True)
                    opt = torch.optim.Adam([z], lr=args.lr)
                    with torch.no_grad():
                        kd_ref = srgb_to_linear(decode(z_pred)[vidx[ok]][:, layout["base_color"]].clamp(0, 1)).mean(0)
                    for _ in range(args.inner):
                        feats = decode(z)
                        kd = srgb_to_linear(feats[vidx[ok]][:, layout["base_color"]].clamp(0, 1))
                        with torch.no_grad():
                            L = fit_lighting(kd, c_lin[ok], T[ok])
                        loss = F.huber_loss(linear_to_srgb(kd * (T[ok] @ L)), c_srgb[ok], delta=0.1) + \
                            args.lam * ((z - z_pred) ** 2).mean() + args.anchor * ((kd.mean(0) / kd_ref - 1) ** 2).sum() + \
                            args.metal_prior * feats[vidx[ok]][:, layout["metallic"]].clamp(0, 1).mean()
                        opt.zero_grad()
                        (loss * args.loss_scale).backward()
                        opt.step()
                    with torch.no_grad():
                        f = decode(z.detach())[vidx[ok]].clamp(0, 1)
                        hist.append({"t": round(t, 3), "loss": float(loss),
                                     "albedo_mae": float((f[:, layout["base_color"]] - gt_bc[ok]).abs().mean())})
                    x0 = x0.replace(feats=z.detach())
                    x = x0 * (1 - t_prev) + eps * (sampler.sigma_min + (1 - sampler.sigma_min) * t_prev)
                else:
                    x = x - v * (t - t_prev)
            with torch.no_grad():
                final = x.replace(feats=x.feats * t_std + t_mean)
                out = dec(final)
                pv = out.replace(out.feats.float() * 0.5 + 0.5)
                if vidx is not None:
                    f = pv.feats[vidx[ok]].clamp(0, 1)
                    hist.append({"t": 0.0, "albedo_mae": float((f[:, layout["base_color"]] - gt_bc[ok]).abs().mean()),
                                 "roughness_mae": float((f[:, layout["roughness"]][:, 0] - gt_ro[ok]).abs().mean()),
                                 "metallic_mae": float((f[:, layout["metallic"]][:, 0] - gt_me[ok]).abs().mean())})
                write_prediction(lighting, frame, pv, args.resolution, os.path.join(out_root, name), image=image,
                                 meta={"method": "rebake_guided", "seed": args.seed, "guidance": vars(args), "history": hist})
            set_block_checkpointing(dec, False)
            print(f"{item.id[:8]}…{lighting.name[:18]}: " + " -> ".join(f"{h['albedo_mae']:.3f}" for h in hist) +
                  f"  ({time.time() - t0:.0f} s)", flush=True)
        except Exception as e:  # log and continue with the next input
            import traceback
            print(f"FAILED {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            os.makedirs(out_root, exist_ok=True)
            open(os.path.join(out_root, f"error_{name}.txt"), "w").write(traceback.format_exc())
            set_block_checkpointing(dec, False)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
