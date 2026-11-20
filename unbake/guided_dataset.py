"""
UnBake on the whole dataset: TRELLIS.2 material stage guided by the SH rebake loss (with visibility, on
observations sampled at 4x texture density, smooth normals) and the 2D roughness / metallic prior of
SuperMat's single-view network on 16 views (unbake/guided_trellis2.py, unbake/prior2d.py, Method.md), with
one prior weight beta. Output per input as the baselines (unbake/predictions.py):

    <output>/<item id>__<lighting name>/textured.glb    prediction
                                       /cond.png        conditioning image
                                       /supermat_sv_*   renders and per-view SuperMat predictions (if SuperMat ran)
                                       /meta.json       settings and losses (written last: marks the input as done)

    python -m unbake.guided_dataset --sm-dir ../SuperMat         # 1024^3, visibility, 4x observations, beta 1
    python -m unbake.guided_dataset --reuse-views-from output/predictions/guided_supermat_sv_b1_1024  # no SuperMat run

Stages, so every model is loaded once:
  1. DINOv3 conditioning of every pending input, then DINOv3 is freed.
  2. SuperMat single-view: 16 renders and predictions per input that has none saved yet; freed.
  3. TRELLIS.2: per item the shape latent once, then per lighting: observations, prior fused from the saved
     per-view predictions (in memory, ~3 s), guided sampling, UV bake, save.
Finished inputs (meta.json) are skipped, so rerunning continues a run. Apart from the predictions only the
per-view SuperMat outputs are stored (~1 MB per input), and only when SuperMat has to run.
"""
import os

os.environ.setdefault("HF_HOME", "/tmp/hf_cache/")
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import time

import numpy as np
import torch
import trimesh

import unbake.guided_trellis2 as G  # sets up xformers for Blackwell before TRELLIS.2 is imported
from unbake.dataset import UnbakeDataset
from unbake.paths import DATA
from unbake.predictions import PREDICTIONS, input_name
from unbake.prior2d import CHANNELS, Prior2D, fuse, load_supermat_sv, supermat_sv_views
from unbake.reference import PITCH_DEG, YAW_DEG, render_unlit
from unbake.uv_bake import TrellisFrame, merge

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--data", default=DATA, help="dataset root")
parser.add_argument("--max-items", type=int, default=None, help="only the first N items")
parser.add_argument("--resolution", type=int, default=1024, choices=(512, 1024))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--beta", type=float, default=1.0, help="share of the 2D prior's normalized gradient per step")
parser.add_argument("--views", type=int, default=16, help="SuperMat single-view renders per input")
parser.add_argument("--guide-t", type=float, nargs=2, default=(0.0, 0.8))
parser.add_argument("--iters", type=int, default=3)
parser.add_argument("--lr", type=float, default=0.05)
parser.add_argument("--huber-delta", type=float, default=0.05)
parser.add_argument("--lambda-roughness", type=float, default=1.0)
parser.add_argument("--lambda-metallic", type=float, default=1.0)
parser.add_argument("--visibility", type=int, default=1, choices=(0, 1), help="visibility in the rebake transfer")
parser.add_argument("--supersample", type=int, default=4, help="UV rasterization density of the observations")
parser.add_argument("--normals", default="smooth", choices=("smooth", "face"))
parser.add_argument("--vis-dirs", type=int, default=256)
parser.add_argument("--vis-res", type=int, default=2048)
parser.add_argument("--reuse-views-from", default=None,
                    help="earlier run (same layout) whose per-input SuperMat predictions are read in place instead of "
                         "running SuperMat")
parser.add_argument("--sm-dir", default=None, help="clone of https://github.com/hyj542682306/SuperMat "
                                                   "(only needed for inputs without saved per-view predictions)")
parser.add_argument("--sm-ckpt-dir", default="/tmp/supermat", help="folder with supermat.pth (oyiya/SuperMat)")
parser.add_argument("--output", default=None,
                    help=f"default: {PREDICTIONS}/guided_supermat_sv_b<beta>[_1024][_vis]_ss<s>[_face]")
args = parser.parse_args()
res = args.resolution
output_dir = args.output or (f"{PREDICTIONS}/guided_supermat_sv_b{args.beta:g}" + ("" if res == 512 else f"_{res}")
                              + (f"_vis_ss{args.supersample}" if args.visibility else f"_ss{args.supersample}")
                              + ("" if args.normals == "smooth" else "_face"))
obs_kwargs = dict(supersample=args.supersample, normals=args.normals)

dataset = UnbakeDataset(args.data)
items = [dataset[i] for i in range(len(dataset))[:args.max_items]]
out_dir = lambda item, lighting: os.path.join(output_dir, input_name(item, lighting))
pending = [(item, [l for l in item.lightings if not os.path.isfile(os.path.join(out_dir(item, l), "meta.json"))])
           for item in items]
pending = [(item, lightings) for item, lightings in pending if lightings]
total = sum(len(l) for _, l in pending)
print(f"{total} predictions to make ({len(items)} items, resolution {res}, beta {args.beta:g}, visibility "
      f"{args.visibility}, supersample {args.supersample}, {args.normals} normals) -> {output_dir}/")
if not total:
    raise SystemExit


def frame_of(glb):
    return TrellisFrame(np.concatenate([p.positions for p in glb.world_primitives()]))


start = time.time()
pipeline = G.load_pipeline(res)

# ---------------------------------------------------------------- 1. conditioning
print(f"[1/3] Image conditioning (DINOv3) for {total} reference images")
conds = {}
with torch.no_grad():
    for item, lightings in pending:
        for lighting in lightings:
            glb = lighting.baked()
            image = render_unlit(glb, frame_of(glb))
            os.makedirs(out_dir(item, lighting), exist_ok=True)
            image.save(os.path.join(out_dir(item, lighting), "cond.png"))
            conds[item.id, lighting.name] = pipeline.get_cond([pipeline.preprocess_image(image)], res,
                                                              include_neg_cond=False)["cond"].cpu()
pipeline.image_cond_model = None
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------- 2. 2D prior: per-view predictions
have_views = lambda d: d is not None and all(os.path.isfile(os.path.join(d, "supermat_sv_pred", f"{k:04d}", f"{c}.png"))
                                             for k in range(args.views) for c in CHANNELS)


def view_dir(item, lighting):
    """Folder with this input's SuperMat per-view predictions, or None."""
    for d in (out_dir(item, lighting),
              os.path.join(args.reuse_views_from, input_name(item, lighting)) if args.reuse_views_from else None):
        if have_views(d):
            return d
    return None


todo = [(item, l) for item, lightings in pending for l in lightings if view_dir(item, l) is None]
print(f"[2/3] SuperMat single-view predictions ({args.views} views) for {len(todo)} inputs")
if todo:
    assert args.sm_dir, "inputs without saved SuperMat predictions: pass --sm-dir"
    supermat = load_supermat_sv(args.sm_dir, args.sm_ckpt_dir)
    for n, (item, lighting) in enumerate(todo):
        t0 = time.time()
        supermat_sv_views(lighting.baked_path, out_dir(item, lighting), supermat, args.views, seed=args.seed)
        print(f"  [{n + 1}/{len(todo)}] {item.id[:12]} {lighting.name} ({time.time() - t0:.0f} s)", flush=True)
    del supermat
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def prior_for(item, lighting, obs):
    """Per-voxel prior fused from the saved per-view predictions (nothing is written)."""
    views, _, depths, preds = supermat_sv_views(lighting.baked_path, view_dir(item, lighting), None, args.views)
    value, weight, nviews = fuse(views, depths, preds, obs["centers"], obs["normals"])
    prior = Prior2D(value, weight, {"roughness": args.lambda_roughness, "metallic": args.lambda_metallic})
    return prior, float((nviews >= 2).float().mean())


# ---------------------------------------------------------------- 3. guided material stage
print("[3/3] Guided material stage")
flow = pipeline.models[f"tex_slat_flow_model_{res}"].cuda()
decoder = pipeline.models["tex_slat_decoder"].cuda()
done = 0
for n, (item, lightings) in enumerate(pending):
    glb0 = lightings[0].baked()
    frame = frame_of(glb0)
    positions, _, faces = merge(glb0.world_primitives(), frame)
    with torch.no_grad():
        shape_slat = pipeline.encode_shape_slat(trimesh.Trimesh(vertices=positions, faces=faces, process=False), res)
    print(f"  item {n + 1}/{len(pending)} {item.id[:12]}: {len(faces)} faces, {shape_slat.feats.shape[0]} latent voxels")
    for lighting in lightings:
        t0 = time.time()
        glb = lighting.baked()
        obs = G.voxel_observations(glb, frame, res, visibility=bool(args.visibility), vis_dirs=args.vis_dirs,
                                   vis_res=args.vis_res, **obs_kwargs)
        prior, seen2 = prior_for(item, lighting, obs)
        guide = G.Rebake(obs, res, huber_delta=args.huber_delta, prior=prior, beta=args.beta)
        cond = conds.pop((item.id, lighting.name)).cuda()
        cond = {"cond": cond, "neg_cond": torch.zeros_like(cond)}
        slat, history = G.guided_sample(pipeline, flow, decoder, cond, shape_slat, guide, args.seed,
                                        tuple(args.guide_t), args.iters, args.lr, log=lambda *a: None)
        with torch.no_grad():
            pbr = decoder(slat) * 0.5 + 0.5
            _, final = guide(pbr)
        first_loss, n_obs = (history[0]["rebake"] if history else None), len(obs["keys"])
        del obs, prior, guide, history  # free the observations / prior before the UV bake (nvdiffrast OOM otherwise)
        gc.collect()
        torch.cuda.empty_cache()
        G.write_prediction(lighting.baked_path, frame, pbr, res, os.path.join(out_dir(item, lighting), "textured.glb"))
        meta = {"input": lighting.baked_path, "method": "guided", "prior": "supermat_sv", "views": args.views,
                "visibility": args.visibility, "supersample": args.supersample, "normals": args.normals,
                "resolution": res, "seed": args.seed, "beta": args.beta, "guide_t": list(args.guide_t),
                "iters": args.iters, "lr": args.lr, "huber_delta": args.huber_delta,
                "lambda": {"roughness": args.lambda_roughness, "metallic": args.lambda_metallic},
                "view_yaw": YAW_DEG, "view_pitch": PITCH_DEG,
                "rebake_loss_first": first_loss, "final": final,
                "prior_seen_2_views": round(seen2, 4), "observed_voxels": n_obs,
                "time_s": round(time.time() - t0, 1)}
        with open(os.path.join(out_dir(item, lighting), "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        done += 1
        elapsed = time.time() - start
        print(f"    [{done}/{total}] {lighting.name}: rebake {meta['rebake_loss_first']:.5f} -> {final['rebake']:.5f}, "
              + ", ".join(f"{k} {v:.3f}" for k, v in final.items() if k.startswith("l2d_"))
              + f"  ({meta['time_s']} s; {elapsed / 60:.1f} min, ~{elapsed / done * (total - done) / 60:.0f} min left)",
              flush=True)
        del slat, pbr, cond
        torch.cuda.empty_cache()
    del shape_slat
    torch.cuda.empty_cache()

print(f"Saved {done} predictions to {output_dir}/")
