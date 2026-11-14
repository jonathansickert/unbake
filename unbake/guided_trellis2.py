"""
UnBake, basic version: TRELLIS.2 material stage guided by a rebake loss under order-2 spherical harmonics
(SH) lighting.

At selected steps of the rectified-flow sampler, the clean-latent estimate x0 is decoded with the frozen
material decoder into per-voxel base color c_i. The observed (baked) color o_i of each voxel is modeled as
Lambertian shading under SH lighting L (9 x 3):

    o_i = c_i * (t_i^T L),    t_i = A_l Y_lm(n_i)                                (clamped cosine), or
                              t_i = 4pi/K sum_d V_i(d) max(0, n_i . d) Y(d)   (with visibility, --visibility)

V_i(d) is 1 where voxel i sees direction d unoccluded, from orthographic shadow maps of the whole mesh in
K directions, so baked shadows and occlusion can be explained by the lighting instead of the base color. The
observations are the baked texture sampled at the surface points of a UV rasterization `supersample` times
the texture size (bilinear), averaged per voxel; more samples reach more of the 1024^3 voxels.

L is fitted by least squares (c_i detached), then x0 is corrected by a few normalized gradient steps on
Huber(o_hat - o) through the decoder only, and sampling continues from the corrected x0 with the predicted
noise. All networks stay frozen. Colors are compared in linear space (sRGB ~ x^2.2).

Optionally (default on) the 2D prior L2D on roughness and metallic only: fused RGB->X (16 views) or SuperMat (6 views)
estimates with confidence weights (unbake/prior2d.py), total loss L = L_rebake + beta * L2D.

Single asset, guided vs unguided with the same seed, scored against the ground truth if given:

    python -m unbake.guided_trellis2 --input examples/cauldron_baked.glb --gt examples/cauldron_pbr.glb \
        --out output/guided/cauldron
"""
import os

os.environ.setdefault("HF_HOME", "/tmp/hf_cache/")
os.environ["ATTN_BACKEND"] = "xformers"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import time

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from xformers.ops.fmha import dispatch as _xformers_dispatch

_xformers_dispatch._set_use_fa3(False)  # FA3 kernel crashes on Blackwell (see unguided_trellis2.py)

from trellis2 import models
from trellis2.pipelines import Trellis2TexturingPipeline, rembg

from unbake.evaluation.metrics import material_maps, uv_metrics
from unbake.gltf_io import Glb
from unbake.reference import render_unlit
from unbake.uv_bake import PBR_LAYOUT, TrellisFrame, _rasterize_uv, bake_voxels_to_uv, merge, to_gltf_images

REPO = "microsoft/TRELLIS.2-4B"
GAMMA = 2.2
# clamped-cosine convolution per SH band (Ramamoorthi & Hanrahan 2001)
SH_BAND_GAIN = torch.tensor([math.pi] + [2 * math.pi / 3] * 3 + [math.pi / 4] * 5)


def sh9(n: torch.Tensor) -> torch.Tensor:
    """Real SH basis up to order 2 at unit directions n (N, 3) -> (N, 9)."""
    x, y, z = n.unbind(-1)
    return torch.stack([
        torch.full_like(x, 0.282095),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1), 1.092548 * x * z, 0.546274 * (x * x - y * y),
    ], -1)


def fibonacci_dirs(n):
    i = torch.arange(n, dtype=torch.float32) + 0.5
    z = 1 - 2 * i / n
    phi = math.pi * (3 - math.sqrt(5)) * i
    r = (1 - z * z).sqrt()
    return torch.stack([r * phi.cos(), r * phi.sin(), z], -1)


@torch.no_grad()
def visibility_transfer(centers, normals, positions, faces, resolution, num_dirs=256, map_res=2048, half_extent=0.9):
    """
    SH transfer of the cosine-weighted visibility per point (M, 9): 4pi/K sum_d V(x, d) max(0, n . d) Y(d).
    V from an orthographic depth map of the mesh per direction (TRELLIS space, object within radius 0.87);
    points are offset along the normal by 1.5 voxels against self-shadowing.
    """
    from unbake.baselines.supermat import c2w_matrix, to_view
    ctx = dr.RasterizeCudaContext()
    v = torch.as_tensor(positions, dtype=torch.float32, device="cuda")
    f = torch.as_tensor(faces, dtype=torch.int32, device="cuda")
    p = centers + normals * (1.5 / resolution)
    tol = 2 * (2 * half_extent / map_res)
    dirs = fibonacci_dirs(num_dirs)
    T = torch.zeros(len(p), 9, device="cuda")
    unoccluded = torch.zeros(len(p), device="cuda")
    for d in dirs:
        dn = d.numpy().astype(np.float64)
        c2w = c2w_matrix(dn, (0, 1, 0) if abs(dn[2]) > 0.95 else (0, 0, 1))
        x, y, dep = to_view(v, c2w)
        clip = torch.stack([x / half_extent, y / half_extent, dep - 1, torch.ones_like(dep)], -1)[None].contiguous()
        rast, _ = dr.rasterize(ctx, clip, f, (map_res, map_res))
        dmap = dr.interpolate(dep[None, :, None].contiguous(), rast, f)[0][0, ..., 0]
        dmap = torch.where(rast[0, ..., 3] > 0, dmap, torch.full_like(dmap, 1e9))
        xp, yp, dp = to_view(p, c2w)
        px = ((xp / half_extent + 1) / 2 * map_res).long().clamp(0, map_res - 1)
        py = ((yp / half_extent + 1) / 2 * map_res).long().clamp(0, map_res - 1)  # GL rows bottom-up, as rasterized
        vis = (dp <= dmap[py, px] + tol).float()
        cos = (normals @ d.cuda()).clamp(min=0)
        T += (vis * cos)[:, None] * sh9(d.cuda()[None])
        unoccluded += vis * (cos > 0)
    hemisphere = (normals @ dirs.cuda().T > 0).sum(-1).clamp(min=1)
    return T * (4 * math.pi / num_dirs), unoccluded / hemisphere


# ------------------------------------------------------------------------------------------- observations
@torch.no_grad()
def voxel_observations(glb: Glb, frame: TrellisFrame, resolution: int, supersample: int = 1, visibility: bool = False,
                       vis_dirs: int = 256, vis_res: int = 2048):
    """
    Observed baked color (linear) and surface normal per voxel of the `resolution` grid, averaged over the
    surface samples (UV rasterization at `supersample` x texture size, texture sampled bilinearly) that fall
    into the voxel. Returns {keys (M,) sorted, colors (M, 3), transfer (M, 9), centers (M, 3) and normals (M, 3)
    in TRELLIS space, visible_fraction (M,) (1 without visibility)}.
    """
    keys, cols, nrms = [], [], []
    prims = glb.world_primitives()
    for material in sorted({p.material for p in prims if p.material is not None}):
        group = [p for p in prims if p.material == material]
        img = glb.material_base_color_image(material)
        size = glb.image(img).size if img is not None else (1024, 1024)
        tex = torch.from_numpy(material_maps(glb, material, size)["base_color"]).cuda()  # (H, W, 3) sRGB
        positions, uvs, faces = merge(group, frame)
        rast, faces_t = _rasterize_uv(uvs, faces, (size[0] * supersample, size[1] * supersample))
        tri = rast[0, ..., 3].long() - 1
        mask = tri >= 0
        pos_t = torch.from_numpy(positions).cuda()
        p0, p1, p2 = (pos_t[faces_t[:, k].long()] for k in range(3))
        face_n = F.normalize(torch.cross(p1 - p0, p2 - p0, dim=-1), dim=-1)
        texel_pos = dr.interpolate(pos_t[None], rast, faces_t)[0][0][mask]
        uv = dr.interpolate(torch.from_numpy(uvs).cuda()[None], rast, faces_t)[0][0][mask] % 1  # glTF repeat
        color = F.grid_sample(tex.permute(2, 0, 1)[None], (uv * 2 - 1)[None, None], mode="bilinear",
                              padding_mode="border", align_corners=False)[0, :, 0].T  # texel centers: exact at 1x
        ijk = ((texel_pos + 0.5) * resolution).floor().long().clamp(0, resolution - 1)
        keys.append((ijk[:, 0] * resolution + ijk[:, 1]) * resolution + ijk[:, 2])
        cols.append(color.clamp(0, 1) ** GAMMA)
        nrms.append(face_n[tri[mask]])
    keys, cols, nrms = torch.cat(keys), torch.cat(cols), torch.cat(nrms)
    uniq, inv = torch.unique(keys, return_inverse=True)
    count = torch.zeros(len(uniq), device="cuda").index_add_(0, inv, torch.ones_like(keys, dtype=torch.float32))
    col = torch.zeros(len(uniq), 3, device="cuda").index_add_(0, inv, cols) / count[:, None]
    nsum = torch.zeros(len(uniq), 3, device="cuda").index_add_(0, inv, nrms)
    coherent = nsum.norm(dim=-1) > 0.5 * count  # drop voxels with opposing surfaces (thin parts)
    n = F.normalize(nsum, dim=-1)
    keep, n = uniq[coherent], n[coherent]
    ijk = torch.stack([keep // resolution ** 2, keep // resolution % resolution, keep % resolution], -1)
    centers = (ijk.float() + 0.5) / resolution - 0.5
    if visibility:
        positions, _, faces = merge(prims, frame)
        transfer, visible = visibility_transfer(centers, n, positions, faces, resolution, vis_dirs, vis_res)
    else:
        transfer, visible = sh9(n) * SH_BAND_GAIN.cuda(), torch.ones(len(keep), device="cuda")
    return {"keys": keep, "colors": col[coherent], "transfer": transfer, "centers": centers, "normals": n,
            "visible_fraction": visible}


class Rebake:
    """Rebake loss of decoded voxels against the observations (lighting refit by least squares), plus
    beta * L2D of an optional 2D prior (unbake/prior2d.py)."""

    def __init__(self, obs, resolution, huber_delta=0.05, ridge=1e-4, prior=None, beta=0.0):
        self.keys, self.colors, self.transfer = obs["keys"], obs["colors"], obs["transfer"]
        self.resolution, self.delta, self.ridge = resolution, huber_delta, ridge
        self.prior, self.beta = prior, beta
        self.light = None

    def match(self, coords):
        r = self.resolution
        k = (coords[:, 1].long() * r + coords[:, 2].long()) * r + coords[:, 3].long()
        pos = torch.searchsorted(self.keys, k).clamp(max=len(self.keys) - 1)
        hit = self.keys[pos] == k
        return hit.nonzero()[:, 0], pos[hit]

    def __call__(self, pbr):
        """pbr: decoded SparseTensor, feats in [0, 1]. Returns (rebake + beta * L2D, stats)."""
        terms, stats = self.terms(pbr)
        return terms["rebake"] + self.beta * terms.get("l2d", 0.0), stats

    def terms(self, pbr):
        """Unweighted loss terms {"rebake": ..., "l2d": ... (if a prior is used)} and stats."""
        dec_idx, obs_idx = self.match(pbr.coords)
        c = pbr.feats[dec_idx, PBR_LAYOUT["base_color"]].float().clamp(1e-4, 1) ** GAMMA
        T, o = self.transfer[obs_idx], self.colors[obs_idx]
        with torch.no_grad():  # per channel: min || (c_k * T) L_k - o_k ||^2
            light = []
            for ch in range(3):
                A = c[:, ch:ch + 1].detach() * T
                light.append(torch.linalg.solve(A.T @ A + self.ridge * len(A) * torch.eye(9, device=A.device), A.T @ o[:, ch]))
            self.light = torch.stack(light, -1)  # (9, 3)
        shading = (T @ self.light).clamp(min=0)
        pred = c * shading
        loss = F.huber_loss(pred, o, delta=self.delta)
        stats = {"matched": len(dec_idx) / max(len(pbr.feats), 1), "mae": (pred - o).abs().mean().item(),
                 "rebake": loss.item()}
        terms = {"rebake": loss}
        if self.prior is not None and self.beta > 0:
            terms["l2d"], s2d = self.prior(pbr.feats[dec_idx], obs_idx, PBR_LAYOUT)
            stats.update(s2d)
        return terms, stats


# ----------------------------------------------------------------------------------------- guided sampling
def guided_sample(pipeline, flow, decoder, cond, shape_slat, rebake, seed, guide_t=(0.0, 0.8), iters=3, lr=0.05,
                  log=print):
    """TRELLIS.2 texture-latent sampling (as Trellis2TexturingPipeline.sample_tex_slat) with rebake guidance at
    the steps with guide_t[0] <= t <= guide_t[1]. rebake=None gives the unguided sample.

    The gradient of each loss term is normalized to unit RMS on its own, the terms are combined as
    g_rebake + beta * g_2d, and the sum is rescaled to RMS `lr`: beta sets the 2D prior's share of every step
    (independent of the terms' magnitudes) and the step size does not depend on beta."""
    sampler = pipeline.tex_slat_sampler
    params = dict(pipeline.tex_slat_sampler_params)
    steps, rescale_t = params.pop("steps", 50), params.pop("rescale_t", 1.0)
    s_std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].cuda()
    s_mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].cuda()
    t_std = torch.tensor(pipeline.tex_slat_normalization["std"])[None].cuda()
    t_mean = torch.tensor(pipeline.tex_slat_normalization["mean"])[None].cuda()
    shape_n = (shape_slat - s_mean) / s_std
    torch.manual_seed(seed)
    x = shape_n.replace(feats=torch.randn(shape_n.coords.shape[0], flow.in_channels - shape_n.feats.shape[1]).cuda())
    t_seq = np.linspace(1, 0, steps + 1)
    t_seq = (rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)).tolist()
    sig = sampler.sigma_min
    history = []
    for i in range(steps):
        t, t_prev = t_seq[i], t_seq[i + 1]
        with torch.no_grad():
            x0, eps, v = sampler._get_model_prediction(flow, x, t, cond["cond"], neg_cond=cond["neg_cond"],
                                                       concat_cond=shape_n, **params)
        if rebake is None or not (guide_t[0] <= t <= guide_t[1]):
            x = x - (t - t_prev) * v
            continue
        f = x0.feats.detach().clone().requires_grad_(True)
        for k in range(iters):
            with torch.enable_grad():
                pbr = decoder(x0.replace(f) * t_std + t_mean) * 0.5 + 0.5
                terms, stats = rebake.terms(pbr)
                # loss scaled by 1e4: the decoder runs in fp16, small gradients underflow
                grads = [torch.autograd.grad(l * 1e4, f, retain_graph=n < len(terms) - 1)[0]
                         for n, l in enumerate(terms.values())]
            rms = lambda x: x.pow(2).mean().sqrt() + 1e-12
            weights = {"rebake": 1.0, "l2d": rebake.beta}
            g = sum(weights[name] * gk / rms(gk) for name, gk in zip(terms, grads))
            if not history or not torch.isfinite(g).all() or g.abs().max() == 0:
                log("    grad RMS: " + ", ".join(f"{name} {rms(gk).item():.3e}" for name, gk in zip(terms, grads)))
            loss = terms["rebake"] + rebake.beta * terms.get("l2d", 0.0)
            history.append({"step": i, "t": round(t, 4), "iter": k, "loss": loss.item(), **stats})
            f = (f - lr * g / rms(g)).detach().requires_grad_(True)
            del pbr, loss, g, grads, terms
        extra = "".join(f", {k} {history[-1][k]:.4f}" for k in history[-1] if k.startswith("l2d_"))
        log(f"    step {i:2d} t={t:.3f}: loss {history[-iters]['loss']:.5f} -> {history[-1]['loss']:.5f} "
            f"(rebake {history[-1]['rebake']:.5f}, MAE {history[-1]['mae']:.4f}{extra}, matched {history[-1]['matched']:.0%})")
        x0 = x0.replace(f.detach())
        x = x0 * (1 - t_prev) + eps * (sig + (1 - sig) * t_prev)  # re-interpolate with the predicted noise
    return x * t_std + t_mean, history


# --------------------------------------------------------------------------------------------------- main
def load_pipeline(resolution):
    rembg.BiRefNet = lambda **kwargs: None
    Trellis2TexturingPipeline.model_names_to_load = []
    pipeline = Trellis2TexturingPipeline.from_pretrained(REPO, config_file="texturing_pipeline.json")
    pipeline.low_vram = True
    pipeline.cuda()
    for name in ("shape_slat_encoder", f"tex_slat_flow_model_{resolution}", "tex_slat_decoder"):
        path = pipeline._pretrained_args["models"][name]
        pipeline.models[name] = models.from_pretrained(f"{REPO}/{path}" if path.startswith("ckpts/") else path).eval()
    # The decoder weights keep requires_grad=True: flex_gemm's sparse-conv backward crashes when they don't
    # (it reshapes a None weight gradient). Only d loss / d latent is taken (autograd.grad), so they never change.
    if resolution == 1024:  # backprop through the decoder at 1024^3 does not fit in 24 GB without checkpointing
        for m in pipeline.models["tex_slat_decoder"].modules():
            if hasattr(m, "use_checkpoint"):
                m.use_checkpoint = True
    return pipeline


def write_prediction(glb_path, frame, pbr, resolution, out_path):
    glb = Glb(glb_path)
    prims = glb.world_primitives()
    for material in sorted({p.material for p in prims if p.material is not None}):
        img = glb.material_base_color_image(material)
        size = glb.image(img).size if img is not None else (1024, 1024)
        maps = bake_voxels_to_uv([p for p in prims if p.material == material], frame, pbr, resolution, size)
        glb.set_pbr_textures(material, *to_gltf_images(maps))
    glb.save(out_path)


def prior_2d(args, obs, res, tag=""):
    """Fused 2D roughness / metallic per observed voxel (RGB->X or SuperMat), cached in <out>/<prior>_prior<tag>.pt.
    For supermat_sv, per-view predictions already in <out>/supermat_sv_pred are reused (no SuperMat run)."""
    from unbake.prior2d import CHANNELS, Prior2D, fuse, load_supermat_sv, rgbx_views, supermat_sv_views, supermat_views
    cache = os.path.join(args.out, f"{args.prior}_prior{tag}.pt")
    key = {"input": os.path.abspath(args.input), "prior": args.prior, "resolution": res, "channels": CHANNELS,
           "views": 6 if args.prior == "supermat" else args.prior_views, "voxels": len(obs["keys"]),
           "first_key": int(obs["keys"][0])}
    data = torch.load(cache) if os.path.isfile(cache) else None
    if data is None or data.get("key") != key:
        t0 = time.time()
        if args.prior == "rgbx":
            from unbake.baselines.rgbx import load_rgbx
            pipe = load_rgbx(args.rgbx_dir, args.rgbx_weights)
            views, images, depths, preds = rgbx_views(args.input, pipe, args.prior_views)
            del pipe
        elif args.prior == "supermat":
            assert args.sm_dir, "--prior supermat needs --sm-dir"
            views, images, depths, preds = supermat_views(args.input, args.out, args.sm_dir, args.sm_ckpt_dir)
        else:
            pred_dir = os.path.join(args.out, "supermat_sv_pred")
            have = all(os.path.isfile(os.path.join(pred_dir, f"{k:04d}", f"{c}.png"))
                       for k in range(args.prior_views) for c in CHANNELS)
            assert have or args.sm_dir, "--prior supermat_sv needs --sm-dir (no saved per-view predictions)"
            supermat = None if have else load_supermat_sv(args.sm_dir, args.sm_ckpt_dir)
            views, images, depths, preds = supermat_sv_views(args.input, args.out, supermat, args.prior_views)
            del supermat
        torch.cuda.empty_cache()
        value, weight, nviews = fuse(views, depths, preds, obs["centers"], obs["normals"])
        data = {"key": key, "value": value, "weight": weight, "nviews": nviews}
        torch.save(data, cache)
        cols = 4 if len(images) % 4 == 0 else 3
        grid = lambda ims: torch.cat([torch.cat(ims[r * cols:(r + 1) * cols], 1) for r in range(len(ims) // cols)], 0)
        from PIL import Image
        from unbake.baselines.common import to_u8
        for name, ims in [("views", [im[..., :3] for im in images])] + [(c, [x.expand(-1, -1, 3) for x in preds[c]])
                                                                         for c in CHANNELS]:
            Image.fromarray(to_u8(grid(ims).cpu().numpy())).save(os.path.join(args.out, f"{args.prior}_{name}.png"))
        print(f"{args.prior} prior: {len(views)} views in {time.time() - t0:.0f} s")
    nv = data["nviews"]
    print(f"{args.prior} prior: voxels seen in >= 2 views: {(nv >= 2).float().mean():.0%}, "
          + ", ".join(f"{c} mean weight {w.mean():.2f}" for c, w in data["weight"].items()))
    return Prior2D(data["value"], data["weight"], {"roughness": args.lambda_roughness, "metallic": args.lambda_metallic})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="baked-lighting .glb")
    ap.add_argument("--gt", default=None, help="ground-truth PBR .glb (same geometry) for UV metrics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=512, choices=(512, 1024))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guide-t", type=float, nargs=2, default=(0.0, 0.8), help="guide steps with t in [lo, hi]")
    ap.add_argument("--iters", type=int, default=3, help="gradient steps per guided sampler step")
    ap.add_argument("--lr", type=float, default=0.05, help="RMS step size in normalized latent units")
    ap.add_argument("--huber-delta", type=float, default=0.05)
    ap.add_argument("--no-unguided", action="store_true", help="skip the unguided reference run")
    ap.add_argument("--prior", choices=("rgbx", "supermat", "supermat_sv", "none"), default="rgbx",
                    help="source of the 2D roughness / metallic prior (unbake/prior2d.py)")
    ap.add_argument("--prior-views", type=int, default=16, help="views for the rgbx / supermat_sv priors")
    ap.add_argument("--rgbx-weights", default="/tmp/rgb-to-x")
    ap.add_argument("--rgbx-dir", default="rgbx")
    ap.add_argument("--sm-dir", default=None, help="clone of https://github.com/hyj542682306/SuperMat")
    ap.add_argument("--sm-ckpt-dir", default="/tmp/supermat", help="folder with supermat_mv.pth / supermat.pth (oyiya/SuperMat)")
    ap.add_argument("--betas", type=float, nargs="*", default=[0.01, 0.1],
                    help="one rebake + prior run per beta (besides unguided and rebake only): share of the 2D prior's "
                         "normalized gradient in each step, relative to the rebake's")
    ap.add_argument("--lambda-roughness", type=float, default=1.0)
    ap.add_argument("--lambda-metallic", type=float, default=1.0)
    ap.add_argument("--visibility", type=int, nargs="+", default=[0], choices=(0, 1),
                    help="rebake transfer without (0) / with (1) visibility; one run set per value")
    ap.add_argument("--supersample", type=int, nargs="+", default=[1],
                    help="UV rasterization density for the observations (x texture size); one run set per value")
    ap.add_argument("--vis-dirs", type=int, default=256)
    ap.add_argument("--vis-res", type=int, default=2048)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = args.resolution

    glb = Glb(args.input)
    prims = glb.world_primitives()
    frame = TrellisFrame(np.concatenate([p.positions for p in prims]))
    pipeline = load_pipeline(res)

    with torch.no_grad():
        image = render_unlit(glb, frame)
        image.save(os.path.join(args.out, "cond.png"))
        cond = pipeline.get_cond([pipeline.preprocess_image(image)], res, include_neg_cond=False)["cond"]
        cond = {"cond": cond, "neg_cond": torch.zeros_like(cond)}
        pipeline.image_cond_model = None
        torch.cuda.empty_cache()
        positions, _, faces = merge(prims, frame)
        shape_slat = pipeline.encode_shape_slat(trimesh.Trimesh(vertices=positions, faces=faces, process=False), res)
    print(f"{shape_slat.feats.shape[0]} latent voxels at {res}^3")

    flow = pipeline.models[f"tex_slat_flow_model_{res}"].cuda()
    decoder = pipeline.models["tex_slat_decoder"].cuda()
    configs = [(v, ss) for v in args.visibility for ss in args.supersample]
    single = len(configs) == 1
    results = {}
    for ci, (vis, ss) in enumerate(configs):
        tag = "" if single else f"vis{vis}_ss{ss}"
        t0 = time.time()
        obs = voxel_observations(glb, frame, res, supersample=ss, visibility=bool(vis),
                                 vis_dirs=args.vis_dirs, vis_res=args.vis_res)
        print(f"== {tag or 'observations'}: {len(obs['keys'])} observed voxels, "
              f"mean unoccluded hemisphere {obs['visible_fraction'].mean():.2f} ({time.time() - t0:.0f} s)")
        prior = prior_2d(args, obs, res, "_" + tag if tag else "") if args.prior != "none" else None
        rebake = Rebake(obs, res, huber_delta=args.huber_delta)
        runs = [] if (args.no_unguided or ci > 0) else [("unguided", None)]
        runs += [("rebake", rebake)]
        if prior is not None:
            runs += [(f"rebake+{args.prior}_b{b:g}", Rebake(obs, res, huber_delta=args.huber_delta, prior=prior, beta=b))
                     for b in args.betas]
        for name, guide in runs:
            label = name if name == "unguided" or not tag else f"{tag} {name}"
            t0 = time.time()
            print(f"[{label}]")
            slat, history = guided_sample(pipeline, flow, decoder, cond, shape_slat, guide, args.seed,
                                          tuple(args.guide_t), args.iters, args.lr)
            with torch.no_grad():
                pbr = decoder(slat) * 0.5 + 0.5
                final_loss, final_stats = rebake(pbr)  # rebake term only (this config's observations)
            out_path = os.path.join(args.out, f"{label.replace(' ', '_')}.glb")
            write_prediction(args.input, frame, pbr, res, out_path)
            r = {"time_s": round(time.time() - t0, 1), "visibility": vis, "supersample": ss,
                 "final_rebake_loss": final_loss.item(), "final_rebake_mae": final_stats["mae"],
                 "matched": final_stats["matched"], "light_sh": rebake.light.cpu().tolist(), "history": history}
            if args.gt:
                r["uv"] = uv_metrics(Glb(args.gt), Glb(out_path))
            if prior is not None:  # L2D of the final sample, for every run
                dec_idx, obs_idx = rebake.match(pbr.coords)
                r["final_l2d"] = prior(pbr.feats[dec_idx], obs_idx, PBR_LAYOUT)[1]
            results[label] = r
            print(f"  final rebake loss {final_loss.item():.5f}, matched {final_stats['matched']:.0%}, {r['time_s']} s"
                  + (f", base color MAE {r['uv']['base_color']['MAE']:.4f}" if args.gt else ""))
            del pbr, slat
            torch.cuda.empty_cache()
        del obs, prior, rebake
        torch.cuda.empty_cache()
    if args.gt:
        results["input as albedo"] = {"uv": uv_metrics(Glb(args.gt), Glb(args.input), {"roughness": 0.5, "metallic": 0.0})}
        print("\nUV metrics (MAE / PSNR):")
        for name, r in results.items():
            print(f"  {name:34s} " + "  ".join(f"{c} {r['uv'][c]['MAE']:.4f} / {r['uv'][c]['PSNR']:.2f}"
                                             for c in ("base_color", "roughness", "metallic")))
    json.dump({"args": vars(args), "results": results}, open(os.path.join(args.out, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
