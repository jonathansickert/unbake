"""
2D material prior for the guidance: roughness and metallic predicted on unlit orthographic renders of the
baked asset, fused per voxel with confidence weights. Two sources, same fusion:

  rgbx      RGB->X (unbake/baselines/rgbx.py) on N views on a Fibonacci sphere, each view independently.
  supermat     SuperMat's multi-view network (unbake/baselines/supermat.py) on its fixed 6 views (+-X, +-Y,
               +-Z), predicted jointly; no UV refinement (it has no per-view output to measure spread on).
  supermat_sv  SuperMat's single-view network (supermat.pth) on the same N Fibonacci views as rgbx, each view
               independently (the multi-view network is built for exactly its 6 camera-embedded views).

  1. Render the baked texture (unlit) with SuperMat's orthographic setup, with depth maps for visibility.
  2. The source predicts roughness and metallic per view.
  3. Every observed voxel is projected into every view where it is visible (depth test) and faces the
     camera (cos > MIN_COS); the predictions are averaged with weights cos^2.
  4. Per channel k, the weighted standard deviation sigma_k across views gives the confidence weight
     w_k = 1 / (1 + (sigma_k / tau_k)^2), tau_k = median of sigma_k. Voxels seen in fewer than 2 views get
     w = 0.

Albedo is not used: the base color is left to the 3D prior and the rebake loss.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from unbake.baselines.common import load_parts, normalization, normalize
from unbake.baselines.common import to_u8
from unbake.baselines.supermat import EXTENT, VIEWS, c2w_matrix, render_views, to_view

CHANNELS = ("roughness", "metallic")
MIN_COS = 0.1


def fibonacci_views(n):
    """(camera position, up) pairs on the unit sphere (Z up)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = math.pi * (3 - math.sqrt(5)) * i
    r = np.sqrt(1 - z * z)
    pos = np.stack([r * np.cos(phi), r * np.sin(phi), z], -1)
    return [(tuple(p), (0, 1, 0) if abs(p[2]) > 0.95 else (0, 0, 1)) for p in pos]


def _view_parts(input_path):
    parts = load_parts(input_path)
    center, scale = normalization(np.concatenate([p["mesh"].vertices for p in parts]))
    return [{"v": normalize(np.asarray(p["mesh"].vertices), center, scale) * EXTENT,
             "f": np.asarray(p["mesh"].faces, np.int64), "uv": p["uv"], "texture": p["texture"]} for p in parts]


@torch.no_grad()
def supermat_views(input_path, work_dir, sm_dir, ckpt_dir, base_model="sd2-community/stable-diffusion-2-1",
                   render_size=512, seed=42):
    """Unlit renders of the baked asset on SuperMat's 6 views and its multi-view roughness / metallic.
    Returns views, images, depths, preds {channel: [(H, W, 1)]}; renders and predictions go to work_dir."""
    from unbake.baselines.supermat import SuperMat
    work_dir = Path(work_dir).resolve()
    images, depths = render_views(_view_parts(input_path), render_size)
    case, pred_dir = work_dir / "supermat_views", work_dir / "supermat_pred"
    case.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(images):
        Image.fromarray(to_u8(im.cpu().numpy()), "RGBA").save(case / f"color_{k:04d}.png")
    (case / "meta.json").write_text(json.dumps({"locations": [
        {"index": f"{k:04d}", "transform_matrix": c2w_matrix(pos, up).tolist()} for k, (pos, up) in enumerate(VIEWS)]}))
    cwd = os.getcwd()
    try:  # SuperMat chdirs into its repo
        sm = SuperMat(os.path.abspath(sm_dir), os.path.abspath(ckpt_dir), base_model, seed)
        sm.load("mv")
        sm.multiview(case, pred_dir)
        del sm
    finally:
        os.chdir(cwd)
    torch.cuda.empty_cache()
    load = lambda n, k: torch.tensor(np.asarray(Image.open(pred_dir / f"{n}_{k:04d}.png").convert("L").resize(
        (render_size, render_size), Image.BILINEAR)), dtype=torch.float32, device="cuda")[..., None] / 255
    preds = {c: [load(c, k) for k in range(len(VIEWS))] for c in CHANNELS}
    return list(VIEWS), images, depths, preds


class _InDir:
    """Temporarily chdir (SuperMat imports its src/ package relative to its repo)."""

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def __enter__(self):
        self.cwd = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, *exc):
        os.chdir(self.cwd)


def load_supermat_sv(sm_dir, ckpt_dir, base_model="sd2-community/stable-diffusion-2-1"):
    """SuperMat single-view pipeline: (inference module, pipeline, repo dir)."""
    import argparse
    import sys
    sm_dir = os.path.abspath(sm_dir)
    sys.path.insert(0, sm_dir)
    with _InDir(sm_dir):
        import inference_supermat as sv
        pipe = sv.build_pipeline(argparse.Namespace(base_model=base_model, device="cuda",
                                                    checkpoint=os.path.join(os.path.abspath(ckpt_dir), "supermat.pth")))
    return sv, pipe, sm_dir


@torch.no_grad()
def supermat_sv_views(input_path, work_dir, supermat, num_views, render_size=512, seed=42):
    """Unlit renders on N Fibonacci views and SuperMat single-view roughness / metallic for each
    (supermat: from load_supermat_sv). Returns views, images, depths, preds {channel: [(H, W, 1)]};
    renders and predictions go to work_dir."""
    sv, pipe, sm_dir = supermat if supermat is not None else (None, None, None)
    work_dir = Path(work_dir).resolve()
    views = fibonacci_views(num_views)
    images, depths = render_views(_view_parts(input_path), render_size, views)
    case, pred_dir = work_dir / "supermat_sv_views", work_dir / "supermat_sv_pred"
    case.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(images):
        Image.fromarray(to_u8(im.cpu().numpy()), "RGBA").save(case / f"color_{k:04d}.png")
    if supermat is not None:  # None: reuse the predictions already in work_dir
        with _InDir(sm_dir):
            for k in range(len(views)):
                sv.run_one_image(pipe, case / f"color_{k:04d}.png", pred_dir / f"{k:04d}", 512, seed, False,
                                 torch.device("cuda"))
    load = lambda n, k: torch.tensor(np.asarray(Image.open(pred_dir / f"{k:04d}" / f"{n}.png").convert("L").resize(
        (render_size, render_size), Image.BILINEAR)), dtype=torch.float32, device="cuda")[..., None] / 255
    preds = {c: [load(c, k) for k in range(len(views))] for c in CHANNELS}
    return views, images, depths, preds


@torch.no_grad()
def rgbx_views(input_path, pipe, num_views, render_size=512, size=512, steps=50, seed=42):
    """Unlit renders of the baked asset and RGB->X roughness / metallic per view.
    Returns views, images [(H, W, 4)], depths [(H, W)], preds {channel: [(H, W, C)]}."""
    from unbake.baselines.rgbx import PROMPTS, predict
    views = fibonacci_views(num_views)
    images, depths = render_views(_view_parts(input_path), render_size, views)
    saved = dict(PROMPTS)
    for c in [c for c in PROMPTS if c not in CHANNELS]:  # predict() loops over PROMPTS
        PROMPTS.pop(c)
    try:
        preds = {c: [] for c in CHANNELS}
        for im in images:
            p = predict(pipe, im, size, steps, seed, 0.0)
            for c in CHANNELS:
                x = F.interpolate(p[c].permute(2, 0, 1)[None], (render_size,) * 2, mode="bilinear", align_corners=False)
                preds[c].append(x[0].permute(1, 2, 0))
    finally:
        PROMPTS.clear(); PROMPTS.update(saved)
    return views, images, depths, preds


@torch.no_grad()
def fuse(views, depths, preds, pos, nrm, depth_tol=0.01):
    """
    pos, nrm: voxel centers / normals (M, 3) in TRELLIS space. Returns {channel: value (M, C)},
    {channel: weight (M,)}, number of views (M,).
    """
    size = depths[0].shape[0]
    p = pos * EXTENT
    vals = {c: [] for c in CHANNELS}
    wts = []
    for (cam, up), dmap, k in zip(views, depths, range(len(views))):
        x, y, d = to_view(p, c2w_matrix(cam, up))
        to_cam = torch.tensor(cam, dtype=torch.float32, device=p.device)
        cos = (nrm @ (to_cam / to_cam.norm())).clamp(min=0)
        px = ((x / 0.5 + 1) / 2 * size).long().clamp(0, size - 1)
        py = ((1 - (y / 0.5 + 1) / 2) * size).long().clamp(0, size - 1)
        visible = (d <= dmap[py, px] + depth_tol) & (cos > MIN_COS)
        grid = torch.stack([x / 0.5, -(y / 0.5)], -1)[None, None]
        for c in CHANNELS:
            m = preds[c][k]
            vals[c].append(F.grid_sample(m.permute(2, 0, 1)[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T)
        wts.append(cos ** 2 * visible)
    W = torch.stack(wts)                       # (V, M)
    nviews = (W > 0).sum(0)
    wsum = W.sum(0).clamp(min=1e-8)
    value, weight = {}, {}
    for c in CHANNELS:
        V = torch.stack(vals[c])               # (V, M, C)
        mean = (W[..., None] * V).sum(0) / wsum[:, None]
        sigma = ((W[..., None] * (V - mean) ** 2).sum(0) / wsum[:, None]).sqrt().mean(-1)
        ok = nviews >= 2
        tau = sigma[ok].median().clamp(min=1e-6) if ok.any() else torch.tensor(1.0, device=p.device)
        w = 1 / (1 + (sigma / tau) ** 2)
        value[c], weight[c] = mean, torch.where(ok, w, torch.zeros_like(w))
    return value, weight, nviews


class Prior2D:
    """L2D = sum_k lambda_k * sum_i w_ik |p_ik - p~_ik| / sum_i w_ik over the matched voxels."""

    def __init__(self, value, weight, lambdas):
        self.value, self.weight, self.lambdas = value, weight, lambdas

    def __call__(self, pbr_feats, obs_idx, layout):
        loss, stats = 0.0, {}
        for c, lam in self.lambdas.items():
            if lam == 0:
                continue
            pred = pbr_feats[:, layout[c]].float()
            w = self.weight[c][obs_idx]
            l = (w[:, None] * (pred - self.value[c][obs_idx]).abs()).mean(-1).sum() / w.sum().clamp(min=1e-8)
            loss = loss + lam * l
            stats[f"l2d_{c}"] = l.item()
        return loss, stats
