"""
Part-level material metrics: segment an asset into parts with SAM on renders of its baked texture, lift the
masks into the ground truth's UV layout, and score roughness / metallic per part.

  1. N unlit orthographic renders of the baked asset (the 2D prior's Fibonacci views, unbake.prior2d).
  2. SAM automatic mask generation per view; every object pixel gets the smallest mask covering it.
  3. Every texel of every material's UV atlas is projected into every view (depth test, cos > 0.1) and
     picks up that view's mask id.
  4. Neighbouring texels (UV grid, also close in 3D) are joined if the views that see both put them in
     the same mask more often than in different masks. Components are then merged across UV seams,
     materials and disconnected pieces by the same vote on their majority mask per view. Small and unseen
     pieces go to the nearest part in UV.
  5. Metrics (surface-area weighted, as evaluate_uv): texel MAE, and MAE after replacing every texel by its
     part's weighted median (part-constant prediction). Oracles: the ground truth projected onto the SAM
     parts (best any part-constant prediction can do), and parts from the ground truth itself.

Single asset (the dataset driver is unbake.evaluation.part_metrics_dataset):

    python -m unbake.evaluation.part_metrics --gt examples/cauldron_pbr.glb --input examples/cauldron_baked.glb \
        --glb "Unguided=output/guided/cauldron_normals_1024/unguided.glb" ... \
        --supermat-pred output/guided/cauldron_normals_1024/supermat_sv_pred --out output/parts/cauldron
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from unbake.baselines.common import to_u8
from unbake.baselines.supermat import EXTENT, c2w_matrix, render_views, to_view
from unbake.evaluation.metrics import match_materials, material_maps
from unbake.gltf_io import Glb
from unbake.prior2d import MIN_COS, _view_parts, fibonacci_views, fuse
from unbake.uv_bake import TrellisFrame, _rasterize_uv, merge, uv_surface_weights

CHANNELS = ("roughness", "metallic")
CONSTANT = "Constant (r 0.5, m 0)"


# ------------------------------------------------------------------ SAM per view
def load_sam():
    from transformers import pipeline
    return pipeline("mask-generation", model="facebook/sam-vit-huge", device=0, dtype=torch.float32)


def sam_labels(gen, images, points_per_batch=64, verbose=False):
    """Per view an (H, W) int16 map: id of the smallest SAM mask covering the pixel, -1 for none / background."""
    labels = []
    for k, im in enumerate(images):
        rgba = im.cpu().numpy()
        alpha = rgba[..., 3] > 0.5
        rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:])  # on white
        out = gen(Image.fromarray(to_u8(rgb)), points_per_batch=points_per_batch)
        raw = [np.asarray(m, bool) for m in out["masks"]]
        masks = [m & alpha for m in raw]
        masks = [m for m, r in zip(masks, raw) if m.sum() > 0.5 * r.sum()]  # drop background masks
        lab = np.full(alpha.shape, -1, np.int16)
        for j in np.argsort([-m.sum() for m in masks]):  # largest first, so the smallest covering mask wins
            lab[masks[j]] = j
        labels.append(lab)
        if verbose:
            print(f"  view {k}: {len(masks)} masks, {100 * (lab >= 0)[alpha].mean():.0f}% of object pixels labeled")
    return labels


# ------------------------------------------------------------------ texel atlases
def atlases(gt: Glb):
    """One entry per ground-truth material with a base color texture: texel surface points (TRELLIS space),
    face normals, mask, area weights and ground-truth maps at the texture's size."""
    all_prims = gt.world_primitives()
    frame = TrellisFrame(np.concatenate([p.positions for p in all_prims]))
    import nvdiffrast.torch as dr
    out = []
    for material in sorted({p.material for p in all_prims if p.material is not None}):
        prims = [p for p in all_prims if p.material == material]
        img = gt.material_base_color_image(material)
        size = gt.image(img).size if img is not None else (1024, 1024)
        pos, uvs, faces = merge(prims, frame)
        rast, faces_t = _rasterize_uv(uvs, faces, size)
        tri = rast[0, ..., 3].long() - 1
        tpos = dr.interpolate(torch.from_numpy(pos).cuda()[None], rast, faces_t)[0][0]
        p0, p1, p2 = (pos[faces[:, i]] for i in range(3))
        fn = np.cross(p1 - p0, p2 - p0)
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        tnrm = torch.from_numpy(fn.astype(np.float32)).cuda()[tri.clamp(min=0)]
        mask = (tri >= 0).cpu().numpy()
        w = uv_surface_weights(prims, size)
        out.append({"material": material, "size": size, "mask": mask, "mask_t": tri >= 0, "pos": tpos, "nrm": tnrm,
                    "w": np.where(mask, w, 0.0), "gt": material_maps(gt, material, size), "preds": {}})
    return out


@torch.no_grad()
def project_labels(labels, views, depths, pos, nrm, depth_tol=0.01):
    """(V, M) int16 per-texel mask id per view: >= 0 mask, -1 seen without mask, -2 not seen."""
    size = depths[0].shape[0]
    p = pos * EXTENT
    out = []
    for (cam, up), dmap, lab in zip(views, depths, labels):
        x, y, d = to_view(p, c2w_matrix(cam, up))
        to_cam = torch.tensor(cam, dtype=torch.float32, device=p.device)
        cos = nrm @ (to_cam / to_cam.norm())
        px = ((x / 0.5 + 1) / 2 * size).long().clamp(0, size - 1)
        py = ((1 - (y / 0.5 + 1) / 2) * size).long().clamp(0, size - 1)
        visible = (d <= dmap[py, px] + depth_tol) & (cos > MIN_COS)
        l = torch.from_numpy(lab.astype(np.int16)).cuda()[py, px]
        out.append(torch.where(visible, l, torch.full_like(l, -2)).cpu())
    return torch.stack(out).numpy()


# ------------------------------------------------------------------ parts
def segment(atl, Ls, min_part_frac=0.002, max_dist=0.01):
    """Part id per texel for every atlas ((H, W), -1 outside the UV islands); ids are shared across atlases.
    Ls: per atlas (V, H, W) per-view mask ids."""
    offsets, rows, cols, labeled_all, w_all, Lm_all = [0], [], [], [], [], []
    for a, L in zip(atl, Ls):
        mask = a["mask"]
        H, W = mask.shape
        n = int(mask.sum())
        idx = np.full((H, W), -1, np.int64)
        idx[mask] = np.arange(n) + offsets[-1]
        pos = a["pos"].cpu().numpy()
        labeled = (L >= 0).any(0) & mask
        for dy, dx in ((0, 1), (1, 0)):  # texel-grid neighbours
            sa = (slice(0, H - dy), slice(0, W - dx))
            sb = (slice(dy, H), slice(dx, W))
            both = labeled[sa] & labeled[sb]
            both &= np.linalg.norm(pos[sa] - pos[sb], axis=-1) < max_dist  # not across unrelated UV islands
            la, lb = L[:, sa[0], sa[1]], L[:, sb[0], sb[1]]
            seen = (la >= 0) & (lb >= 0)
            same = (seen & (la == lb)).sum(0)
            diff = (seen & (la != lb)).sum(0)
            join = both & (same > diff) & (same >= 1)
            rows.append(idx[sa][join]); cols.append(idx[sb][join])
        labeled_all.append(labeled[mask]); w_all.append(a["w"][mask]); Lm_all.append(L[:, mask])
        offsets.append(offsets[-1] + n)
    n = offsets[-1]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    _, comp = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    labeled, w, Lm = np.concatenate(labeled_all), np.concatenate(w_all), np.concatenate(Lm_all, 1)
    comp = np.where(labeled, comp, -1)
    V = Lm.shape[0]

    # merge components across seams / materials / disconnected pieces: vote on the majority mask per view
    ids, inv = np.unique(comp, return_inverse=True)
    area = np.bincount(inv, weights=w)
    keep = (ids >= 0) & (area >= min_part_frac * w.sum())
    big = ids[keep]
    order = np.argsort(comp, kind="stable")
    bounds = np.searchsorted(comp[order], big), np.searchsorted(comp[order], big, side="right")
    maj = np.full((len(big), V), -1)
    for i in range(len(big)):
        sel = order[bounds[0][i]:bounds[1][i]]
        for v in range(V):
            lv = Lm[v, sel]
            lv = lv[lv >= 0]
            if len(lv) >= 20 and len(lv) >= 0.2 * len(sel):
                vals, cnt = np.unique(lv, return_counts=True)
                maj[i, v] = vals[cnt.argmax()]
    both = (maj[:, None] >= 0) & (maj[None] >= 0)
    same = (both & (maj[:, None] == maj[None])).sum(-1)
    diff = (both & (maj[:, None] != maj[None])).sum(-1)
    adj = (same > diff) & (same >= 2)
    np.fill_diagonal(adj, False)
    _, group = connected_components(adj, directed=False)
    lut = np.full(len(ids), -1)
    lut[keep] = group
    part = lut[inv]

    # small and unseen pieces: nearest part in the same atlas
    grids = []
    for a, k in zip(atl, range(len(atl))):
        mask = a["mask"]
        grid = np.full(mask.shape, -1, np.int64)
        grid[mask] = part[offsets[k]:offsets[k + 1]]
        if (grid[mask] >= 0).any():
            _, (iy, ix) = ndimage.distance_transform_edt(grid < 0, return_indices=True)
            grid = np.where(mask, grid[iy, ix], -1)
        else:  # no labeled texel in this material at all: the material is one part
            grid[mask] = part.max(initial=-1) + 1 + k + len(atl)
        grids.append(grid)
    # compact ids
    allp = np.unique(np.concatenate([g[g >= 0] for g in grids]))
    for g in grids:
        g[g >= 0] = np.searchsorted(allp, g[g >= 0])
    return grids


def gt_parts(atl, levels=10):
    """Reference partition from the ground truth: connected regions of equal quantized roughness / metallic."""
    grids, nxt = [], 0
    for a in atl:
        mask, g = a["mask"], a["gt"]
        key = np.round(g["roughness"] * levels) * (levels + 1) + np.round(g["metallic"] * levels)
        out = np.full(mask.shape, -1, np.int64)
        for k in np.unique(key[mask]):
            lab, n = ndimage.label(mask & (key == k))
            out[lab > 0] = lab[lab > 0] - 1 + nxt
            nxt += n
        grids.append(out)
    return grids


# ------------------------------------------------------------------ metrics
def flat(atl, grids, get):
    """Concatenate a per-atlas map over the masked texels."""
    return np.concatenate([get(a, k)[a["mask"]] for k, a in enumerate(atl)])


def part_median(x, parts, w):
    """Replace every texel by its part's area-weighted median."""
    out = x.copy()
    order = np.lexsort((x, parts))
    xs, ps, ws = x[order], parts[order], w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
    ends = np.r_[starts[1:], len(ps)]
    for s, e in zip(starts, ends):
        c = np.cumsum(ws[s:e])
        med = xs[s + min(np.searchsorted(c, 0.5 * c[-1]), e - s - 1)] if c[-1] > 0 else xs[s:e].mean()
        out[order[s:e]] = med
    return out


def mae(a, b, w):
    return float((np.abs(a - b) * w).sum() / w.sum())


def colorize(parts, seed=0):
    rng = np.random.default_rng(seed)
    pal = rng.uniform(0.15, 1, (max(int(parts.max()), 0) + 2, 3))
    return to_u8(np.where(parts[..., None] >= 0, pal[parts.clip(min=0)], np.array([0.75, 0.85, 1.0])))


# ------------------------------------------------------------------ one asset
def evaluate(gt_path, input_path, glbs, supermat_pred=None, gen=None, cache=None, views=16, render_size=1024,
             supermat_views=16, out_dir=None):
    """Part metrics of one input. glbs: {label: path} predictions in the ground truth's geometry.
    Returns a report dict; SAM labels are cached in `cache` (.npz) if given."""
    gt = Glb(gt_path)
    atl = atlases(gt)
    for a in atl:
        a["preds"][CONSTANT] = {"roughness": np.full(a["mask"].shape, 0.5, np.float32),
                                "metallic": np.zeros(a["mask"].shape, np.float32)}
    for label, path in glbs.items():
        g = Glb(path)
        mapping = match_materials(gt, g)
        for a in atl:
            a["preds"][label] = material_maps(g, mapping[a["material"]], a["size"])
    if supermat_pred:
        from unbake.prior2d import CHANNELS as PC
        sviews = fibonacci_views(supermat_views)
        _, sdepths = render_views(_view_parts(input_path), 512, sviews)
        load = lambda n, k: torch.tensor(np.asarray(Image.open(os.path.join(supermat_pred, f"{k:04d}", f"{n}.png"))
                                                    .convert("L").resize((512, 512), Image.BILINEAR)),
                                         dtype=torch.float32, device="cuda")[..., None] / 255
        sp = {c: [load(c, k) for k in range(len(sviews))] for c in PC}
        for a in atl:
            val, _, nv = fuse(sviews, sdepths, sp, a["pos"][a["mask_t"]], a["nrm"][a["mask_t"]])
            m = {}
            for c in CHANNELS:
                grid = np.full(a["mask"].shape, np.nan, np.float32)
                grid[a["mask"]] = np.where(nv.cpu().numpy() > 0, val[c][:, 0].cpu().numpy(), np.nan)
                if np.isnan(grid[a["mask"]]).all():
                    grid[a["mask"]] = 0.5
                _, (iy, ix) = ndimage.distance_transform_edt(np.isnan(grid), return_indices=True)
                m[c] = grid[iy, ix]
            a["preds"]["SuperMat 16 views, fused (guidance target)"] = m
        del sdepths, sp

    vws = fibonacci_views(views)
    images, depths = render_views(_view_parts(input_path), render_size, vws)
    if cache and os.path.isfile(cache):
        labels = list(np.load(cache)["labels"])
    else:
        labels = sam_labels(gen, images)
        if cache:
            np.savez_compressed(cache, labels=np.stack(labels))
    Ls = []
    for a in atl:
        L = np.full((len(vws),) + a["mask"].shape, -2, np.int16)
        L[:, a["mask"]] = project_labels(labels, vws, depths, a["pos"][a["mask_t"]], a["nrm"][a["mask_t"]])
        Ls.append(L)
    seen = flat(atl, None, lambda a, k: (Ls[k] > -2).any(0))
    w = flat(atl, None, lambda a, k: a["w"])
    partitions = {"SAM": segment(atl, Ls), "Ground-truth regions": gt_parts(atl)}
    del Ls

    report = {"seen": float((seen * w).sum() / w.sum()), "partitions": {}, "methods": {}}
    pflat = {p: flat(atl, g, lambda a, k, g=g: g[k]) for p, g in partitions.items()}
    gtf = {c: flat(atl, None, lambda a, k, c=c: a["gt"][c]) for c in CHANNELS}
    for pname, parts in pflat.items():
        area = np.bincount(parts, weights=w) / w.sum()
        report["partitions"][pname] = {
            "num_parts": int(parts.max() + 1),
            "parts_for_90pct_area": int(np.searchsorted(np.cumsum(np.sort(area)[::-1]), 0.9) + 1),
            **{f"oracle_{c}": mae(part_median(gtf[c], parts, w), gtf[c], w) for c in CHANNELS}}
    for label in atl[0]["preds"]:
        r = {}
        for c in CHANNELS:
            x = flat(atl, None, lambda a, k: a["preds"][label][c])
            r[c] = {"texel": mae(x, gtf[c], w), **{p: mae(part_median(x, parts, w), gtf[c], w) for p, parts in pflat.items()}}
        report["methods"][label] = r

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tiles = [colorize(g) for g in partitions["SAM"]]
        Image.fromarray(np.concatenate([np.asarray(Image.fromarray(t).resize((256, 256), Image.NEAREST)) for t in tiles], 1)
                        ).save(os.path.join(out_dir, "parts_sam.png"))
        show = [0, 4, 8, 12]
        Image.fromarray(np.concatenate([
            np.concatenate([to_u8(images[v].cpu().numpy()[..., :3]) for v in show], 1),
            np.concatenate([colorize(labels[v].astype(np.int64), seed=v) for v in show], 1)], 0)
        ).resize((256 * len(show), 512)).save(os.path.join(out_dir, "sam_views.png"))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gt", required=True)
    ap.add_argument("--input", required=True, help="baked asset (rendered for SAM)")
    ap.add_argument("--glb", nargs="*", default=[], help="label=path of predictions in the ground truth's geometry")
    ap.add_argument("--supermat-pred", help="saved SuperMat single-view predictions (prior2d), fused into texels directly")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    glbs = dict(s.split("=", 1) for s in args.glb)
    report = evaluate(args.gt, args.input, glbs, args.supermat_pred, load_sam(),
                      os.path.join(args.out, "sam_labels.npz"), out_dir=args.out)
    json.dump(report, open(os.path.join(args.out, "part_metrics.json"), "w"), indent=2)
    for pname, s in report["partitions"].items():
        print(f"{pname:22s} parts {s['num_parts']:4d} (90% area: {s['parts_for_90pct_area']})  "
              f"oracle r {s['oracle_roughness']:.3f}  m {s['oracle_metallic']:.3f}")
    for c in CHANNELS:
        print(f"\n{c} MAE: texel | " + " | ".join(report["partitions"]))
        for label, r in report["methods"].items():
            print(f"  {label:44s} {r[c]['texel']:.3f} | " + " | ".join(f"{r[c][p]:.3f}" for p in report["partitions"]))


if __name__ == "__main__":
    main()
