"""Rebake model of the proposal: Lambertian shading under second-order SH lighting with per-point
visibility (precomputed radiance transfer), fitted to the observed baked colors by least squares.

    c_i = kd_i * (T_i @ L),   kd = base_color_linear (the dataset bakes use the full base color),
    T_ik = E_{w ~ cos}[ V_i(w) * Y_k(w) ]   (cosine-weighted rays, V = 1 if the ray escapes)

All computations in TRELLIS space (unit cube, Z up) so that texel positions map to voxels.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from PIL import Image
from cumesh.bvh import cuBVH

from unbake.baselines.common import load_parts, normalization, normalize, uv_coverage


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).clamp(min=0) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-12) ** (1 / 2.4) - 0.055)


def sh_basis(d):
    """Real SH up to l = 2, d: [..., 3] unit vectors -> [..., 9]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    return torch.stack([
        0.282095 * torch.ones_like(x),
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
        1.092548 * x * z, 0.546274 * (x * x - y * y)], -1)


class Asset:
    """Geometry of one baked input in TRELLIS space, its UV atlases and a BVH for visibility."""

    def __init__(self, baked_path):
        self.path = Path(baked_path)
        self.parts = load_parts(baked_path)
        self.center, self.scale = normalization(np.concatenate([p["mesh"].vertices for p in self.parts]))
        rot = lambda n: np.stack([n[:, 0], -n[:, 2], n[:, 1]], 1).astype(np.float32)
        self.v = [normalize(np.asarray(p["mesh"].vertices), self.center, self.scale) for p in self.parts]
        self.n = [rot(np.asarray(p["mesh"].vertex_normals)) for p in self.parts]
        self.f = [np.asarray(p["mesh"].faces, np.int64) for p in self.parts]
        offs = np.cumsum([0] + [len(v) for v in self.v])
        self.bvh = cuBVH(np.concatenate(self.v), np.concatenate([f + o for f, o in zip(self.f, offs)]))
        self.atlases = {}
        for k, p in enumerate(self.parts):
            self.atlases.setdefault(p["tex_key"], []).append(k)

    @torch.no_grad()
    def texels(self, max_points=60000, seed=0):
        """Random covered texels: dict with atlas index, pixel (y, x), position, normal, baked color (sRGB)."""
        out = {"atlas": [], "yx": [], "pos": [], "nrm": [], "baked": []}
        ctx = dr.RasterizeCudaContext()
        for a, (key, idx) in enumerate(self.atlases.items()):
            tex = torch.tensor(self.parts[idx[0]]["texture"], device="cuda")
            h, w = tex.shape[:2]
            for k in idx:
                f = torch.tensor(self.f[k], device="cuda")
                uv = torch.tensor(self.parts[k]["uv"], device="cuda")[f]
                uv = (uv - torch.floor(uv.mean(1, keepdim=True))).reshape(-1, 2)
                tri = torch.arange(len(uv), dtype=torch.int32, device="cuda").reshape(-1, 3)
                clip = torch.cat([uv[:, :1] * 2 - 1, (1 - uv[:, 1:]) * 2 - 1, torch.zeros_like(uv[:, :1]),
                                  torch.ones_like(uv[:, :1])], -1)[None]
                rast, _ = dr.rasterize(ctx, clip, tri, (h, w))
                m = rast[0, ..., 3] > 0
                pos = dr.interpolate(torch.tensor(self.v[k], device="cuda")[f].reshape(-1, 3)[None], rast, tri)[0][0]
                nrm = dr.interpolate(torch.tensor(self.n[k], device="cuda")[f].reshape(-1, 3)[None], rast, tri)[0][0]
                yx = m.nonzero()
                out["atlas"].append(torch.full((len(yx),), a, device="cuda"))
                out["yx"].append(yx)
                out["pos"].append(pos[m])
                out["nrm"].append(F.normalize(nrm[m], dim=-1))
                out["baked"].append(tex[m])
        out = {k: torch.cat(v) for k, v in out.items()}
        g = torch.Generator(device="cuda").manual_seed(seed)
        if len(out["pos"]) > max_points:
            sel = torch.randperm(len(out["pos"]), device="cuda", generator=g)[:max_points]
            out = {k: v[sel] for k, v in out.items()}
        return out

    @torch.no_grad()
    def transfer(self, pos, nrm, n_rays=128, visibility=True, seed=0):
        """[N, 9] SH transfer coefficients with (or without) self-occlusion."""
        g = torch.Generator(device="cuda").manual_seed(seed)
        t = torch.where(nrm[:, :1].abs() < 0.9, torch.tensor([[1.0, 0, 0]], device="cuda"),
                        torch.tensor([[0, 1.0, 0]], device="cuda"))
        tan = F.normalize(torch.cross(t, nrm, dim=-1), dim=-1)
        bit = torch.cross(nrm, tan, dim=-1)
        out = torch.zeros(len(pos), 9, device="cuda")
        chunk = max(1, (1 << 22) // n_rays)
        for s in range(0, len(pos), chunk):
            p, n, ta, bi = pos[s:s + chunk], nrm[s:s + chunk], tan[s:s + chunk], bit[s:s + chunk]
            u1 = torch.rand(len(p), n_rays, device="cuda", generator=g)
            u2 = torch.rand(len(p), n_rays, device="cuda", generator=g)
            r, phi = torch.sqrt(u1), 2 * np.pi * u2
            loc = torch.stack([r * torch.cos(phi), r * torch.sin(phi), torch.sqrt(1 - u1)], -1)
            d = loc[..., :1] * ta[:, None] + loc[..., 1:2] * bi[:, None] + loc[..., 2:] * n[:, None]
            y = sh_basis(d)
            if visibility:
                o = (p + n * 1e-4)[:, None].expand_as(d)
                _, face, _ = self.bvh.ray_trace(o.reshape(-1, 3), d.reshape(-1, 3))
                y = y * (face < 0).reshape(len(p), n_rays, 1)
            out[s:s + chunk] = y.mean(1)
        return out


def fit_lighting(kd_lin, c_lin, T, ridge=1e-4):
    """Least-squares SH lighting [K, 3] (per color channel) for c = kd * (T @ L), K = number of SH coefficients."""
    L = []
    for ch in range(3):
        A = kd_lin[:, ch:ch + 1] * T
        AtA = A.T @ A + ridge * torch.eye(T.shape[1], device=A.device) * A.pow(2).sum() / len(A)
        L.append(torch.linalg.solve(AtA, A.T @ c_lin[:, ch]))
    return torch.stack(L, 1)


def rebake(kd_lin, T, L):
    return kd_lin * (T @ L)


def rebake_error(kd_lin, c_srgb, T, L=None):
    """Fit lighting (unless given), rebake and compare with the observed colors in sRGB."""
    c_lin = srgb_to_linear(c_srgb)
    if L is None:
        L = fit_lighting(kd_lin, c_lin, T)
    pred = linear_to_srgb(rebake(kd_lin, T, L))
    d = pred - c_srgb
    mse = float((d ** 2).mean())
    return {"mae": float(d.abs().mean()), "psnr": float(10 * np.log10(1 / max(mse, 1e-10)))}, L


def sample_maps(folder_maps, texels):
    """Gather candidate maps at the texels. folder_maps: {atlas: (bc [H,W,3], rough [H,W], metal [H,W])} in [0, 1]."""
    bc = torch.zeros(len(texels["atlas"]), 3, device="cuda")
    ro = torch.zeros(len(texels["atlas"]), device="cuda")
    me = torch.zeros(len(texels["atlas"]), device="cuda")
    for a, (b, r, m) in folder_maps.items():
        sel = texels["atlas"] == a
        y, x = texels["yx"][sel, 0], texels["yx"][sel, 1]
        bc[sel] = torch.as_tensor(b, device="cuda")[y, x]
        ro[sel] = torch.as_tensor(r, device="cuda")[y, x]
        me[sel] = torch.as_tensor(m, device="cuda")[y, x]
    return bc, ro, me


def pbr_maps(path):
    """{frozenset(part names): {"basecolor", "roughness", "metallic", "mask"}} per texture atlas of a PBR .glb
    (ground truth or prediction), in [0, 1], base color in texture (sRGB) space."""
    out = {}
    groups = {}
    for p in load_parts(path):
        groups.setdefault(p["tex_key"], []).append(p)
    for group in groups.values():
        mat = group[0]["mesh"].visual.material
        bc = group[0]["texture"]  # base color texture x factor
        h, w = bc.shape[:2]
        mr = getattr(mat, "metallicRoughnessTexture", None)
        rf = getattr(mat, "roughnessFactor", None)
        mf = getattr(mat, "metallicFactor", None)
        rf = 1.0 if rf is None else float(rf)  # glTF defaults
        mf = 1.0 if mf is None else float(mf)
        if mr is not None:
            mr = np.asarray(mr.convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255
            rough, metal = mr[..., 1] * rf, mr[..., 2] * mf
        else:
            rough, metal = np.full((h, w), rf, np.float32), np.full((h, w), mf, np.float32)
        mask = np.zeros((h, w), bool)
        for p in group:
            mask |= uv_coverage(p["uv"], np.asarray(p["mesh"].faces), h, w)
        out[frozenset(p["name"] for p in group)] = {"basecolor": bc, "roughness": rough, "metallic": metal, "mask": mask}
    return out


def match_parts(src, dst):
    """Map part names of src to dst: same face count, closest face-centroid mean and extent
    (robust to different vertex splitting on export); the name is the fallback."""
    def sig(p):
        m = p["mesh"]
        return len(m.faces), np.concatenate([m.triangles_center.mean(0), np.ptp(m.vertices, 0)])
    dst_sig = [(p["name"], *sig(p)) for p in dst]
    scale = max(float(np.ptp(np.concatenate([np.asarray(p["mesh"].vertices) for p in dst]), 0).max()), 1e-8)
    out = {}
    for p in src:
        n_f, s = sig(p)
        cands = [(np.abs(s - ds).max(), name) for name, dn, ds in dst_sig if dn == n_f]
        best = min(cands, default=(np.inf, None))
        out[p["name"]] = best[1] if best[0] < 1e-2 * scale else p["name"]
    return out


def load_glb_maps(asset, path):
    """Maps of a PBR .glb with the asset's geometry, resized to the baked atlas sizes, keyed by the baked
    atlas index (parts matched by geometry)."""
    maps = pbr_maps(path)
    names = match_parts(asset.parts, load_parts(path))
    out = {}
    for a, (key, idx) in enumerate(asset.atlases.items()):
        h, w = asset.parts[idx[0]]["texture"].shape[:2]
        g = maps[frozenset(names[asset.parts[k]["name"]] for k in idx)]
        rs = lambda x: np.asarray(Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                                  np.float32) / 255
        out[a] = (rs(g["basecolor"]), rs(g["roughness"]), rs(g["metallic"]))
    return out


def load_gt_maps(asset, gt_path):
    """Ground-truth maps keyed by the baked atlas index."""
    return load_glb_maps(asset, gt_path)


def load_result_maps(result_dir, asset):
    """Maps of a prediction folder (textured.glb, see unbake/predictions.py) keyed by the baked atlas index."""
    return load_glb_maps(asset, str(Path(result_dir) / "textured.glb"))


def sh_basis_order(d, lmax):
    """Real SH up to degree lmax via associated Legendre recursion, d: [..., 3] -> [..., (lmax+1)^2]."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    theta_c = z.clamp(-1, 1)
    phi = torch.atan2(y, x)
    s = torch.sqrt((1 - theta_c ** 2).clamp(min=0))
    P = {(0, 0): torch.ones_like(z)}
    for m in range(1, lmax + 1):
        P[(m, m)] = -(2 * m - 1) * s * P[(m - 1, m - 1)]
    for m in range(0, lmax):
        P[(m + 1, m)] = (2 * m + 1) * theta_c * P[(m, m)]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            P[(l, m)] = ((2 * l - 1) * theta_c * P[(l - 1, m)] - (l + m - 1) * P[(l - 2, m)]) / (l - m)
    from math import factorial, pi, sqrt
    out = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            k = sqrt((2 * l + 1) / (4 * pi) * factorial(l - am) / factorial(l + am))
            if m == 0:
                out.append(k * P[(l, 0)])
            elif m > 0:
                out.append(sqrt(2) * k * torch.cos(m * phi) * P[(l, m)])
            else:
                out.append(sqrt(2) * k * torch.sin(am * phi) * P[(l, am)])
    return torch.stack(out, -1)


@torch.no_grad()
def cosine_rays(asset, pos, nrm, n_rays, seed=0):
    """Cosine-weighted directions [N, R, 3] and visibility [N, R] per point."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t = torch.where(nrm[:, :1].abs() < 0.9, torch.tensor([[1.0, 0, 0]], device="cuda"), torch.tensor([[0, 1.0, 0]], device="cuda"))
    tan = F.normalize(torch.cross(t, nrm, dim=-1), dim=-1)
    bit = torch.cross(nrm, tan, dim=-1)
    u1 = torch.rand(len(pos), n_rays, device="cuda", generator=g)
    u2 = torch.rand(len(pos), n_rays, device="cuda", generator=g)
    r, phi = torch.sqrt(u1), 2 * np.pi * u2
    loc = torch.stack([r * torch.cos(phi), r * torch.sin(phi), torch.sqrt(1 - u1)], -1)
    d = loc[..., :1] * tan[:, None] + loc[..., 1:2] * bit[:, None] + loc[..., 2:] * nrm[:, None]
    vis = torch.empty(len(pos), n_rays, dtype=torch.bool, device="cuda")
    chunk = max(1, (1 << 22) // n_rays)
    for s in range(0, len(pos), chunk):
        o = (pos[s:s + chunk] + nrm[s:s + chunk] * 1e-4)[:, None].expand_as(d[s:s + chunk])
        _, face, _ = asset.bvh.ray_trace(o.reshape(-1, 3), d[s:s + chunk].reshape(-1, 3))
        vis[s:s + chunk] = (face < 0).reshape(-1, n_rays)
    return d, vis
