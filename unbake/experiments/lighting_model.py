"""How much of the rebake residual comes from the lighting model? For the ground-truth materials:
SH lighting of degree 2 / 4 / 6 (with visibility) vs. the true HDR (rotation and exposure searched,
global scale fitted), i.e. an upper bound for any lighting model under Lambertian shading."""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json, sys
from pathlib import Path
import cv2, numpy as np, torch
from unbake.method.rebake import (Asset, srgb_to_linear, linear_to_srgb, sample_maps, load_gt_maps, sh_basis_order,
                        cosine_rays)

torch.set_grad_enabled(False)
envdir = Path(__file__).resolve().parents[1] / "env_maps"
data = Path(sys.argv[1])
rows = {}
for baked in sorted(data.glob("*/*_rot*.glb")):
    asset = Asset(baked)
    tx = asset.texels(30000)
    gt_bc, _, gt_me = sample_maps(load_gt_maps(asset, baked.parent / "pbr.glb"), tx)
    kd = srgb_to_linear(gt_bc.clamp(0, 1)) * (1 - gt_me[:, None])
    c = tx["baked"]; c_lin = srgb_to_linear(c)
    d, vis = cosine_rays(asset, tx["pos"], tx["nrm"], 256)
    r = {}
    for lmax in [2, 4, 6]:
        T = (sh_basis_order(d, lmax) * vis[..., None]).mean(1)
        L = []
        for ch in range(3):
            A = kd[:, ch:ch + 1] * T
            L.append(torch.linalg.lstsq(A, c_lin[:, ch:ch + 1]).solution[:, 0])
        pred = linear_to_srgb(kd * (T @ torch.stack(L, 1)))
        r[f"sh{lmax}"] = float((pred - c).abs().mean())
    # true HDR, search rotation about the up axis (both directions), fit a scale per channel
    meta = [e for e in json.loads((baked.parent / "meta.json").read_text()) if e["name"] == baked.stem][0]
    env = cv2.imread(str(envdir / meta["env_map"]), cv2.IMREAD_UNCHANGED)[..., :3][..., ::-1].copy()
    env = torch.tensor(cv2.resize(env, (512, 256), interpolation=cv2.INTER_AREA), device="cuda")
    best = (1e9, None)
    for sign in [1, -1]:
        for up in ["z", "y"]:
            for rot_deg in np.arange(0, 360, 5):
                # TRELLIS space is Z up; lat-long with the pole along "up"
                if up == "z":
                    theta = torch.acos(d[..., 2].clamp(-1, 1)); az = torch.atan2(d[..., 1], d[..., 0])
                else:
                    theta = torch.acos(d[..., 1].clamp(-1, 1)); az = torch.atan2(d[..., 0], d[..., 2])
                az = (sign * az + np.radians(rot_deg)) % (2 * np.pi)
                iy = (theta / np.pi * 255).long().clamp(0, 255); ix = (az / (2 * np.pi) * 511).long().clamp(0, 511)
                E = (env[iy, ix] * vis[..., None]).mean(1)
                s = ((kd * E) * c_lin).sum(0) / ((kd * E) ** 2).sum(0).clamp(min=1e-12)
                err = float((linear_to_srgb(kd * E * s) - c).abs().mean())
                if err < best[0]:
                    best = (err, (sign, up, int(rot_deg)))
    r["true_hdr"] = best[0]; r["true_hdr_setting"] = best[1]; r["meta_rotation"] = round(meta["rotation_deg"], 1)
    rows[f"{baked.parent.name}__{baked.stem}"] = r
    print(f"{baked.parent.name[:8]} {baked.stem[:22]:22s} SH2 {r['sh2']:.4f} SH4 {r['sh4']:.4f} SH6 {r['sh6']:.4f} "
          f"true HDR {r['true_hdr']:.4f} {r['true_hdr_setting']} (meta rot {r['meta_rotation']})", flush=True)
    torch.cuda.empty_cache()
m = {k: float(np.mean([r[k] for r in rows.values()])) for k in ["sh2", "sh4", "sh6", "true_hdr"]}
print("\nmean GT rebake MAE (sRGB):", {k: round(v, 4) for k, v in m.items()})
Path(sys.argv[2]).write_text(json.dumps({"mean": m, "assets": rows}, indent=2))
