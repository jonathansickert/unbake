"""Is there any roughness / metallic signal in the baked textures?

For the ground-truth base color (kd = base color, as the bakes use it) the Lambertian SH model with visibility
is fitted; if the bakes contained specular light, the residual c - kd*E would depend on roughness and
metallic. Reported per input: residual MAE, Pearson correlation of the (luminance) residual with roughness and
metallic, the fraction of residual variance a small regression on [rough, metal, rough*metal, their squares]
explains, and the mean residual for metallic / glossy texels vs. the rest.

    python -m unbake.experiments.specular_signal data/unbake_sub10 out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from unbake.method.rebake import Asset, srgb_to_linear, linear_to_srgb, sample_maps, load_gt_maps, fit_lighting

torch.set_grad_enabled(False)
rows = {}
for data in sys.argv[1:-1]:
    for baked in sorted(Path(data).glob("*/*_rot*.glb")):
        asset = Asset(baked)
        tx = asset.texels(60000)
        T = asset.transfer(tx["pos"], tx["nrm"], 128)
        bc, ro, me = sample_maps(load_gt_maps(asset, baked.parent / "pbr.glb"), tx)
        kd = srgb_to_linear(bc.clamp(0, 1))
        c = tx["baked"]
        L = fit_lighting(kd, srgb_to_linear(c), T)
        pred = linear_to_srgb(kd * (T @ L))
        res = (c - pred).mean(1).cpu().numpy()  # signed luminance residual in sRGB
        r, m = ro.cpu().numpy(), me.cpu().numpy()
        X = np.stack([np.ones_like(r), r, m, r * m, r ** 2, m ** 2], 1)
        coef, *_ = np.linalg.lstsq(X, res, rcond=None)
        r2 = 1 - np.var(res - X @ coef) / max(np.var(res), 1e-12)
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
        metal, gloss = m > 0.5, (r < 0.3) & (m <= 0.5)
        row = {"mae": float(np.abs(res).mean()), "corr_rough": corr(res, r), "corr_metal": corr(res, m),
               "corr_absres_rough": corr(np.abs(res), r), "r2_material": float(r2),
               "frac_metal": float(metal.mean()), "frac_gloss": float(gloss.mean()),
               "res_metal": float(res[metal].mean()) if metal.any() else float("nan"),
               "res_gloss": float(res[gloss].mean()) if gloss.any() else float("nan"),
               "res_rest": float(res[~metal & ~gloss].mean()) if (~metal & ~gloss).any() else float("nan"),
               "rough_std": float(r.std()), "metal_std": float(m.std())}
        name = f"{baked.parent.name}__{baked.stem}"
        rows[name] = row
        print(f"{name[:8]} {baked.stem[:20]:20s} MAE {row['mae']:.3f} | corr rough {row['corr_rough']:+.2f} metal "
              f"{row['corr_metal']:+.2f} | R2 {row['r2_material']:.3f} | metal {row['frac_metal']:.2f} "
              f"(res {row['res_metal']:+.3f}) gloss {row['frac_gloss']:.2f} (res {row['res_gloss']:+.3f}) "
              f"rest res {row['res_rest']:+.3f}", flush=True)
        torch.cuda.empty_cache()
keys = ["mae", "corr_rough", "corr_metal", "corr_absres_rough", "r2_material"]
mean = {k: float(np.nanmean([r[k] for r in rows.values()])) for k in keys}
print("\nmean:", {k: round(v, 3) for k, v in mean.items()})
Path(sys.argv[-1]).write_text(json.dumps({"mean": mean, "assets": rows}, indent=2))
