"""
Brightness calibration of predictions, fitted on other objects (leave one object out).

The bakes are exposure-normalized, so the conditioning image does not show how bright an object is, and the
rebake model cannot tell either (the lighting is refit). TRELLIS.2 then regresses towards a typical brightness:
dark objects come out too bright and bright ones too dark (evaluation/diagnostics.py). Per input and channel,
with m the area-weighted mean base color (sRGB), a line log m_gt = a + b log m_pred is fitted on the inputs of
all other objects, and the held-out prediction is scaled by exp(a + b log m_pred) / m_pred. --fixed-slope 1
fits only the offset a (one global scale).

    python -m unbake.calibrate --data data --prior output/predictions/guided_delight --output output/predictions/guided_delight_cal
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from unbake.evaluation.metrics import match_materials, material_maps
from unbake.gltf_io import Glb
from unbake.predictions import PREDICTION_FILE
from unbake.uv_bake import to_gltf_images, uv_surface_weights


def channel_means(gt: Glb, pred: Glb):
    """Area-weighted mean base color (3,) of the ground truth and of the prediction."""
    sg, sp, sw = 0.0, 0.0, 0.0
    prims = gt.world_primitives()
    for material, pred_material in match_materials(gt, pred).items():
        img = gt.material_base_color_image(material)
        size = gt.image(img).size if img is not None else (1024, 1024)
        w = uv_surface_weights([p for p in prims if p.material == material], size)[..., None]
        sg = sg + (w * material_maps(gt, material, size)["base_color"]).sum((0, 1))
        sp = sp + (w * material_maps(pred, pred_material, size)["base_color"]).sum((0, 1))
        sw += w.sum()
    return sg / sw, sp / sw


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prior", required=True, help="prediction folder to calibrate")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-slope", type=float, default=None, help="fix b (1 = one global scale)")
    args = parser.parse_args()

    cache = Path(args.output) / "means.json"
    means = json.loads(cache.read_text()) if cache.exists() else {}
    metas = {p.parent.name: json.loads(p.read_text()) for p in sorted(Path(args.prior).glob("*/meta.json"))}
    for name, meta in metas.items():
        if name not in means:
            gt = Glb(os.path.join(os.path.dirname(meta["input"]), "pbr.glb"))
            g, p = channel_means(gt, Glb(str(Path(args.prior) / name / PREDICTION_FILE)))
            means[name] = {"gt": g.tolist(), "pred": p.tolist()}
    os.makedirs(args.output, exist_ok=True)
    cache.write_text(json.dumps(means))

    obj = lambda name: name.split("__")[0]
    fits = {}
    for name, meta in metas.items():
        out = Path(args.output) / name
        train = [n for n in means if obj(n) != obj(name)]
        x = np.log(np.clip([means[n]["pred"] for n in train], 1e-4, None)).ravel()
        y = np.log(np.clip([means[n]["gt"] for n in train], 1e-4, None)).ravel()
        if args.fixed_slope is None:
            b, a = np.polyfit(x, y, 1)
        else:
            b = args.fixed_slope
            a = float(np.mean(y - b * x))
        fits[obj(name)] = (float(a), float(b))
        if (out / "meta.json").exists():
            continue
        m = np.clip(means[name]["pred"], 1e-4, None)
        scale = np.exp(a + b * np.log(m)) / m
        glb = Glb(str(Path(args.prior) / name / PREDICTION_FILE))
        for material in sorted({p.material for p in glb.world_primitives() if p.material is not None}):
            img = glb.material_base_color_image(material)
            size = glb.image(img).size if img is not None else (1024, 1024)
            maps = material_maps(glb, material, size)
            maps["base_color"] = np.clip(maps["base_color"] * scale, 0, 1)
            glb.set_pbr_textures(material, *to_gltf_images(maps))
        os.makedirs(out, exist_ok=True)
        glb.save(str(out / PREDICTION_FILE))
        (out / "meta.json").write_text(json.dumps({"input": meta["input"], "method": "calibrate",
                                                   "prior": str(Path(args.prior) / name), "a": a, "b": b,
                                                   "scale": scale.tolist()}, indent=2))
    for o, (a, b) in sorted(fits.items()):
        print(f"{o[:12]}: log m_gt = {a:+.3f} + {b:.3f} log m_pred")


if __name__ == "__main__":
    main()
