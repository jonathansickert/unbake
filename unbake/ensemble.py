"""
Ensemble of predictions: the mean of several predictions of the same inputs (e.g. TRELLIS.2 or guided samples
with different seeds), written in the prediction layout of unbake/predictions.py.

The error of a single sample is largely variance between samples, so the mean base color of a few seeds is
closer to the ground truth than any single seed. --const-rm replaces roughness and metallic by the constants of
the input-as-albedo baseline (0.5 / 0), which no method beats on this dataset.

    python -m unbake.ensemble --const-rm --output output/predictions/guided_ensemble \
        --members output/predictions/guided_s42 output/predictions/guided_s1 output/predictions/guided_s2
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from unbake.evaluation.metrics import material_maps
from unbake.gltf_io import Glb
from unbake.predictions import INPUT_AS_ALBEDO_CONSTANTS, PREDICTION_FILE
from unbake.uv_bake import to_gltf_images


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--members", nargs="+", required=True, help="prediction folders, one per member")
    parser.add_argument("--output", required=True)
    parser.add_argument("--const-rm", action="store_true", help="roughness 0.5 and metallic 0 instead of the mean")
    args = parser.parse_args()

    members = [Path(d) for d in args.members]
    names = sorted(set.intersection(*[{p.parent.name for p in d.glob(f"*/{PREDICTION_FILE}")} for d in members]))
    done = 0
    for name in names:
        out = Path(args.output) / name
        if (out / "meta.json").exists():
            continue
        meta = json.loads((members[0] / name / "meta.json").read_text())
        preds = [Glb(str(d / name / PREDICTION_FILE)) for d in members]
        glb = Glb(meta["input"])  # write into the baked input, so geometry and UV layout stay the input's
        for material in sorted({p.material for p in glb.world_primitives() if p.material is not None}):
            img = glb.material_base_color_image(material)
            size = glb.image(img).size if img is not None else (1024, 1024)
            maps = [material_maps(g, material, size) for g in preds]
            mean = {k: np.mean([m[k] for m in maps], 0) for k in ("base_color", "roughness", "metallic")}
            if args.const_rm:
                for k in ("roughness", "metallic"):
                    mean[k] = np.full_like(mean[k], INPUT_AS_ALBEDO_CONSTANTS[k])
            glb.set_pbr_textures(material, *to_gltf_images(mean))
        os.makedirs(out, exist_ok=True)
        glb.save(str(out / PREDICTION_FILE))
        (out / "meta.json").write_text(json.dumps({"input": meta["input"], "method": "ensemble",
                                                   "const_rm": args.const_rm,
                                                   "members": [str(d / name) for d in members]}, indent=2))
        done += 1
    print(f"{done} new ensembles ({len(names)} inputs in all members) -> {args.output}")


if __name__ == "__main__":
    main()
