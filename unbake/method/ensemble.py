"""Combine several predictions per input (e.g. seeds) into one, in the prediction layout of unbake/predictions.py.

    python -m unbake.method.ensemble --const-rm --output output/predictions/rebake_opt_ensemble \
        --members output/predictions/rebake_opt output/predictions/rebake_opt_s1 output/predictions/rebake_opt_s2

Base color (and roughness / metallic unless --const-rm) is the mean over the members' textures.
--const-rm sets roughness / metallic to the constants of the input-as-albedo reference (0.5 / 0).
All members keep the input's structure, so their materials correspond one to one.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from unbake.gltf_io import Glb
from unbake.predictions import INPUT_AS_ALBEDO_CONSTANTS, PREDICTION_FILE
from unbake.uv_bake import to_gltf_images
from unbake.method.common import material_maps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True, help="prediction folders of one method each")
    ap.add_argument("--output", required=True)
    ap.add_argument("--const-rm", action="store_true")
    args = ap.parse_args()
    members = [Path(d) for d in args.members]
    names = sorted(set.intersection(*[{p.parent.name for p in d.glob(f"*/{PREDICTION_FILE}")} for d in members]))
    done = 0
    for name in names:
        out = Path(args.output) / name
        if (out / "meta.json").exists():
            continue
        meta = json.loads((members[0] / name / "meta.json").read_text())
        glbs = [Glb(str(d / name / PREDICTION_FILE)) for d in members]
        glb = Glb(meta["input"])  # the baked input: write into its own materials
        for material in sorted({p.material for p in glb.world_primitives() if p.material is not None}):
            maps = [material_maps(g, material) for g in glbs]
            bc = np.mean([m[0] for m in maps], 0)
            h, w = bc.shape[:2]
            if args.const_rm:
                rough = np.full((h, w), INPUT_AS_ALBEDO_CONSTANTS["roughness"], np.float32)
                metal = np.full((h, w), INPUT_AS_ALBEDO_CONSTANTS["metallic"], np.float32)
            else:
                rough, metal = np.mean([m[1] for m in maps], 0), np.mean([m[2] for m in maps], 0)
            glb.set_pbr_textures(material, *to_gltf_images({"base_color": bc, "roughness": rough, "metallic": metal}))
        os.makedirs(out, exist_ok=True)
        glb.save(str(out / PREDICTION_FILE))
        (out / "meta.json").write_text(json.dumps({"input": meta["input"], "method": "ensemble", "const_rm": args.const_rm,
                                                   "members": [str(d / name) for d in members]}, indent=2))
        done += 1
    print(f"{done} new ensembles ({len(names)} inputs in all members) -> {args.output}")


if __name__ == "__main__":
    main()
