"""
Prediction layout and method registry, shared by the baselines (which write predictions) and the
evaluation (which reads them). Every method writes one folder per dataset input (an item's asset
baked under one lighting), with the input's geometry and UV layout:

    <predictions>/<method folder>/<item id>__<lighting name>/textured.glb   (+ cond.png, meta.json, ...)
"""
import os
from pathlib import Path

from .dataset import DatasetItem, Lighting

PREDICTIONS = "output/predictions"
PREDICTION_FILE = "textured.glb"

# Baseline: baked texture as base color, with fixed roughness/metallic. Files store no reliable
# values for these (an unlit export writes 0.9/0, a glTF without factors means 1/1 by spec).
INPUT_AS_ALBEDO = "input as albedo"
INPUT_AS_ALBEDO_CONSTANTS = {"roughness": 0.5, "metallic": 0.0}
# Prediction folder -> display name, in table order. Other folders under --predictions are
# evaluated too, under their folder name (e.g. extra seeds: trellis_unguided_512_s0).
METHOD_NAMES = {
    "trellis_unguided_512": "TRELLIS.2 (unguided, 512)",
    "trellis_unguided_1024": "TRELLIS.2 (unguided, 1024)",
    "supermat": "SuperMat",
    "material_anything": "Material Anything",
    "rgbx": "RGB<->X",
}


def input_name(item: DatasetItem, lighting: Lighting) -> str:
    return f"{item.id}__{lighting.name}"


def methods(predictions: str) -> dict:
    """name -> f(item, lighting) -> (predicted .glb path, constants overriding roughness/metallic or None)."""
    out = {INPUT_AS_ALBEDO: lambda item, lighting: (lighting.baked_path, INPUT_AS_ALBEDO_CONSTANTS)}
    folders = sorted(p.name for p in Path(predictions).iterdir() if p.is_dir()) if os.path.isdir(predictions) else []
    for folder in [f for f in METHOD_NAMES if f in folders] + [f for f in folders if f not in METHOD_NAMES]:
        out[METHOD_NAMES.get(folder, folder)] = (
            lambda item, lighting, folder=folder:
            (os.path.join(predictions, folder, input_name(item, lighting), PREDICTION_FILE), None))
    return out
