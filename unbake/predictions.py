"""
Prediction layout and method registry, shared by the baselines (which write predictions) and the
evaluation (which reads them). Every method writes one folder per dataset input (an item's asset
baked under one lighting), with the input's geometry and UV layout:

    <predictions>/<method folder>/<item id>__<lighting name>/textured.glb   (+ cond.png, meta.json, ...)
"""
import os
from pathlib import Path

from .dataset import DatasetItem, Lighting
from .paths import OUTPUT

PREDICTIONS = os.path.join(OUTPUT, "predictions")
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
    "guided_supermat_sv_b1": "UnBake basic (rebake + SuperMat 16v, beta 1)",
    "guided_supermat_sv_b1_1024": "UnBake basic 1024 (rebake + SuperMat 16v, beta 1)",
    "guided_supermat_sv_b1_1024_vis_ss4": "UnBake 1024 (rebake + visibility + SuperMat 16v, beta 1)",
    "guided_supermat_sv_b1_1024_vis_ss4_detail": "UnBake 1024 + detail recovery",
    "trellis1024_bc+supermat_rm": "Hybrid (TRELLIS.2 base color + SuperMat r/m)",
    "hybrid_detail": "Hybrid + detail recovery",
    "guided+sm16_rm": "UnBake base color + fused SuperMat 16v r/m",
    "guided_detail+sm16_rm": "UnBake base color + detail + fused SuperMat 16v r/m",
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


def first_complete_lighting(item: DatasetItem, methods: dict) -> Lighting:
    """The item's first lighting condition for which every method has a prediction (methods: as
    returned by methods()). With complete methods this is simply the first lighting condition, so
    figures don't pick examples in anyone's favour."""
    for lighting in item.lightings:
        if all(os.path.isfile(locate(item, lighting)[0]) for locate in methods.values()):
            return lighting
    raise ValueError(f"item {item.id}: no lighting condition has predictions of every method")
