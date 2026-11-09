"""
Material accuracy in UV space: surface-area-weighted MAE / PSNR of base color, roughness and
metallic against the ground-truth PBR asset (unbake.evaluation.uv_metrics). Every method is
scored once per dataset input (item x lighting). Predictions must have the ground truth's
geometry and UV layout (checked; mesh names, order and material numbering may differ).

    python -m unbake.evaluation.evaluate_uv                     # every method in output/predictions
    python -m unbake.evaluation.evaluate_uv --max-items 2

Methods: the input-as-albedo baseline (the dataset's baked assets) and every folder in
--predictions (see unbake/predictions.py); missing predictions are skipped and reported.
Writes <output>/metrics_uv.json (summary + one record per item/lighting/method).
"""
import argparse
import json
import os
import time

from unbake.evaluation.metrics import uv_metrics
from unbake.evaluation.summary import add_eval_args, evaluation_inputs, report_all
from unbake.gltf_io import Glb
from unbake.predictions import INPUT_AS_ALBEDO_CONSTANTS

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
add_eval_args(parser)
args = parser.parse_args()
dataset, items, methods = evaluation_inputs(args)
print(f"{len(items)} items, methods: {', '.join(methods)}")

records, missing = [], {name: [] for name in methods}
start = time.time()
for n, index in enumerate(items):
    item = dataset[index]
    gt = item.ground_truth()
    print(f"[{n + 1}/{len(items)}] {item.id[:12]}  ({len(item.lightings)} lighting conditions, {time.time() - start:.0f}s elapsed)")
    for lighting in item.lightings:
        for name, locate in methods.items():
            path, constants = locate(item, lighting)
            if not os.path.isfile(path):
                missing[name].append(path)
                continue
            records.append({"item": item.id, "lighting": lighting.name, "method": name, "path": path,
                            "uv": uv_metrics(gt, Glb(path), constants)})

summary, summary_common = report_all(records, list(methods), missing)
os.makedirs(args.output, exist_ok=True)
with open(os.path.join(args.output, "metrics_uv.json"), "w") as f:
    json.dump({"data": args.data, "items": [dataset.ids[i] for i in items], "methods": list(methods),
               "input_as_albedo_constants": INPUT_AS_ALBEDO_CONSTANTS,
               "summary": summary, "summary_common_pairs": summary_common,
               "missing_predictions": {n: len(p) for n, p in missing.items()}, "records": records}, f, indent=2)
print(f"\nSaved {args.output}/metrics_uv.json")
