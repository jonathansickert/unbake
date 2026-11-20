"""
Part-level roughness / metallic metrics (unbake.evaluation.part_metrics) on every dataset input: SAM is run on
each baked input (it is all the method sees), and every method is scored per texel and part-constant.

    python -m unbake.evaluation.part_metrics_dataset --out output/parts/dataset

Resumable: per-input reports and SAM labels are kept in <out>/<item>__<lighting>/.
Writes <out>/part_metrics.json (records + means) and <out>/part_metrics.md.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from unbake.dataset import UnbakeDataset
from unbake.evaluation.part_metrics import CHANNELS, evaluate, load_sam
from unbake.paths import DATA, OUTPUT
from unbake.predictions import PREDICTION_FILE, input_name

METHODS = {  # label -> prediction folder
    "TRELLIS.2 unguided 1024": os.path.join(OUTPUT, "predictions", "trellis_unguided_1024"),
    # the SuperMat folder is gone; the hybrid stores SuperMat's roughness / metallic maps unchanged
    "SuperMat / hybrid": os.path.join(OUTPUT, "hybrid_test", "predictions", "trellis1024_bc+supermat_rm"),
    "Guided (exp. 13)": os.path.join(OUTPUT, "predictions", "guided_supermat_sv_b1_1024_vis_ss4"),
}
SUPERMAT_PRED = os.path.join(OUTPUT, "predictions", "guided_supermat_sv_b1_1024")  # per-view SuperMat (exp. 10)


def summarize(records):
    methods = list(records[0]["methods"])
    parts = list(records[0]["partitions"])
    mean = lambda f: float(np.mean([f(r) for r in records]))
    out = {"n": len(records), "seen": mean(lambda r: r["seen"]),
           "partitions": {p: {k: mean(lambda r: r["partitions"][p][k]) for k in records[0]["partitions"][p]} for p in parts},
           "methods": {m: {c: {k: mean(lambda r: r["methods"][m][c][k]) for k in records[0]["methods"][m][c]}
                           for c in CHANNELS} for m in methods}}
    items = sorted({r["item"] for r in records})
    out["per_item"] = {i: {"partitions": {p: {k: float(np.mean([r["partitions"][p][k] for r in records if r["item"] == i]))
                                              for k in records[0]["partitions"][p]} for p in parts},
                           "methods": {m: {c: {k: float(np.mean([r["methods"][m][c][k] for r in records if r["item"] == i]))
                                               for k in records[0]["methods"][m][c]} for c in CHANNELS} for m in methods}}
                       for i in items}
    return out


def markdown(s):
    parts = list(s["partitions"])
    lines = [f"{s['n']} inputs; texels seen by SAM views: {100 * s['seen']:.1f}%", "",
             "| partition | parts | parts for 90% area | oracle roughness | oracle metallic |", "|---|---|---|---|---|"]
    for p, v in s["partitions"].items():
        lines.append(f"| {p} | {v['num_parts']:.1f} | {v['parts_for_90pct_area']:.1f} | {v['oracle_roughness']:.3f} | {v['oracle_metallic']:.3f} |")
    for c in CHANNELS:
        lines += ["", f"**{c} MAE**", "", "| method | texel | " + " | ".join(parts) + " |", "|---" * (len(parts) + 2) + "|"]
        for m, v in s["methods"].items():
            lines.append(f"| {m} | {v[c]['texel']:.3f} | " + " | ".join(f"{v[c][p]:.3f}" for p in parts) + " |")
    methods = list(s["methods"])
    lines += ["", "**Per item, roughness MAE texel → SAM parts**", "",
              "| item | SAM parts | oracle | " + " | ".join(methods) + " |", "|---" * (len(methods) + 3) + "|"]
    for i, v in s["per_item"].items():
        lines.append(f"| {i[:8]} | {v['partitions']['SAM']['num_parts']:.0f} | {v['partitions']['SAM']['oracle_roughness']:.3f} | "
                     + " | ".join(f"{v['methods'][m]['roughness']['texel']:.3f} → {v['methods'][m]['roughness']['SAM']:.3f}"
                                  for m in methods) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=os.path.join(OUTPUT, "parts", "dataset"))
    ap.add_argument("--max-lightings", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dataset = UnbakeDataset(args.data)
    gen, records, start = None, [], time.time()
    todo = [(item, l) for item in dataset for l in item.lightings[:args.max_lightings]]
    for n, (item, lighting) in enumerate(todo):
        name = input_name(item, lighting)
        d = os.path.join(args.out, name)
        os.makedirs(d, exist_ok=True)
        rep_path = os.path.join(d, "report.json")
        if os.path.isfile(rep_path):
            records.append(json.load(open(rep_path)))
            continue
        glbs = {label: os.path.join(folder, name, PREDICTION_FILE) for label, folder in METHODS.items()}
        missing = [p for p in glbs.values() if not os.path.isfile(p)]
        assert not missing, f"missing predictions: {missing}"
        cache = os.path.join(d, "sam_labels.npz")
        if gen is None and not os.path.isfile(cache):
            gen = load_sam()
        sm = os.path.join(SUPERMAT_PRED, name, "supermat_sv_pred")
        rep = evaluate(item.ground_truth_path, lighting.baked_path, glbs, sm if os.path.isdir(sm) else None,
                       gen, cache, out_dir=d)
        rep.update(item=item.id, lighting=lighting.name)
        json.dump(rep, open(rep_path, "w"), indent=2)
        records.append(rep)
        g = rep["methods"]["Guided (exp. 13)"]["roughness"]
        print(f"[{n + 1}/{len(todo)}] {name[:8]} {lighting.name}: {rep['partitions']['SAM']['num_parts']} parts, "
              f"guided roughness {g['texel']:.3f} -> {g['SAM']:.3f} ({time.time() - start:.0f}s)", flush=True)
        torch.cuda.empty_cache()
    s = summarize(records)
    json.dump({"summary": s, "records": records}, open(os.path.join(args.out, "part_metrics.json"), "w"), indent=2)
    open(os.path.join(args.out, "part_metrics.md"), "w").write(markdown(s))
    print(markdown(s))


if __name__ == "__main__":
    main()
