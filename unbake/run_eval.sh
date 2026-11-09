#!/usr/bin/env bash
# Unguided TRELLIS.2 at 512 and 1024 on the whole dataset, then both evaluations of every method
# in $OUT/predictions (also the 2D baselines, if their predictions are there) and the tables.
#
#   $OUT/predictions/<method>/<item>__<lighting>/textured.glb
#   $OUT/eval/metrics_uv.json, metrics_relit.json, items/, renders/
#   $OUT/tables.tex, $OUT/logs/*.log
#
# Prediction steps skip finished inputs, so rerunning continues an interrupted run.
#   nohup unbake/run_eval.sh > output/logs/run.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-output}
mkdir -p "$OUT/logs"

set +u  # conda's activation hooks use unset variables
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate trellis2
set -u

stamp() { echo "[$(date '+%F %T')] $*"; }
for res in 1024 512; do
    stamp "predict, unguided TRELLIS.2 at $res"
    python -m unbake.baselines.unguided_trellis2 --resolution "$res" --output "$OUT/predictions/trellis_unguided_$res" \
        > "$OUT/logs/predict_$res.log" 2>&1
done
for kind in uv relit; do
    stamp "evaluate $kind"
    python -m unbake.evaluation.evaluate_$kind --predictions "$OUT/predictions" --output "$OUT/eval" > "$OUT/logs/evaluate_$kind.log" 2>&1
done
python -m unbake.evaluation.compare --eval "$OUT/eval" --latex "$OUT/tables.tex" > "$OUT/logs/compare.log"
stamp "done: $OUT/eval, $OUT/tables.tex"
