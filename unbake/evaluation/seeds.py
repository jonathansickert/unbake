"""
Mean +- std over runs of one method (e.g. seeds: unbake.baselines.unguided_trellis2 --seed N writes
<predictions>/trellis_unguided_<res>_s<N>, evaluated as method "trellis_unguided_<res>_s<N>").

    python -m unbake.evaluation.seeds "TRELLIS.2 (unguided, 1024)" trellis_unguided_1024_s0 trellis_unguided_1024_s1

Each run is summarized on the (item, lighting) inputs that all given runs have, as in compare.py.
"""
import argparse

import numpy as np

from unbake.evaluation.compare import RELIT_COLS, UV_COLS, load
from unbake.evaluation.summary import summaries


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="method names of the runs (as in the metrics files)")
    ap.add_argument("--eval", default="output/eval", help="folder with metrics_uv.json and metrics_relit.json")
    args = ap.parse_args()

    records, methods, _ = load(args.eval)
    unknown = [r for r in args.runs if r not in methods]
    assert not unknown, f"not in the metrics: {unknown} (available: {methods})"
    _, summary = summaries(records, args.runs)
    fmt = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    head = [f"{c} {m}" for c, m in UV_COLS] + [f"relit {k}" for k, _ in RELIT_COLS]
    print(f"{len(args.runs)} runs, {summary[args.runs[0]]['count']} inputs each\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    print("| " + " | ".join([fmt([summary[r]["uv"][c][m] for r in args.runs]) for c, m in UV_COLS]
                            + [fmt([summary[r]["relit"][k] for r in args.runs]) for k, _ in RELIT_COLS]) + " |")


if __name__ == "__main__":
    main()
