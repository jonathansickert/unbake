"""
Comparison of all methods from the metrics of evaluate_uv.py and evaluate_relit.py: a markdown
table, and two LaTeX tables (booktabs, full text width) for the paper.

    python -m unbake.evaluation.compare                                  # reads output/eval
    python -m unbake.evaluation.compare --latex output/tables.tex

By default only the (item, lighting) inputs that every method has predictions for are used
(--all: every method on all of its inputs; in the LaTeX tables a method with fewer inputs than
the others is marked with a dagger, its subset is named in the caption, and it is not considered
for the best value). Best value per column in bold.
"""
import argparse
import json
import os


from unbake.evaluation.metrics import CHANNELS
from unbake.evaluation.summary import summaries
from unbake.paths import OUTPUT
from unbake.predictions import INPUT_AS_ALBEDO

LATEX_NAMES = {
    INPUT_AS_ALBEDO: "Input as albedo",
    "TRELLIS.2 (unguided, 512)": r"TRELLIS.2 (unguided, $512^3$)",
    "TRELLIS.2 (unguided, 1024)": r"TRELLIS.2 (unguided, $1024^3$)",
    "RGB<->X": r"RGB$\leftrightarrow$X",
}
UV_COLS = [(c, m) for c in CHANNELS for m in ("MAE", "PSNR")]
RELIT_COLS = [("LPIPS", True), ("PSNR", False), ("SSIM", False), ("MAE", True)]


def load(eval_dir):
    with open(os.path.join(eval_dir, "metrics_uv.json")) as f:
        uv = json.load(f)
    with open(os.path.join(eval_dir, "metrics_relit.json")) as f:
        relit = json.load(f)
    # one record per (item, lighting, method) with both metric groups
    key = lambda r: (r["item"], r["lighting"], r["method"])
    relit_by_key = {key(r): r["relit"] for r in relit["records"]}
    records = [{**r, "relit": relit_by_key[key(r)]} for r in uv["records"] if key(r) in relit_by_key]
    methods = [m for m in uv["methods"] if m in relit["methods"]]
    return records, methods, relit


def markdown(summary):
    head = [f"{c} {m}" for c, m in UV_COLS] + [f"relit {k}" for k, _ in RELIT_COLS]
    lines = ["| method | n | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
    for name, s in summary.items():
        cells = [s["uv"][c][m] for c, m in UV_COLS] + [s["relit"][k] for k, _ in RELIT_COLS]
        lines.append(f"| {name} | {s['count']} | " + " | ".join(f"{x:.3f}" if abs(x) < 10 else f"{x:.2f}" for x in cells) + " |")
    return "\n".join(lines)


def latex(summary, pairs, num_envs, num_views, label_suffix=""):
    """`pairs`: {method: set of (item, lighting) inputs behind its row}."""
    names = list(summary)
    full = max(len(pairs[m]) for m in names)
    partial = [m for m in names if len(pairs[m]) < full]
    reference = next(m for m in names if m not in partial)

    def extent(m):  # "10 assets $\times$ 8 baked lighting conditions"
        n_items = len({item for item, _ in pairs[m]})
        return r"%d assets $\times$ %d baked lighting conditions" % (n_items, len(pairs[m]) // n_items)
    note = "".join(r" $^\dagger$%s: %s only." % (LATEX_NAMES.get(m, m), extent(m)) for m in partial)

    def col(vals, lower, fmt):
        best = (min if lower else max)(v for v, m in zip(vals, names) if m not in partial)
        return [(r"\textbf{%s}" if v == best and m not in partial else "%s") % fmt(v) for v, m in zip(vals, names)]
    f3, f2 = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.2f}")
    label = lambda m: LATEX_NAMES.get(m, m) + (r"$^\dagger$" if m in partial else "")
    rows = lambda cols: [f"{label(m)} & " + " & ".join(c[i] for c in cols) + r" \\" for i, m in enumerate(names)]
    full_width = lambda n: r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}l" + "c" * n + "}"

    cols = [col([summary[m]["uv"][c][metric] for m in names], metric == "MAE", f3 if metric == "MAE" else f2)
            for c, metric in UV_COLS]
    t1 = [r"\begin{table}[t]", r"\centering",
          r"\caption{Material accuracy in UV space, averaged over %s. "
          r"MAE is weighted by surface area; PSNR (dB) is computed from the mean squared error. The input-as-albedo "
          r"baseline uses the baked texture as albedo with constant roughness 0.5 and metallic 0.%s}" % (extent(reference), note),
          r"\label{tab:materials%s}" % label_suffix, r"\small", full_width(6), r"\toprule",
          r" & \multicolumn{2}{c}{Albedo} & \multicolumn{2}{c}{Roughness} & \multicolumn{2}{c}{Metallic} \\",
          r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
          r"Method & MAE $\downarrow$ & PSNR $\uparrow$ & MAE $\downarrow$ & PSNR $\uparrow$ & MAE $\downarrow$ & PSNR $\uparrow$ \\",
          r"\midrule", *rows(cols), r"\bottomrule", r"\end{tabular*}", r"\end{table}"]

    cols = [col([summary[m]["relit"][k] for m in names], lower, f2 if k == "PSNR" else f3) for k, lower in RELIT_COLS]
    t2 = [r"\begin{table}[t]", r"\centering",
          r"\caption{Relighting under %d held-out environment maps not used to create the dataset, %d views each, "
          r"averaged over %s. PSNR in dB.%s}" % (num_envs, num_views, extent(reference), note),
          r"\label{tab:relighting%s}" % label_suffix, r"\small", full_width(4), r"\toprule",
          r"Method & LPIPS $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & MAE $\downarrow$ \\",
          r"\midrule", *rows(cols), r"\bottomrule", r"\end{tabular*}", r"\end{table}"]

    return (r"% Requires \usepackage{booktabs}. Generated by unbake/evaluation/compare.py from metrics_uv.json and metrics_relit.json." + "\n"
            r"% Both tables span the full text width (\linewidth) of a single-column layout." + "\n\n"
            + "\n".join(t1) + "\n\n" + "\n".join(t2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", default=os.path.join(OUTPUT, "eval"), help="folder with metrics_uv.json and metrics_relit.json")
    ap.add_argument("--all", action="store_true", help="every method on all of its inputs, not only the common ones")
    ap.add_argument("--latex", default=None, help="also write the LaTeX tables to this file")
    ap.add_argument("--methods", nargs="+", default=None, help="only these methods (names as in the metrics files)")
    ap.add_argument("--label-suffix", default="", help="appended to the LaTeX labels, e.g. -subset")
    args = ap.parse_args()

    records, methods, relit = load(args.eval)
    if args.methods:
        unknown = [m for m in args.methods if m not in methods]
        assert not unknown, f"not in the metrics: {unknown} (available: {methods})"
        methods = [m for m in methods if m in args.methods]
    summary, summary_common = summaries(records, methods)
    if not args.all:
        summary = summary_common
    # inputs behind each row: all of the method's (--all) or those every method has
    pairs = {m: {(r["item"], r["lighting"]) for r in records if r["method"] == m} for m in summary}
    if not args.all:
        common = set.intersection(*pairs.values())
        pairs = {m: common for m in pairs}
    print(markdown(summary))
    if args.latex:
        with open(args.latex, "w") as f:
            f.write(latex(summary, pairs, len(relit["env_maps"]), relit["num_views"], args.label_suffix))
        print(f"\nSaved {args.latex}")


if __name__ == "__main__":
    main()
