"""
Shared by the evaluation scripts: command-line arguments, and summaries / printed tables of the
per-record metrics (one record per item x lighting x method).
"""
import math

import numpy as np

from ..dataset import UnbakeDataset
from ..predictions import PREDICTIONS, methods
from .metrics import CHANNELS


def add_eval_args(parser, output="output/eval"):
    parser.add_argument("--data", default="data", help="dataset root")
    parser.add_argument("--predictions", default=PREDICTIONS, help="folder with one subfolder per method")
    parser.add_argument("--max-items", type=int, default=None, help="evaluate only the first N items")
    parser.add_argument("--output", default=output, help="output directory")


def evaluation_inputs(args):
    """(dataset, item indices, methods) for the evaluation scripts."""
    dataset = UnbakeDataset(args.data)
    return dataset, list(range(len(dataset)))[:args.max_items], methods(args.predictions)


def summarize(records: list[dict]) -> dict:
    """
    Over records (one per item x lighting x method) of one method. UV space: MAE averaged over
    records, PSNR from the mean MSE (a perfect channel, e.g. metallic 0 everywhere, has infinite
    PSNR on its own). Relit: every metric averaged over records and env maps.
    """
    out = {"count": len(records)}
    if "uv" in records[0]:
        out["uv"] = {c: {"MAE": float(np.mean([r["uv"][c]["MAE"] for r in records])),
                         "PSNR": 10 * math.log10(1 / m) if (m := float(np.mean([r["uv"][c]["MSE"] for r in records]))) > 0
                         else float("inf")}
                     for c in CHANNELS}
    if "relit" in records[0]:
        envs = list(records[0]["relit"])
        metrics = list(records[0]["relit"][envs[0]])
        per_env = {env: {k: float(np.mean([r["relit"][env][k] for r in records])) for k in metrics} for env in envs}
        out["relit"] = {k: float(np.mean([per_env[env][k] for env in envs])) for k in metrics}
        out["relit_per_env"] = per_env
    return out


def summaries(records: list[dict], names: list[str]) -> tuple[dict, dict]:
    """Per method: summary over all its records, and over the (item, lighting) pairs every method has."""
    by_method = {n: [r for r in records if r["method"] == n] for n in names}
    by_method = {n: rows for n, rows in by_method.items() if rows}
    pairs = [{(r["item"], r["lighting"]) for r in rows} for rows in by_method.values()]
    common = set.intersection(*pairs) if pairs else set()
    return ({n: summarize(rows) for n, rows in by_method.items()},
            {n: summarize([r for r in rows if (r["item"], r["lighting"]) in common])
             for n, rows in by_method.items() if common})


def print_table(title, rows, cols, fmt):
    print(f"\n{title}")
    print(f"{'':30s}{'n':>6s}" + "".join(f"{c:>14s}" for c in cols))
    for name, (count, vals) in rows.items():
        print(f"{name:30s}{count:6d}" + "".join(f"{fmt(c, vals[c]):>14s}" for c in cols))


def report(summary: dict, heading: str, num_envs: int = None, num_views: int = None):
    print(f"\n==================== {heading}")
    first = next(iter(summary.values()))
    if "uv" in first:
        for metric in ("MAE", "PSNR"):
            print_table(f"UV space: {metric} ({'lower' if metric == 'MAE' else 'higher'} is better)",
                        {n: (r["count"], {c: r["uv"][c][metric] for c in CHANNELS}) for n, r in summary.items()},
                        list(CHANNELS), lambda c, v: f"{v:.4f}" if metric == "MAE" else f"{v:.2f} dB")
    if "relit" in first:
        print_table(f"Relit renders, mean over {num_envs} env maps x {num_views} views "
                    "(LPIPS/MAE lower, PSNR/SSIM higher is better)",
                    {n: (r["count"], r["relit"]) for n, r in summary.items()},
                    ["LPIPS", "PSNR", "SSIM", "MAE"], lambda c, v: f"{v:.2f} dB" if c == "PSNR" else f"{v:.4f}")


def report_all(records: list[dict], names: list[str], missing: dict, **kwargs) -> tuple[dict, dict]:
    summary, summary_common = summaries(records, names)
    report(summary, "all available predictions (n = item x lighting pairs)", **kwargs)
    if summary_common and any(r["count"] != next(iter(summary_common.values()))["count"] for r in summary.values()):
        report(summary_common, f"only the {next(iter(summary_common.values()))['count']} pairs every method has predictions for",
               **kwargs)
    for name, paths in missing.items():
        if paths:
            print(f"\n{name}: {len(paths)} predictions missing (skipped), e.g. {paths[0]}")
    return summary, summary_common
