"""Merge relit metrics: keep the per-asset entries of the persistent copy for assets the new evaluation could
not render (their textured.glb is not persisted), recompute the summary as the plain mean over assets (as
evaluate_relit.py does) and write the merged file to both places.

    python method/merge_relit.py <work method dir> <keep method dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

work, keep = Path(sys.argv[1]) / "metrics_relit.json", Path(sys.argv[2]) / "metrics_relit.json"
new = json.loads(work.read_text()) if work.exists() else None
old = json.loads(keep.read_text()) if keep.exists() else None
if new is None and old is None:
    sys.exit(0)
merged = new or old
if old and new:
    merged["assets"] = {**old["assets"], **new["assets"]}
summary = {}
for m in ["pred", "input_as_albedo"]:
    rows = [r[m] for r in merged["assets"].values() if m in r]
    if rows:
        summary[m] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        summary[m]["n_assets"] = len(rows)
merged["summary"] = summary
for p in [work, keep]:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2))
print(f"{work.parent}: relit for {len(merged['assets'])} assets")
