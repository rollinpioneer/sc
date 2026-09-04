from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def describe(values):
    if not values: return {"n": 0}
    x = np.asarray(values, float)
    return {"n": len(values), "min": float(x.min()), "q25": float(np.quantile(x, .25)), "median": float(np.median(x)), "mean": float(x.mean()), "q75": float(np.quantile(x, .75)), "max": float(x.max())}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--metadata", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--horizons", default="20,40,60,80,100"); a = p.parse_args()
    buckets = defaultdict(list); kept = 0
    for line in a.metadata.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        if not r.get("is_effective_intervention") or r.get("failure_onset") is None: continue
        d = int(r["failure_onset"]) - int(r["perturb_t"])
        if d < 0: continue
        kept += 1
        for key in ("all", f"split/{r['split']}", f"task/{r['task']}", f"split_task/{r['split']}/{r['task']}"): buckets[key].append(d)
    horizons = [int(x) for x in a.horizons.split(",")]
    result = {"effective_with_onset": kept, "descriptive": {k: describe(v) for k, v in sorted(buckets.items())}, "coverage": {k: {str(h): {"count": sum(x <= h for x in v), "fraction": sum(x <= h for x in v) / len(v)} for h in horizons} for k, v in sorted(buckets.items())}}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
