from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


REQUIRED = ("actions", "states_pre", "states_post", "rewards", "staged_rewards", "success")


def text(v):
    return v.decode() if isinstance(v, bytes) else v


def main():
    p = argparse.ArgumentParser(); p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--metadata", type=Path, required=True); p.add_argument("--split-manifest", type=Path); p.add_argument("--expected-pairs", type=int, default=None); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    counts = Counter(); errors = []; pair_rows = []
    with h5py.File(a.benchmark, "r") as h5:
        for name, g in h5["data"].items():
            variant = text(g.attrs.get("variant", "")); task = text(g.attrs.get("task", "")); counts[(task, variant)] += 1
            for key in REQUIRED:
                if key not in g: errors.append(f"{name}:missing:{key}")
            if variant == "clean":
                if "success" in g and (len(g["success"]) == 0 or not bool(g["success"][-1])): errors.append(f"{name}:clean_final_success")
                continue
            if variant != "perturbed": continue
            actions = np.asarray(g["actions"]); pre = np.asarray(g["states_pre"]); post = np.asarray(g["states_post"])
            clean = h5["data/" + text(g.attrs["clean_demo_id"])]
            clean_actions = np.asarray(clean["actions"])
            t = int(g.attrs.get("perturb_t", -1)); diff_count = int(np.count_nonzero(np.any(actions != clean_actions, axis=1)))
            finite = all(np.isfinite(np.asarray(g[key])).all() for key in ("actions", "states_pre", "states_post", "rewards", "staged_rewards"))
            row = {"name": name, "task": task, "perturb_t": t, "action_diff_count": diff_count, "lengths_equal": len(actions) == len(pre) == len(post) == len(g["success"]), "finite": bool(finite), "failure_type": text(g.attrs.get("failure_type", "")), "effective": bool(g.attrs.get("is_effective_intervention", False))}
            pair_rows.append(row)
            if diff_count != 1: errors.append(f"{name}:action_diff_count={diff_count}")
            if not row["lengths_equal"]: errors.append(f"{name}:length_mismatch")
            if not finite: errors.append(f"{name}:nonfinite")
    metadata_rows = [json.loads(x) for x in a.metadata.read_text(encoding="utf-8").splitlines() if x.strip()]
    result = {"benchmark": str(a.benchmark), "metadata_rows": len(metadata_rows), "counts": {f"{t}/{v}": n for (t, v), n in counts.items()}, "pair_count": len(pair_rows), "effective_count": sum(x["effective"] for x in pair_rows), "failure_type_counts": dict(Counter(x["failure_type"] for x in pair_rows)), "action_diff_one_rate": float(sum(x["action_diff_count"] == 1 for x in pair_rows) / len(pair_rows)) if pair_rows else 0.0, "finite_rate": float(sum(x["finite"] for x in pair_rows) / len(pair_rows)) if pair_rows else 0.0, "lengths_equal_rate": float(sum(x["lengths_equal"] for x in pair_rows) / len(pair_rows)) if pair_rows else 0.0, "errors": errors, "audit_pass": not errors}
    if a.split_manifest:
        manifest = json.loads(a.split_manifest.read_text(encoding="utf-8")); seen = set()
        for split, pair_ids in manifest.items():
            for pair_id in pair_ids:
                if pair_id in seen: errors.append(f"duplicate_split_pair:{pair_id}")
                seen.add(pair_id)
        metadata_ids = {str(r["pair_id"]) for r in metadata_rows}
        if seen != metadata_ids: errors.append("split_manifest_metadata_mismatch")
        if "test" in manifest and manifest["test"]: errors.append("v01_test_leakage")
        result["split_counts"] = {k: len(v) for k, v in manifest.items()}
    if a.expected_pairs is not None and len(pair_rows) != a.expected_pairs: errors.append(f"pair_count={len(pair_rows)} expected={a.expected_pairs}")
    result["has_effective_and_no_effect"] = bool(result["effective_count"] and result["failure_type_counts"].get("no_effect", 0))
    # Split validation runs after the initial per-pair scan, so make the exit
    # status reflect every error collected above rather than a stale snapshot.
    result["audit_pass"] = not errors
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2)); raise SystemExit(0 if result["audit_pass"] else 1)


if __name__ == "__main__": main()
