from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .common import compare_rollouts, env_for_dataset, replay, text


def split_bases(split_manifest, pair_meta):
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    pair_split = {}
    for split in ("train", "validation", "test"):
        for pair_id in manifest.get(split, []):
            pair_split[str(pair_id)] = split
    result = {"can": {}, "square": {}}
    for line in Path(pair_meta).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = pair_split.get(row.get("pair_id"))
        if split in ("train", "validation"):
            result[str(row["task"])].setdefault(str(row["base_demo_id"]), split)
    return result


def demo_number(demo_id):
    try:
        return int(str(demo_id).rsplit("_", 1)[-1])
    except ValueError:
        return 10**9


def check_demo(source, demo_id, env_a, env_b, source_h5, max_steps=None):
    group = source_h5[f"data/{demo_id}"]
    initial = np.asarray(group["states"][0]).copy()
    actions = np.asarray(group["actions"][:max_steps] if max_steps else group["actions"][:]).copy()
    model_xml = env_a.env.model.get_xml()
    a = replay(env_a, initial, actions, render_images=False, model_xml=model_xml)
    b = replay(env_b, initial, actions, render_images=False, model_xml=model_xml)
    comparison = compare_rollouts(a, b)
    return {
        "task": "can" if "can" in str(source).lower() else "square",
        "demo_id": demo_id,
        "steps": int(len(actions)),
        "determinism_pass": bool(comparison["pass"]),
        "final_success": bool(a["success"][-1]) if len(a["success"]) else False,
        "comparison": comparison,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--can-source", type=Path, required=True)
    p.add_argument("--square-source", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--pair-meta", type=Path, required=True)
    p.add_argument("--num-per-task", type=int, default=5)
    p.add_argument("--allow-fewer", action="store_true", help="write every passing allowed candidate instead of requiring the requested cap")
    p.add_argument("--allowed-splits", default="train,validation")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--detail-output", type=Path, required=True)
    a = p.parse_args()
    allowed = [x.strip() for x in a.allowed_splits.split(",")]
    bases = split_bases(a.split_manifest, a.pair_meta)
    sources = {"can": a.can_source, "square": a.square_source}
    selected, details = {"can": [], "square": []}, []
    environments = {}
    source_handles = {}
    try:
        for task in ("can", "square"):
            environments[task] = (env_for_dataset(sources[task])[0], env_for_dataset(sources[task])[0])
            source_handles[task] = h5py.File(sources[task], "r")
            candidates = [d for d, split in bases[task].items() if split in allowed]
            candidates.sort(key=demo_number)
            target = min(a.num_per_task, len(candidates)) if a.allow_fewer else a.num_per_task
            for demo_id in candidates:
                if len(selected[task]) >= target:
                    break
                row = check_demo(sources[task], demo_id, *environments[task], source_handles[task])
                row["split"] = bases[task][demo_id]
                details.append(row)
                if row["determinism_pass"] and row["final_success"]:
                    selected[task].append(row)
    finally:
        for env_a, env_b in environments.values():
            env_a.close(); env_b.close()
        for handle in source_handles.values():
            handle.close()
    if not a.allow_fewer and any(len(selected[t]) < a.num_per_task for t in selected):
        raise RuntimeError("insufficient deterministic successful demos: " + json.dumps({k: len(v) for k, v in selected.items()}))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.detail_output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    a.detail_output.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in details) + "\n", encoding="utf-8")
    print(json.dumps({"selected": {k: [x["demo_id"] for x in v] for k, v in selected.items()}, "tested": len(details)}, indent=2))


if __name__ == "__main__":
    main()
