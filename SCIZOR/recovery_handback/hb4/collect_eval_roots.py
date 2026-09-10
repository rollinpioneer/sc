"""Register canonical evaluation roots without loading any policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_array, sha256_file, write_table


SOURCE = Path("/home/__compress_data/xushijie/work/cr_scizor/data/robomimic/square/ph/image.hdf5")


def collect(start_seed: int, count: int, role: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "roots.jsonl"
    existing = {}
    if records_path.is_file():
        existing = {int(row["seed"]): row for row in (json.loads(line) for line in records_path.read_text().splitlines() if line.strip())}
    env = EnvAdapter(SOURCE, **load_observation_spec(SOURCE))
    rows = dict(existing)
    try:
        for seed in range(start_seed, start_seed + count):
            if seed in rows:
                continue
            _obs, payload = env.new_episode(seed)
            stem = f"{seed}"
            payload_path = output_dir / f"{stem}.npz"
            model_path = output_dir / f"{stem}.xml"
            meta_path = output_dir / f"{stem}.json"
            np.savez_compressed(payload_path, states=np.asarray(payload["states"], dtype=np.float64))
            model_path.write_text(payload["model"], encoding="utf-8")
            meta_path.write_text(json.dumps({"seed": seed, "role": role, "env_name": payload["env_name"], "control_freq": payload["control_freq"]}, sort_keys=True, indent=2), encoding="utf-8")
            rows[seed] = {
                "schema_version": "hb4_eval_root_v1", "role": role, "task": "square", "seed": seed,
                "canonical_group_id": f"square:HB4_{role}:{seed}",
                "payload_path": str(payload_path.resolve()), "model_path": str(model_path.resolve()),
                "meta_path": str(meta_path.resolve()), "initial_state_hash": sha256_array(np.asarray(payload["states"], dtype=np.float64)),
                "model_hash": sha256_file(model_path), "control_freq": int(payload["control_freq"]), "horizon_steps": 400,
            }
            ordered = sorted(rows.values(), key=lambda row: row["seed"])
            write_table(ordered, records_path)
            atomic_json_dump({"schema_version": "hb4_eval_root_summary_v1", "role": role, "seed_range": [start_seed, start_seed + count - 1], "roots": len(ordered), "rows": ordered}, output_dir / "summary.json")
    finally:
        env.close()
    summary = json.loads((output_dir / "summary.json").read_text())
    return {"role": role, "roots": summary["roots"], "seed_range": summary["seed_range"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.start_seed, args.count, args.role, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
