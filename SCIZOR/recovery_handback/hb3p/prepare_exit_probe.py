"""Prepare the bounded fixed-entry HB3-P intermediate-exit probe."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb3p.io import mark


ROLE = "hb3p_exit_probe"
SEED_START = 500000
SQUARE_SEED_OFFSET = 100000
ROOT_COUNT = 40
SEEDS = list(range(SEED_START + SQUARE_SEED_OFFSET, SEED_START + SQUARE_SEED_OFFSET + ROOT_COUNT))


def _git_head(code_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(code_root.parent), "rev-parse", "HEAD"], text=True
    ).strip()


def prepare(parent_root: Path, hb2_root: Path, output_root: Path, code_root: Path) -> dict:
    parent_config = json.loads((parent_root / "config/hb3p.json").read_text(encoding="utf-8"))
    parent_protocol = json.loads((parent_root / "config/frozen_protocol.json").read_text(encoding="utf-8"))
    for relative in (
        "config", "assets", "roots", "episodes", "metrics", "report", "package",
        "logs", "status", "ipc",
    ):
        (output_root / relative).mkdir(parents=True, exist_ok=True)

    shutil.copy2(parent_root / "assets/assets.json", output_root / "assets/assets.json")
    shutil.copy2(parent_root / "assets/policy_pair_square.json", output_root / "assets/policy_pair_square.json")
    atomic_json_dump({
        "schema_version": "hb3p_exit_probe_resolved_inputs_v1",
        "parent_experiment": str(parent_root.resolve()),
        "parent_protocol_sha256": sha256_file(parent_root / "config/frozen_protocol.json"),
        "semantic_pair_id": parent_protocol["semantic_pair_id"],
        "checkpoints": json.loads((parent_root / "assets/resolved_inputs.json").read_text(encoding="utf-8"))["checkpoints"],
        "source_code_parent_commit": parent_protocol["code_commit"],
    }, output_root / "assets/resolved_inputs.json")

    config = json.loads(json.dumps(parent_config))
    config["assets_file"] = str((output_root / "assets/assets.json").resolve())
    config["output_root"] = str(output_root.resolve())
    config["anchor_times"] = [20]
    config["roles"] = dict(config.get("roles", {}))
    config["roles"][ROLE] = {"seed_start": SEED_START, "roots_per_task": ROOT_COUNT}
    config["hb2"] = dict(config["hb2"])
    config["hb2"]["selected_policy_pair_path"] = str(
        (output_root / "assets/policy_pair_square.json").resolve()
    )
    config["hb3p"] = dict(config["hb3p"])
    config["hb3p"].update({
        "scope": "fixed_entry_exit_probe",
        "candidate_times": [20],
        "helper_lengths": [40, 60, 80],
        "new_test_roots": ROOT_COUNT,
        "test_seed_start": SEEDS[0],
        "role": ROLE,
        "parent_experiment": str(parent_root.resolve()),
    })
    atomic_json_dump(config, output_root / "config/exit_probe.json")

    protocol = json.loads(json.dumps(parent_protocol))
    protocol.update({
        "schema_version": "hb3p_exit_probe_protocol_draft_v1",
        "frozen": False,
        "test_locked": False,
        "scope": "fixed_entry_t20_exit_lengths_40_60_80",
        "source_experiment": str(parent_root.resolve()),
        "source_ref": parent_protocol["source_ref"],
        "candidate_times": [20],
        "helper_lengths": [40, 60, 80],
        "new_test_roots": ROOT_COUNT,
        "test_seed_start": SEEDS[0],
        "test_seeds": SEEDS,
        "method_aliases": [],
        "methods": {
            "NONE": {"kind": "none"},
            "FIXED_L40": {"kind": "scheduled", "scheduled_t": 20, "length": 40},
            "FIXED_L60": {"kind": "scheduled", "scheduled_t": 20, "length": 60},
            "FIXED_L80": {"kind": "scheduled", "scheduled_t": 20, "length": 80},
        },
        "primary_method": "FIXED_L40",
        "primary_comparator": "FIXED_L80",
        "learned_exit_enabled": False,
        "allow_retakeover": False,
        "max_takeovers": 1,
        "online_constraints": {
            "fixed_entry_t": 20,
            "fixed_exit_lengths": [40, 60, 80],
            "candidate_times_only": True,
            "max_takeovers": 1,
            "fixed_exit": True,
            "learned_exit_enabled": False,
            "repeated_takeover_tested": False,
            "arbitrary_query_times_tested": False,
        },
    })
    atomic_json_dump(protocol, output_root / "config/protocol.draft.json")
    (output_root / "config/source_commit.txt").write_text(parent_protocol["source_ref"] + "\n", encoding="utf-8")
    mark(output_root, "exit-probe-prepared.done", f"{ROOT_COUNT} roots, fixed t=20, exits 40/60/80")
    return {
        "output_root": str(output_root.resolve()),
        "role": ROLE,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "code_head": _git_head(code_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--hb2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
