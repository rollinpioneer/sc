"""Prepare the independent HB3-P stop/continue pilot directories and draft."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file


SEED_START = 700000
ROOT_COUNT = 40


def prepare(probe_root: Path, output_root: Path, code_root: Path) -> dict:
    probe_root, output_root, code_root = Path(probe_root), Path(output_root), Path(code_root)
    for relative in ("config", "assets", "dataset", "model", "roots", "episodes", "metrics", "report", "package", "logs", "status"):
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    for name in ("assets.json", "policy_pair_square.json"):
        shutil.copy2(probe_root / "assets" / name, output_root / "assets" / name)
    parent_protocol = json.loads((probe_root / "config/frozen_protocol.json").read_text(encoding="utf-8"))
    config = json.loads((probe_root / "config/exit_probe.json").read_text(encoding="utf-8"))
    config.update({
        "schema_version": "hb3p_stop_continue_config_v1",
        "assets_file": str((output_root / "assets/assets.json").resolve()),
        "output_root": str(output_root.resolve()),
        "horizon_steps": 400,
        "anchor_t": 20,
        "decision_t": 80,
        "stop_length": 60,
        "continue_length": 80,
        "expected_control_freq": 20,
        "minimum_autonomous_steps": 20,
        "lambda": 0.25,
        "noninferiority_margin_absolute": 0.05,
        "probe_root": str(probe_root.resolve()),
        "probe_protocol_sha256": sha256_file(probe_root / "config/frozen_protocol.json"),
        "policy_pair_path": str((output_root / "assets/policy_pair_square.json").resolve()),
        "role": "hb3p_stop_continue_test",
        "test_seed_start": SEED_START,
        "new_test_roots": ROOT_COUNT,
        "features": {
            "frames": 4,
            "proprio_dim": 9,
            "action_dim": 7,
            "feature_dim": 65,
            "source": "FIXED_L60 handoff history at t=80",
        },
        "training": {"seed": 20260909, "epochs": 300, "device": "cuda"},
    })
    config["hb2"] = dict(config.get("hb2", {}))
    config["hb2"]["selected_policy_pair_path"] = config["policy_pair_path"]
    atomic_json_dump(config, output_root / "config/stop_continue.json")
    draft = {
        "schema_version": "hb3p_stop_continue_protocol_draft_v1",
        "frozen": False,
        "test_locked": False,
        "scope": "fixed_entry_t20_stop_at_t80_or_continue_to_t100",
        "source_probe_protocol_sha256": config["probe_protocol_sha256"],
        "source_probe_root": str(probe_root.resolve()),
        "source_ref": parent_protocol["source_ref"],
        "semantic_pair_id": parent_protocol["semantic_pair_id"],
        "role": "hb3p_stop_continue_test",
        "horizon_steps": 400,
        "minimum_autonomous_steps": 20,
        "lambda": 0.25,
        "statistics": {"unit": "one_complete_episode_per_stat_group_id", "bootstrap_repeats": 2000, "seed": 20260909},
        "anchor_t": 20,
        "decision_t": 80,
        "stop_interval": [20, 80],
        "continue_interval": [20, 100],
        "max_takeovers": 1,
        "model_queries": [80],
        "selected_short_exit": 60,
        "methods": ["NONE", "FIXED_L60", "FIXED_L80", "LEARNED_STOP_CONTINUE"],
        "new_test_roots": ROOT_COUNT,
        "test_seed_start": SEED_START,
        "test_seeds": list(range(SEED_START, SEED_START + ROOT_COUNT)),
        "success_rate_noninferiority_margin_absolute": 0.05,
        "noninferiority_margin_absolute": 0.05,
        "noninferiority_comparator": "FIXED_L80",
        "short_exit_comparator": "FIXED_L60",
        "pilot": True,
        "formal_claim_allowed": False,
        "pilot_reason": "40 roots is below the estimated sample size for a formal 0.05 non-inferiority claim",
        "input_schema": {
            "allowed": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "base_action_history", "absolute_time"],
            "forbidden": ["reward", "success", "object_state", "future_result", "root_id", "seed", "repair_action", "images"],
        },
    }
    atomic_json_dump(draft, output_root / "config/protocol.draft.json")
    (output_root / "config/source_commit.txt").write_text(parent_protocol["source_ref"] + "\n", encoding="utf-8")
    return {"output_root": str(output_root.resolve()), "seed_range": [SEED_START, SEED_START + ROOT_COUNT - 1], "code_root": str(code_root.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
