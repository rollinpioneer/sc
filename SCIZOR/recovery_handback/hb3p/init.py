"""Initialize HB3-P from the fixed HB2 experiment without changing HB2."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.hb3p.io import mark


DIRECTORIES = (
    "config",
    "assets",
    "inputs",
    "metrics/clarification",
    "metrics/development",
    "metrics/pilot",
    "metrics/test",
    "report",
    "package",
    "logs",
    "status",
    "ipc",
    "roots",
    "episodes",
)


def _checkpoint(role: str, path: Path, expected: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {role} checkpoint: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} checkpoint hash mismatch: {actual} != {expected}")
    return {
        "role": role,
        "original_path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def initialize(hb2_root: Path, output_root: Path, code_root: Path, source_ref: str, template: Path) -> dict:
    for name in DIRECTORIES:
        (output_root / name).mkdir(parents=True, exist_ok=True)
    config = json.loads((hb2_root / "config/hb2.json").read_text(encoding="utf-8"))
    f_protocol = json.loads((hb2_root / "config/frozen_protocol.json").read_text(encoding="utf-8"))
    h_protocol = json.loads((hb2_root / "config/frozen_handoff_protocol.json").read_text(encoding="utf-8"))
    pair = json.loads((hb2_root / "assets/policy_pair_square.json").read_text(encoding="utf-8"))
    template_data = json.loads(template.read_text(encoding="utf-8"))
    if template_data["source_ref"] != source_ref:
        raise ValueError("template source_ref does not match requested source")

    current_commit = subprocess.check_output(
        ["git", "-C", str(code_root.parent), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != source_ref:
        raise ValueError(f"worktree must start at {source_ref}, got {current_commit}")

    expected = template_data["expected_checkpoints"]
    resolved = [
        _checkpoint("square_base", Path(pair["base"]["checkpoint"]), expected["square_base"]),
        _checkpoint("square_repair", Path(pair["repair"]["checkpoint"]), expected["square_repair"]),
    ]
    for model in ("M0_time", "M1_proprio", "M4_paired"):
        entry = f_protocol["models"][model]["seed0"]
        resolved.append(
            _checkpoint(f"{model}_seed0", Path(entry["checkpoint"]), expected[f"{model}_seed0"])
        )

    copied_inputs = {
        "hb2_frozen_protocol.original.json": hb2_root / "config/frozen_protocol.json",
        "hb2_frozen_handoff_protocol.original.json": hb2_root / "config/frozen_handoff_protocol.json",
        "hb2_decision.original.json": hb2_root / "metrics/hb2_decision.json",
        "hb2_test_summary.original.json": hb2_root / "metrics/test/summary.json",
    }
    for name, source in copied_inputs.items():
        shutil.copy2(source, output_root / "inputs" / name)
    shutil.copy2(hb2_root / "assets/assets.json", output_root / "assets/assets.json")
    shutil.copy2(hb2_root / "assets/policy_pair_square.json", output_root / "assets/policy_pair_square.json")

    config["schema"] = template_data["schema"]
    config["source_ref"] = source_ref
    config["repository"] = template_data["repository"]
    config["research_branch"] = template_data["research_branch"]
    config["roles"]["hb3p_test"] = template_data["roles"]["hb3p_test"]
    config["hb3p"] = template_data["hb3p"]
    config["hb3p"].update({
        "hb2_root": str(hb2_root.resolve()),
        "hb2_frozen_protocol": str((hb2_root / "config/frozen_protocol.json").resolve()),
        "hb2_frozen_handoff_protocol": str((hb2_root / "config/frozen_handoff_protocol.json").resolve()),
        "expected_checkpoints": expected,
        "hb2_status_preserved": "HB2_SIGNAL_PRESENT_VISUAL_GAIN_UNPROVEN",
        "h_control_enabled": False,
    })
    config["assets_file"] = str((output_root / "assets/assets.json").resolve())
    config["output_root"] = str(output_root.resolve())
    config["hb2"]["selected_policy_pair_path"] = str(
        (output_root / "assets/policy_pair_square.json").resolve()
    )
    config_path = output_root / "config/hb3p.json"
    atomic_json_dump(config, config_path)
    (output_root / "config/source_commit.txt").write_text(source_ref + "\n", encoding="utf-8")
    atomic_json_dump({
        "schema_version": "hb3p_resolved_inputs_v1",
        "source_ref": source_ref,
        "semantic_pair_id": f_protocol["semantic_pair_id"],
        "checkpoints": resolved,
        "f_protocol_sha256": sha256_file(hb2_root / "config/frozen_protocol.json"),
        "h_protocol_sha256": sha256_file(hb2_root / "config/frozen_handoff_protocol.json"),
        "h_control_enabled": False,
        "h_exit_model_ready": bool(h_protocol.get("readiness_rule", {}).get("ready", False)),
    }, output_root / "assets/resolved_inputs.json")
    mark(output_root, "hb3p-A-init.done", "fixed inputs resolved and hashed")
    return {"config": str(config_path), "checkpoints": len(resolved), "source_ref": source_ref}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hb2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(initialize(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
