"""Select at most eight fixed, interpretable HB1 cases without cherry-picking."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recovery_handback.common import atomic_json_dump, read_table


def _slug(value: str) -> str:
    return value.replace(":", "__").replace("/", "_")


def _save_history_images(source: str | None, anchor_t: int, output_dir: Path, prefix: str) -> int:
    if not source or not Path(source).is_file():
        return 0
    saved = 0
    with np.load(source, allow_pickle=False) as handle:
        for key in sorted(handle.files):
            value = np.asarray(handle[key])
            if not key.endswith("_image") or value.ndim != 3:
                continue
            if prefix == "anchor" and not key.startswith(f"{anchor_t}_"):
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.imsave(output_dir / f"{prefix}_{key}.png", value.astype(np.uint8))
            saved += 1
    return saved


def _write_timeline(branch: dict, anchor_t: int, output: Path) -> int:
    trajectory = Path(str(branch.get("trajectory_path", "")))
    if not trajectory.is_file():
        return 0
    with np.load(trajectory, allow_pickle=False) as handle:
        actions = np.asarray(handle["actions"], dtype=np.float32)
        base = np.asarray(handle["base_actions"], dtype=np.float32)
        helper = np.asarray(handle["helper_mask"], dtype=bool)
        success = np.asarray(handle["success"], dtype=bool)
        rewards = np.asarray(handle["rewards"], dtype=np.float32)
    rows = []
    for index in range(len(actions)):
        row = {
            "absolute_t": anchor_t + index,
            "controller": "helper" if helper[index] else "base",
            "helper_active": bool(helper[index]),
            "action_delta_l2": float(np.linalg.norm(actions[index] - base[index])),
            "raw_success": bool(success[index]),
            "reward": float(rewards[index]),
        }
        for dim in range(actions.shape[1]):
            row[f"action_{dim}"] = float(actions[index, dim])
            row[f"base_action_{dim}"] = float(base[index, dim])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paired-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args()
    paired = pd.DataFrame(read_table(args.paired_results))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    used = set()
    category_counts = {}
    for anchor_id, subset in paired.groupby("anchor_id", sort=True):
        by = subset.set_index("branch_name")
        if not bool(subset["complete_pair"].iloc[0]):
            continue
        if by.loc["none", "system_success"]:
            category = "autonomous_help_unnecessary_or_harmful"
        elif any(bool(by.loc[name, "genuine_handoff_success"]) for name in ("l5", "l20", "l80")):
            category = "genuine_short_handoff"
        elif any(bool(by.loc[name, "helper_completed_task"]) for name in ("l5", "l20", "l80")) or bool(by.loc["full", "system_success"]):
            category = "helper_completed_without_genuine_handoff"
        else:
            finite = [bool(by.loc[name, "system_success"]) for name in ("l5", "l20", "l80")]
            category = "unresolved_or_nonmonotonic" if finite != sorted(finite) or not any(finite) else "unresolved_or_nonmonotonic"
        if category_counts.get(category, 0) >= 2 or anchor_id in used:
            continue
        category_counts[category] = category_counts.get(category, 0) + 1
        used.add(anchor_id)
        selected.append({
            "anchor_id": anchor_id, "task": subset.iloc[0]["task"], "root_id": subset.iloc[0]["root_id"],
            "anchor_t": int(subset.iloc[0]["anchor_t"]), "category": category,
            "branches": json.dumps({
                name: {
                    "system_success": bool(by.loc[name, "system_success"]),
                    "genuine_handoff_success": bool(by.loc[name, "genuine_handoff_success"]),
                    "helper_completed_task": bool(by.loc[name, "helper_completed_task"]),
                    "helper_steps_actual": int(by.loc[name, "helper_steps_actual"] or 0),
                    "trajectory_path": by.loc[name, "trajectory_path"],
                    "handoff_history_path": by.loc[name].get("handoff_history_path"),
                } for name in ("none", "l5", "l20", "l80", "full")
            }, sort_keys=True),
            "anchor_history_path": subset.iloc[0].get("anchor_history_path"),
        })
        if len(selected) >= args.max_cases:
            break
    pd.DataFrame(selected).to_csv(args.output_dir / "selected_examples.csv", index=False)
    exported = []
    for index, case in enumerate(selected):
        case_dir = args.output_dir / "cases" / f"{index:02d}_{_slug(case['anchor_id'])}"
        case_dir.mkdir(parents=True, exist_ok=True)
        branches = json.loads(case["branches"])
        image_count = _save_history_images(
            case.get("anchor_history_path"), int(case["anchor_t"]), case_dir, "anchor"
        )
        timeline_rows = 0
        for branch_name, branch in branches.items():
            timeline_rows += _write_timeline(
                branch, int(case["anchor_t"]), case_dir / f"timeline_{branch_name}.csv"
            )
            image_count += _save_history_images(
                branch.get("handoff_history_path"), int(case["anchor_t"]),
                case_dir, f"handoff_{branch_name}",
            )
        case_record = {
            **case, "image_snippets": image_count,
            "timeline_rows": timeline_rows,
            "source_only_export": True,
        }
        atomic_json_dump(case_record, case_dir / "case.json")
        exported.append({
            "anchor_id": case["anchor_id"], "directory": str(case_dir.resolve()),
            "image_snippets": image_count, "timeline_rows": timeline_rows,
        })
    atomic_json_dump({
        "schema_version": "hb1_selected_cases_v1", "selected": len(selected),
        "category_counts": category_counts,
        "missing_categories": [
            name for name in (
                "genuine_short_handoff", "helper_completed_without_genuine_handoff",
                "autonomous_help_unnecessary_or_harmful", "unresolved_or_nonmonotonic",
            ) if category_counts.get(name, 0) < 2
        ],
        "rendering": "trajectory_and_action_timeline_only_unless_saved_render_assets_are_available",
        "exports": exported,
    }, args.output_dir / "selection_manifest.json")


if __name__ == "__main__":
    main()
