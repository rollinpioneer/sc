"""Build paired HB1 metrics, clustered intervals, and the stage decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from recovery_handback.common import atomic_json_dump, read_table, sha256_json, write_table
from recovery_handback.execution.label_reference import BRANCH_SPECS, diagnostic_shortest


BRANCHES = [name for name, _ in BRANCH_SPECS]
FINITE = ["l5", "l20", "l80"]


def _load_anchors(roots_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(roots_root.glob("*/probe/anchors.parquet")):
        rows.extend(read_table(path))
    if not rows:
        raise RuntimeError(f"no probe anchors found under {roots_root}")
    frame = pd.DataFrame(rows)
    if frame["anchor_id"].duplicated().any():
        raise RuntimeError("duplicate authoritative anchor_id")
    return frame


def _load_branches(branches_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(branches_root.glob("*/branch_results.parquet")):
        rows.extend(read_table(path))
    if not rows:
        return pd.DataFrame(columns=["task", "anchor_id", "branch_name"])
    frame = pd.DataFrame(rows)
    if frame.duplicated(["task", "anchor_id", "branch_name"]).any():
        duplicates = frame.loc[frame.duplicated(["task", "anchor_id", "branch_name"], keep=False), ["task", "anchor_id", "branch_name"]]
        raise RuntimeError(f"duplicate branch primary keys: {duplicates.head().to_dict(orient='records')}")
    return frame


def _bootstrap_ci(root_values: np.ndarray, repeats: int, seed: int) -> tuple[float, float]:
    values = np.asarray(root_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, [0.025, 0.975]))


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _pivot(paired: pd.DataFrame, value: str) -> pd.DataFrame:
    return paired.pivot(index="anchor_id", columns="branch_name", values=value).reindex(columns=BRANCHES)


def _curves(paired: pd.DataFrame, config: dict, group_columns: list[str]) -> pd.DataFrame:
    output = []
    repeats = int(config["statistics"]["bootstrap_groups"])
    seed = int(config["statistics"]["seed"])
    group_arg = group_columns[0] if len(group_columns) == 1 else group_columns
    for group_index, (group_key, subset) in enumerate(paired.groupby(group_arg, dropna=False)):
        keys = (group_key,) if len(group_columns) == 1 else tuple(group_key)
        labels = dict(zip(group_columns, keys))
        complete_ids = subset.groupby("anchor_id")["complete_pair"].first()
        complete = subset[subset["anchor_id"].isin(complete_ids[complete_ids].index)].copy()
        if complete.empty:
            for branch in BRANCHES:
                output.append({
                    **labels, "branch_name": branch, "anchors": 0, "roots": 0,
                    "system_success_rate_anchor": float("nan"),
                    "system_success_rate_root": float("nan"),
                    "system_success_ci95_low": float("nan"),
                    "system_success_ci95_high": float("nan"),
                    "paired_success_rate_difference_root": float("nan"),
                    "paired_difference_ci95_low": float("nan"),
                    "paired_difference_ci95_high": float("nan"),
                    "genuine_rescue_anchors": 0, "genuine_rescue_roots": 0,
                    "rescue_fraction_among_baseline_failed_anchors": float("nan"),
                    "helper_completed_anchors": 0, "helper_completed_fraction": float("nan"),
                    "handoff_success_raw_fraction": float("nan"),
                    "genuine_handoff_success_fraction": float("nan"),
                    "success_seen_under_helper_anchors": 0,
                    "success_seen_under_helper_fraction": float("nan"),
                    "handoff_failure_fraction": float("nan"),
                    "first_success_wait_after_handoff_mean": float("nan"),
                    "first_success_wait_after_handoff_median": float("nan"),
                    "first_success_wait_after_handoff_p25": float("nan"),
                    "first_success_wait_after_handoff_p75": float("nan"),
                    "autonomous_execution_steps_mean": float("nan"),
                    "autonomous_execution_steps_median": float("nan"),
                    "mean_helper_steps_actual": float("nan"),
                    "mean_changed_action_steps": float("nan"),
                    "mean_helper_steps_on_baseline_success": float("nan"),
                    "mean_changed_action_steps_on_baseline_success": float("nan"),
                    "repair_calls_after_handoff_total": 0,
                    "interference_fraction_among_baseline_success": float("nan"),
                })
            continue
        y = _pivot(complete, "system_success").fillna(False).astype(bool)
        genuine = _pivot(complete, "genuine_handoff_success").fillna(False).astype(bool)
        helper_done = _pivot(complete, "helper_completed_task").fillna(False).astype(bool)
        handoff_raw = _pivot(complete, "handoff_success_raw").fillna(False).astype(bool)
        success_under_helper = _pivot(complete, "success_seen_under_helper").fillna(False).astype(bool)
        costs = _pivot(complete, "helper_steps_actual").astype(float)
        changed = _pivot(complete, "changed_action_steps").astype(float)
        index_meta = complete.drop_duplicates("anchor_id").set_index("anchor_id")
        root_ids = index_meta["stat_group_id"].reindex(y.index)
        root_success = y.astype(float).groupby(root_ids).mean().reindex(columns=BRANCHES)
        rng = np.random.default_rng(seed + group_index)
        sampled_indices = rng.integers(
            0, len(root_success), size=(repeats, len(root_success))
        )
        sampled_success = root_success.to_numpy()[sampled_indices].mean(axis=1)
        none_index = BRANCHES.index("none")
        for branch in BRANCHES:
            anchor_metric = y[branch].astype(float)
            root_metric = root_success[branch]
            branch_index = BRANCHES.index(branch)
            ci_low, ci_high = (
                float(value)
                for value in np.quantile(sampled_success[:, branch_index], [0.025, 0.975])
            )
            baseline_failed = ~y["none"]
            rescue = baseline_failed & genuine[branch] if branch in FINITE else pd.Series(False, index=y.index)
            paired_delta = y[branch].astype(int) - y["none"].astype(int)
            root_delta = root_success[branch] - root_success["none"]
            sampled_delta = sampled_success[:, branch_index] - sampled_success[:, none_index]
            delta_low, delta_high = (
                float(value) for value in np.quantile(sampled_delta, [0.025, 0.975])
            )
            handoff_rows = complete[(complete["branch_name"] == branch) & _bool(complete["handoff_executed"])]
            successful_handoff_rows = handoff_rows[_bool(handoff_rows["handoff_success_raw"])]
            wait = pd.to_numeric(
                successful_handoff_rows.get("first_success_wait_after_handoff"), errors="coerce"
            ).dropna()
            autonomous_execution = pd.to_numeric(
                handoff_rows.get("autonomous_execution_steps_after_handoff"), errors="coerce"
            ).dropna()
            baseline_success_ids = y.index[y["none"]]
            unnecessary_rows = complete[
                (complete["branch_name"] == branch)
                & complete["anchor_id"].isin(baseline_success_ids)
            ]
            output.append({
                **labels, "branch_name": branch, "anchors": int(len(y)),
                "roots": int(index_meta["stat_group_id"].nunique()),
                "system_success_rate_anchor": float(anchor_metric.mean()),
                "system_success_rate_root": float(root_metric.mean()),
                "system_success_ci95_low": ci_low, "system_success_ci95_high": ci_high,
                "paired_success_rate_difference_root": float(root_delta.mean()),
                "paired_difference_ci95_low": delta_low, "paired_difference_ci95_high": delta_high,
                "genuine_rescue_anchors": int(rescue.sum()),
                "genuine_rescue_roots": int(index_meta.loc[rescue, "stat_group_id"].nunique()),
                "rescue_fraction_among_baseline_failed_anchors": float(rescue.sum() / baseline_failed.sum()) if baseline_failed.sum() else float("nan"),
                "helper_completed_anchors": int(helper_done[branch].sum()),
                "helper_completed_fraction": float(helper_done[branch].mean()),
                "handoff_success_raw_fraction": float(handoff_raw[branch].mean()),
                "genuine_handoff_success_fraction": float(genuine[branch].mean()),
                "success_seen_under_helper_anchors": int(success_under_helper[branch].sum()),
                "success_seen_under_helper_fraction": float(success_under_helper[branch].mean()),
                "handoff_failure_fraction": float((~_bool(handoff_rows["system_success"])).mean()) if len(handoff_rows) else float("nan"),
                "first_success_wait_after_handoff_mean": float(wait.mean()) if len(wait) else float("nan"),
                "first_success_wait_after_handoff_median": float(wait.median()) if len(wait) else float("nan"),
                "first_success_wait_after_handoff_p25": float(wait.quantile(0.25)) if len(wait) else float("nan"),
                "first_success_wait_after_handoff_p75": float(wait.quantile(0.75)) if len(wait) else float("nan"),
                "autonomous_execution_steps_mean": float(autonomous_execution.mean()) if len(autonomous_execution) else float("nan"),
                "autonomous_execution_steps_median": float(autonomous_execution.median()) if len(autonomous_execution) else float("nan"),
                "mean_helper_steps_actual": float(costs[branch].mean()),
                "mean_changed_action_steps": float(changed[branch].mean()),
                "mean_helper_steps_on_baseline_success": float(unnecessary_rows["helper_steps_actual"].mean()) if len(unnecessary_rows) else float("nan"),
                "mean_changed_action_steps_on_baseline_success": float(unnecessary_rows["changed_action_steps"].mean()) if len(unnecessary_rows) else float("nan"),
                "repair_calls_after_handoff_total": int(pd.to_numeric(complete.loc[complete["branch_name"] == branch, "repair_calls_after_handoff"], errors="coerce").fillna(0).sum()),
                "interference_fraction_among_baseline_success": float(((y["none"]) & (~y[branch])).sum() / y["none"].sum()) if y["none"].sum() else float("nan"),
            })
    return pd.DataFrame(output)


def _opportunities(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for anchor_id, subset in paired.groupby("anchor_id", sort=True):
        by_branch = {row["branch_name"]: row for row in subset.to_dict(orient="records")}
        complete = bool(subset["complete_pair"].iloc[0])
        shortest = diagnostic_shortest(by_branch) if complete else "incomplete"
        saving = None
        if shortest in (5, 20, 80):
            finite = by_branch[f"l{shortest}"]
            full = by_branch["full"]
            full_cost = int(full.get("helper_steps_actual") or 0)
            if bool(finite.get("system_success")) and bool(finite.get("genuine_handoff_success")) and bool(full.get("system_success")) and full_cost > 0:
                saving = 1.0 - int(finite.get("helper_steps_actual") or 0) / full_cost
        meta = subset.iloc[0]
        rows.append({
            "anchor_id": anchor_id, "task": meta["task"], "role": meta.get("role", "probe"),
            "cohort": meta.get("cohort", "main"), "root_id": meta["root_id"],
            "stat_group_id": meta["stat_group_id"], "anchor_t": int(meta["anchor_t"]),
            "complete_pair": complete, "diagnostic_shortest": shortest,
            "conditional_helper_cost_saving": saving,
            "full_success": bool(by_branch.get("full", {}).get("system_success", False)),
        })
    return pd.DataFrame(rows)


def _decision(paired: pd.DataFrame, opportunities: pd.DataFrame, policy_pairs_dir: Path, config: dict) -> dict:
    task_results = {}
    qualified_tasks = []
    for task in config["tasks"]:
        task_rows = paired[paired["task"] == task]
        task_ops = opportunities[opportunities["task"] == task]
        complete_ops = task_ops[task_ops["complete_pair"]]
        complete_ids = set(complete_ops["anchor_id"])
        complete_rows = task_rows[task_rows["anchor_id"].isin(complete_ids)]
        y = _pivot(complete_rows, "system_success").fillna(False).astype(bool) if complete_ids else pd.DataFrame(columns=BRANCHES)
        g = _pivot(complete_rows, "genuine_handoff_success").fillna(False).astype(bool) if complete_ids else pd.DataFrame(columns=BRANCHES)
        meta = complete_rows.drop_duplicates("anchor_id").set_index("anchor_id") if complete_ids else pd.DataFrame()
        valid_roots = int(meta["stat_group_id"].nunique()) if complete_ids else 0
        failed_ids = y.index[~y["none"]] if complete_ids else []
        failed_roots = int(meta.loc[failed_ids, "stat_group_id"].nunique()) if len(failed_ids) else 0
        rescued_mask = ((~y["none"]) & g[FINITE].any(axis=1)) if complete_ids else pd.Series(dtype=bool)
        rescued_roots = int(meta.loc[rescued_mask, "stat_group_id"].nunique()) if len(rescued_mask) else 0
        control_mask = y["none"] | (~y[FINITE + ["full"]].any(axis=1)) if complete_ids else pd.Series(dtype=bool)
        control_roots = int(meta.loc[control_mask, "stat_group_id"].nunique()) if len(control_mask) else 0
        pair_path = policy_pairs_dir / f"policy_pair_{task}.json"
        repair_status = None
        task_status = None
        if pair_path.is_file():
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
            task_status = pair.get("status")
            repair_status = pair.get("qualification", {}).get("status")
        if task_status == "NEED_BASE_POLICY" or repair_status == "NEED_BASE_POLICY":
            status = "NEED_BASE_POLICY"
        elif repair_status == "HOLD_ENGINEERING_FIX":
            status = "HOLD_ENGINEERING_FIX"
        elif repair_status == "NEED_STRONGER_REPAIRER":
            status = "NEED_STRONGER_REPAIRER"
        elif len(task_ops) == 0 or len(complete_ops) < len(task_ops):
            status = "HOLD_ENGINEERING_FIX"
        elif valid_roots < int(config["feasibility"]["min_probe_roots"]):
            status = "INSUFFICIENT_VALID_ROOTS"
        elif failed_roots < int(config["feasibility"]["min_baseline_failed_roots"]):
            status = "INSUFFICIENT_NATURAL_FAILURES"
        elif rescued_roots >= int(config["feasibility"]["min_genuine_rescued_roots"]) and control_roots >= int(config["feasibility"]["min_control_roots"]):
            status = "READY_HB2"
            qualified_tasks.append(task)
        elif y["full"].sum() > 0:
            status = "NO_LOCAL_HANDOFF_EVIDENCE_FOR_CURRENT_PAIR_AND_GRID"
        else:
            status = "NEED_STRONGER_REPAIRER"
        shortest_counts = complete_ops["diagnostic_shortest"].astype(str).value_counts().to_dict()
        finite_nonzero = [key for key in ("5", "20", "80") if shortest_counts.get(key, 0)]
        duration_note = "DURATION_ADAPTATION_UNPROVEN" if len(finite_nonzero) <= 1 and rescued_roots > 0 else None
        task_results[task] = {
            "status": status, "duration_note": duration_note,
            "authoritative_anchors": int(len(task_ops)), "complete_anchors": int(len(complete_ops)),
            "valid_roots": valid_roots, "baseline_failed_roots": failed_roots,
            "genuine_rescued_roots": rescued_roots, "control_roots": control_roots,
            "repair_qualification_status": repair_status,
        }
    if len(qualified_tasks) == len(config["tasks"]):
        overall = "READY_HB2"
    elif qualified_tasks:
        overall = "READY_HB2_SINGLE_TASK"
    else:
        overall = "HOLD"
    return {"schema_version": "hb1_decision_v1", "status": overall, "ready_tasks": qualified_tasks, "tasks": task_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--branches-root", type=Path, required=True)
    parser.add_argument("--roots-root", type=Path, required=True)
    parser.add_argument("--policy-pairs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    anchors = _load_anchors(args.roots_root)
    branches = _load_branches(args.branches_root)
    expected = anchors.assign(_join=1).merge(pd.DataFrame({"branch_name": BRANCHES, "_join": 1}), on="_join").drop(columns="_join")
    overlap = [column for column in branches.columns if column in expected.columns and column not in ("task", "anchor_id", "branch_name")]
    paired = expected.merge(branches.drop(columns=overlap, errors="ignore"), on=["task", "anchor_id", "branch_name"], how="left", validate="one_to_one")
    result_defaults = {
        "schema_version": None, "policy_pair_hash": None, "config_hash": None,
        "engineering_ok": False, "system_success": False,
        "genuine_handoff_success": False, "helper_completed_task": False,
        "handoff_success_raw": False, "success_seen_under_helper": False,
        "helper_steps_actual": float("nan"), "changed_action_steps": float("nan"),
        "handoff_executed": False, "first_success_wait_after_handoff": float("nan"),
        "autonomous_execution_steps_after_handoff": float("nan"),
        "repair_calls_after_handoff": 0, "trajectory_path": None,
        "handoff_history_path": None,
    }
    for column, default in result_defaults.items():
        if column not in paired:
            paired[column] = default
    paired["branch_present"] = paired["schema_version"].notna() if "schema_version" in paired else False
    expected_config_hash = sha256_json(config)
    pair_hashes = {}
    for task in config["tasks"]:
        pair_path = args.policy_pairs_dir / f"policy_pair_{task}.json"
        if pair_path.is_file():
            pair_hashes[task] = sha256_json(json.loads(pair_path.read_text(encoding="utf-8")))
    paired["identity_ok"] = (
        paired.get("config_hash", pd.Series(None, index=paired.index)).eq(expected_config_hash)
        & paired.apply(lambda row: row.get("policy_pair_hash") == pair_hashes.get(row["task"]), axis=1)
    )
    paired["engineering_ok"] = _bool(paired.get("engineering_ok", pd.Series(False, index=paired.index))) & paired["identity_ok"]
    paired["complete_pair"] = paired.groupby("anchor_id")["engineering_ok"].transform(lambda values: bool(len(values) == 5 and values.all()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_table(paired.to_dict(orient="records"), args.output_dir / "paired_results.parquet")
    opportunities = _opportunities(paired)
    write_table(opportunities.to_dict(orient="records"), args.output_dir / "handoff_opportunities.parquet")
    curves_task = _curves(paired, config, ["task"])
    curves_time = _curves(paired, config, ["task", "anchor_t"])
    curves_task.to_csv(args.output_dir / "curves_by_task.csv", index=False)
    curves_time.to_csv(args.output_dir / "curves_by_anchor_time.csv", index=False)
    complete = paired[paired["complete_pair"]].copy()
    if len(complete):
        root_long = complete.groupby(["task", "stat_group_id", "branch_name"], as_index=False).agg(
            system_success=("system_success", "mean"), genuine_handoff_success=("genuine_handoff_success", "mean"),
            helper_steps_actual=("helper_steps_actual", "mean"), anchors=("anchor_id", "nunique"),
        )
        root_summary = root_long.pivot(index=["task", "stat_group_id"], columns="branch_name", values=["system_success", "genuine_handoff_success", "helper_steps_actual"]).reset_index()
        root_summary.columns = [
            "_".join(str(part) for part in column if part).rstrip("_") if isinstance(column, tuple) else str(column)
            for column in root_summary.columns
        ]
    else:
        root_summary = pd.DataFrame(columns=["task", "stat_group_id"])
    write_table(root_summary.to_dict(orient="records"), args.output_dir / "roots_summary.parquet")
    coverage = {
        "schema_version": "hb1_coverage_v1", "authoritative_anchors": int(anchors["anchor_id"].nunique()),
        "expected_branches": int(len(expected)), "present_branches": int(paired["branch_present"].sum()),
        "engineering_ok_branches": int(paired["engineering_ok"].sum()),
        "complete_anchors": int(paired.loc[paired["complete_pair"], "anchor_id"].nunique()),
        "coverage_fraction": float(paired.loc[paired["complete_pair"], "anchor_id"].nunique() / anchors["anchor_id"].nunique()),
        "missing_keys": paired.loc[~paired["branch_present"], ["task", "anchor_id", "branch_name"]].to_dict(orient="records"),
    }
    atomic_json_dump(coverage, args.output_dir / "coverage.json")
    decision = _decision(paired, opportunities, args.policy_pairs_dir, config)
    atomic_json_dump(decision, args.output_dir / "hb1_decision.json")
    y = _pivot(paired, "system_success")
    g = _pivot(paired, "genuine_handoff_success")
    helper = _pivot(paired, "helper_completed_task")
    costs = _pivot(paired, "helper_steps_actual")
    anchor_meta = anchors.set_index("anchor_id")
    dataset = []
    for anchor_id in anchor_meta.index:
        row = anchor_meta.loc[anchor_id]
        dataset.append({
            "anchor_id": anchor_id, "stat_group_id": row["stat_group_id"], "task": row["task"],
            "role": row.get("role", "probe"), "cohort": row.get("cohort", "main"),
            "anchor_history_path": row.get("anchor_history_path"), "proprio_history_path": row.get("anchor_history_path"),
            "base_action_at_anchor": (
                np.asarray(np.load(row["rollout_path"], allow_pickle=False)["suggestions"][int(row["anchor_t"])], dtype=np.float32).tolist()
                if Path(row["rollout_path"]).is_file() else None
            ),
            "remaining_steps": int(row["remaining_steps"]),
            "policy_pair_hash": paired.loc[paired["anchor_id"] == anchor_id, "policy_pair_hash"].dropna().iloc[0] if paired.loc[paired["anchor_id"] == anchor_id, "policy_pair_hash"].notna().any() else None,
            "y0": y.at[anchor_id, "none"] if anchor_id in y.index else None,
            "y5": y.at[anchor_id, "l5"] if anchor_id in y.index else None,
            "y20": y.at[anchor_id, "l20"] if anchor_id in y.index else None,
            "y80": y.at[anchor_id, "l80"] if anchor_id in y.index else None,
            "yfull": y.at[anchor_id, "full"] if anchor_id in y.index else None,
            "g5": g.at[anchor_id, "l5"] if anchor_id in g.index else None,
            "g20": g.at[anchor_id, "l20"] if anchor_id in g.index else None,
            "g80": g.at[anchor_id, "l80"] if anchor_id in g.index else None,
            "helper_completed_5": helper.at[anchor_id, "l5"] if anchor_id in helper.index else None,
            "helper_completed_20": helper.at[anchor_id, "l20"] if anchor_id in helper.index else None,
            "helper_completed_80": helper.at[anchor_id, "l80"] if anchor_id in helper.index else None,
            "cost5": costs.at[anchor_id, "l5"] if anchor_id in costs.index else None,
            "cost20": costs.at[anchor_id, "l20"] if anchor_id in costs.index else None,
            "cost80": costs.at[anchor_id, "l80"] if anchor_id in costs.index else None,
            "costfull": costs.at[anchor_id, "full"] if anchor_id in costs.index else None,
            "complete_pair": bool(paired.loc[paired["anchor_id"] == anchor_id, "complete_pair"].all()),
            "schema_version": "hb1_handoff_curve_v1",
        })
    write_table(dataset, args.output_dir / "handoff_curve_dataset.parquet")
    print(json.dumps({"coverage": coverage, "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
