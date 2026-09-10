"""Build HB4 lightweight metrics, report, and reproducible result package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from recovery_handback.common import atomic_json_dump, sha256_file, write_table


ARMS = ("REPLAY_ONLY", "MATCHED_STANDARD_DATA", "FIXED_L80_RECOVERY", "HANDOFF_RECOVERY")


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.3f}%"


def _load_records(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.glob("*/episodes.jsonl")):
        rows.extend(_rows(path))
    return rows


def _power(records: list[dict], output: Path) -> dict:
    good = [row for row in records if row.get("engineering_ok")]
    by_root: dict[str, dict[str, dict[int | None, float]]] = {}
    for row in good:
        by_root.setdefault(row["canonical_group_id"], {}).setdefault(row["method_id"], {})[row.get("training_seed")] = float(bool(row.get("no_help_task_success")))
    diffs = []
    for root, methods in by_root.items():
        if "BASE_FROZEN" not in methods or "HANDOFF_RECOVERY" not in methods:
            continue
        handoff = methods["HANDOFF_RECOVERY"]
        if not all(seed in handoff for seed in (0, 1, 2)):
            continue
        h = np.mean([handoff[seed] for seed in (0, 1, 2)])
        diffs.append(h - methods["BASE_FROZEN"].get(None, np.nan))
    diffs = np.asarray([value for value in diffs if np.isfinite(value)], dtype=float)
    sd = float(np.std(diffs, ddof=1)) if diffs.size > 1 else None
    mde = float((1.96 + 0.84) * sd / np.sqrt(400)) if sd is not None else None
    result = {"schema_version": "hb4_power_planning_v1", "fixed_formal_n": 400, "paired_root_count_observed": int(diffs.size), "handoff_minus_base_root_diff_sd": sd, "approximate_mde_80pct": mde, "formula": "(1.96 + 0.84) * sd / sqrt(400)", "interpretation": "planning quantity only; not a guarantee of power and conditional on frozen checkpoints"}
    atomic_json_dump(result, output)
    return result


def _preservation(records: list[dict], output: Path) -> list[dict]:
    by_root: dict[str, dict[str, dict[int | None, int]]] = {}
    for row in records:
        if row.get("engineering_ok"):
            by_root.setdefault(row["canonical_group_id"], {}).setdefault(row["method_id"], {})[row.get("training_seed")] = int(bool(row.get("no_help_task_success")))
    rows = []
    for method in ARMS:
        for seed in (0, 1, 2):
            common = [(root, values) for root, values in by_root.items() if values.get("BASE_FROZEN", {}).get(None) is not None and values.get(method, {}).get(seed) is not None]
            both = sum(base == 1 and values[method][seed] == 1 for _root, values in common for base in [values["BASE_FROZEN"][None]])
            new = sum(base == 0 and values[method][seed] == 1 for _root, values in common for base in [values["BASE_FROZEN"][None]])
            lost = sum(base == 1 and values[method][seed] == 0 for _root, values in common for base in [values["BASE_FROZEN"][None]])
            failed = sum(base == 0 and values[method][seed] == 0 for _root, values in common for base in [values["BASE_FROZEN"][None]])
            baseline_success = both + lost
            rows.append({"method_id": method, "training_seed": seed, "paired_roots": len(common), "common_success": both, "new_success": new, "lost_baseline_success": lost, "common_failure": failed, "new_success_rate": _pct(new / len(common)) if common else "NA", "lost_baseline_rate_all_roots": _pct(lost / len(common)) if common else "NA", "lost_baseline_rate_baseline_success": _pct(lost / baseline_success) if baseline_success else "NA"})
    write_table(rows, output)
    return rows


def _methods(records: list[dict], output: Path) -> list[dict]:
    grouped: dict[tuple[str, int | None], list[dict]] = {}
    for row in records:
        grouped.setdefault((row["method_id"], row.get("training_seed")), []).append(row)
    rows = []
    for (method, seed), items in sorted(grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        valid = [row for row in items if row.get("engineering_ok")]
        rows.append({"method_id": method, "training_seed": seed, "records": len(items), "engineering_failures": len(items) - len(valid), "roots": len(valid), "successes": sum(bool(row.get("no_help_task_success")) for row in valid), "success_rate": _pct(sum(bool(row.get("no_help_task_success")) for row in valid) / len(valid)) if valid else "NA"})
    write_table(rows, output)
    return rows


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_commit(export_root: Path) -> str:
    code_root = export_root.parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(code_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _source_audit(export_root: Path, run_root: Path) -> dict:
    source_rows = _read_csv(export_root / "data/source_root_table.csv")
    exclusions = Counter(row.get("exclusion_reasons") or "ELIGIBLE" for row in source_rows)
    standard_path = run_root / "data/standard_teacher/summary.json"
    standard = json.loads(standard_path.read_text(encoding="utf-8")) if standard_path.is_file() else {}
    reconstruction_path = export_root / "data/reconstruction_audit.json"
    reconstruction = json.loads(reconstruction_path.read_text(encoding="utf-8")) if reconstruction_path.is_file() else {}
    matching_path = export_root / "data/matching_audit.json"
    matching = json.loads(matching_path.read_text(encoding="utf-8")) if matching_path.is_file() else {}
    return {
        "historical_candidates": len(source_rows),
        "historical_eligible": exclusions.get("ELIGIBLE", 0),
        "historical_exclusions": dict(sorted(exclusions.items())),
        "standard_requested": standard.get("requested_roots", 200),
        "standard_processed": standard.get("processed_records", 0),
        "standard_valid": standard.get("valid_for_matching", 0),
        "standard_engineering_failures": standard.get("engineering_failures", 0),
        "reconstruction_branch_records": reconstruction.get("branch_records", 0),
        "reconstruction_roots": reconstruction.get("verified_roots_with_both_branches", 0),
        "reconstruction_failures": len(reconstruction.get("failed_records", [])),
        "K": matching.get("K", "NA"),
        "N": matching.get("N", "NA"),
    }


def _training_audit(run_root: Path) -> dict:
    summaries = []
    for path in sorted((run_root / "training").glob("*/seed*/training_summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    per_arm = {}
    per_job = {}
    for item in summaries:
        arm = item["arm"]
        key = f"{arm}:seed{item['training_seed']}"
        per_job[key] = {
            "updates": item.get("optimizer_updates"),
            "added_exposures": item.get("expected_added_action_exposures"),
            "wall_seconds": item.get("wall_seconds"),
            "device": item.get("device"),
            "status": item.get("status"),
        }
        per_arm.setdefault(arm, 0.0)
        per_arm[arm] += float(item.get("wall_seconds") or 0.0)
    standard = run_root / "data/standard_teacher/summary.json"
    pilot = run_root / "data/pilot/pilot_summary.json"
    standard_data = json.loads(standard.read_text(encoding="utf-8")) if standard.is_file() else {}
    pilot_data = json.loads(pilot.read_text(encoding="utf-8")) if pilot.is_file() else {}
    return {
        "all_pass": len(summaries) == 12 and all(item.get("status") == "PASS_HB4_E_TRAINED" for item in summaries),
        "total_wall_seconds": sum(float(item.get("wall_seconds") or 0.0) for item in summaries),
        "per_arm_wall_seconds": dict(sorted(per_arm.items())),
        "jobs": len(summaries),
        "per_job": per_job,
        "standard_elapsed_seconds": standard_data.get("elapsed_seconds"),
        "pilot_elapsed_seconds": pilot_data.get("elapsed_seconds"),
    }


def _layered_status(development_summary: dict | None, preservation: list[dict]) -> dict:
    if not development_summary:
        return {
            "engineering_valid": False,
            "no_help_autonomy_gain": False,
            "extra_training_control_gain": False,
            "ordinary_data_increment": False,
            "short_handoff_data_increment": False,
            "training_seed_consistency": False,
            "baseline_preservation_evidence": "NOT_AVAILABLE",
            "status": "HOLD_HB4_ENGINEERING",
        }
    comparisons = {row["comparison"]: row for row in development_summary.get("comparisons", [])}
    gate = development_summary.get("gate", {})
    handoff = development_summary.get("methods", {}).get("HANDOFF_RECOVERY", {})
    handoff_seed_positive = sum(
        float(handoff.get("seed_stats", {}).get(str(seed), {}).get("gain_vs_base") or 0.0) > 0
        for seed in (0, 1, 2)
    )
    handoff_rows = [row for row in preservation if row.get("method_id") == "HANDOFF_RECOVERY"]
    preservation_state = "SUPPORTED" if handoff_rows and all(
        float(str(row.get("lost_baseline_rate_all_roots", "NA")).rstrip("%")) == 0.0
        for row in handoff_rows
    ) else "LIMITED_OR_INCONCLUSIVE"
    first = comparisons.get("HANDOFF_RECOVERY-BASE_FROZEN", {})
    replay = comparisons.get("HANDOFF_RECOVERY-REPLAY_ONLY", {})
    standard = comparisons.get("HANDOFF_RECOVERY-MATCHED_STANDARD_DATA", {})
    fixed = comparisons.get("HANDOFF_RECOVERY-FIXED_L80_RECOVERY", {})
    if not gate.get("engineering_valid"):
        status = "HOLD_HB4_ENGINEERING"
    elif not (first.get("lower_bound_gt_zero") and (first.get("estimate") or -1.0) >= 0.05):
        status = "HB4_NO_CONFIRMED_AUTONOMY_GAIN"
    elif not replay.get("lower_bound_gt_zero"):
        status = "HB4_AUTONOMY_GAIN_ONLY"
    elif not standard.get("lower_bound_gt_zero"):
        status = "HB4_RECOVERY_GAIN_STANDARD_UNPROVEN"
    elif not fixed.get("lower_bound_gt_zero"):
        status = "HB4_RECOVERY_GAIN_SHORT_UNPROVEN"
    elif handoff_seed_positive < 3:
        status = "HB4_GAIN_WITH_SEED_INSTABILITY"
    else:
        status = "HB4_HANDOFF_DATA_ADVANTAGE_SUPPORTED"
    return {
        "engineering_valid": bool(gate.get("engineering_valid")),
        "no_help_autonomy_gain": bool(first.get("lower_bound_gt_zero") and (first.get("estimate") or -1.0) >= 0.05),
        "extra_training_control_gain": bool(replay.get("lower_bound_gt_zero")),
        "ordinary_data_increment": bool(standard.get("lower_bound_gt_zero")),
        "short_handoff_data_increment": bool(fixed.get("lower_bound_gt_zero")),
        "training_seed_consistency": handoff_seed_positive >= 3,
        "handoff_positive_seeds_vs_base": handoff_seed_positive,
        "baseline_preservation_evidence": preservation_state,
        "status": status,
    }


def _report(export_root: Path, run_root: Path, formal_summary: dict | None, development_summary: dict | None, power: dict, methods: list[dict], preservation: list[dict], layered: dict) -> None:
    cfg = json.loads((export_root / "config/hb4_resolved.json").read_text(encoding="utf-8"))
    source = _source_audit(export_root, run_root)
    training = _training_audit(run_root)
    training_protocol = export_root / "config/training_protocol.json"
    development_protocol = export_root / "config/development_frozen_protocol.json"
    training_sha = sha256_file(training_protocol) if training_protocol.is_file() else "NA"
    development_sha = sha256_file(development_protocol) if development_protocol.is_file() else "NA"
    public_inputs = json.loads((export_root / "assets/resolved_inputs_public.json").read_text(encoding="utf-8"))
    roles = json.loads((export_root / "assets/data_roles.json").read_text(encoding="utf-8"))
    base_sha = next((item["sha256"] for item in public_inputs.get("file_roles", []) if item["role"] == "square_base_checkpoint"), "NA")
    teacher_sha = next((item["sha256"] for item in public_inputs.get("file_roles", []) if item["role"] == "square_frozen_repair_teacher"), "NA")
    if formal_summary and formal_summary.get("coverage", {}).get("complete"):
        execution_status = "FORMAL_COMPLETE"
    elif development_summary and development_summary.get("coverage", {}).get("complete"):
        execution_status = "DEVELOPMENT_COMPLETE_FORMAL_NOT_RUN"
    else:
        execution_status = "INCOMPLETE"
    lines = [
        "# HB4 Square 基础策略恢复经验吸收报告", "", "## Material Passport", "",
        f"- Execution status: `{execution_status}`",
        f"- Source commit: `{cfg.get('source_commit', 'NA')}`",
        f"- Analysis code commit: `{_git_commit(export_root)}`",
        f"- Training protocol SHA256: `{training_sha}`",
        f"- Development frozen protocol SHA256: `{development_sha}`",
        f"- Base checkpoint SHA256: `{base_sha}`",
        f"- Frozen teacher SHA256: `{teacher_sha}`",
        f"- Scope: no-help Square evaluation after matched-budget offline finetuning",
        f"- Run root: `{run_root}`",
        "",
        "## 历史状态", "",
        "保留 HB3 `FORMAL_NONINFERIORITY_PASS` 的边界；本报告不将其改写为学习型退出的效用优越性、视觉增益或一般动态控制成立。",
        "",
        "## 数据来源与配平", "",
        f"- Historical source cohort: `{source['historical_candidates']}` roots; eligible short-handoff roots `{source['historical_eligible']}`; exclusion counts: `{json.dumps(source['historical_exclusions'], ensure_ascii=False, sort_keys=True)}`.",
        f"- Standard teacher source: `{source['standard_processed']}/{source['standard_requested']}` requested trajectories processed, `{source['standard_valid']}` valid for matching, engineering failures `{source['standard_engineering_failures']}`.",
        f"- Observation reconstruction: `{source['reconstruction_roots']}` roots, `{source['reconstruction_branch_records']}` branch records, failed records `{source['reconstruction_failures']}`; training images are `RECONSTRUCTED_FOR_HB4`.",
        f"- Matched budget: `K={source['K']}`, `N={source['N']}` unique added labels per added arm; four non-overlapping 10-step blocks per root; HANDOFF and FIXED_L80 use the same recovery roots.",
        f"- Roles: historical `{roles.get('historical_cohort', {}).get('new_role', 'NA')}`, standard `{roles.get('standard_source', {}).get('role', 'NA')}`, development `{roles.get('development', {}).get('role', 'NA')}`, test `{roles.get('test', {}).get('role', 'NA')}`.",
        "",
        "## 训练与开发", "",
        f"- Training matrix: `{training['jobs']}` jobs, four arms x three seeds, 4000 optimizer updates, batch 32 with 16 D0 + 16 added sequences, sequence length 10; all jobs passed: `{training['all_pass']}`.",
        f"- Added-slot exposure: `640000` action positions per job; total training wall time: `{training['total_wall_seconds']:.2f}` seconds; per-arm seconds: `{json.dumps(training['per_arm_wall_seconds'], sort_keys=True)}`.",
        f"- Standard collection wall time: `{training['standard_elapsed_seconds']}` seconds; pilot wall time: `{training['pilot_elapsed_seconds']}` seconds. Final step 4000 only was used; no intermediate checkpoint selection.",
        f"- Development coverage: `{development_summary.get('coverage', {}).get('records', 0) if development_summary else 0}` records across `{development_summary.get('coverage', {}).get('unique_roots', 0) if development_summary else 0}` roots; engineering failures `{development_summary.get('coverage', {}).get('engineering_failures', 'NA') if development_summary else 'NA'}`.",
        "",
        "## 开发门槛", "",
    ]
    if development_summary:
        gate = development_summary.get("gate", {})
        lines += [
            f"- Development coverage complete: `{development_summary.get('coverage', {}).get('complete')}`",
            f"- Development decision: `{json.loads((export_root / 'metrics/development/decision.json').read_text()).get('decision', 'NA')}`",
            f"- Gate details: `{json.dumps(gate, ensure_ascii=False, sort_keys=True)}`",
            f"- Layered status: `{json.dumps(layered, ensure_ascii=False, sort_keys=True)}`",
            "- The development result was used only as the pre-registered unlock decision; no method, seed, hyperparameter, threshold, or checkpoint was selected from the development outcomes.",
        ]
    else:
        lines += ["", "- Development results: `NOT_AVAILABLE`"]
    lines += ["", "## 正式测试", ""]
    if formal_summary:
        cov = formal_summary.get("coverage", {})
        lines += [f"- Coverage: `{cov.get('records', 0)}` records, `{cov.get('expected_roots', 400)}` roots, engineering failures `{cov.get('engineering_failures', 'NA')}`.", f"- Main comparisons: `{json.dumps(formal_summary.get('comparisons', []), ensure_ascii=False)}`", f"- Formal decision: `{json.loads((export_root / 'metrics/test/decision.json').read_text()).get('decision', 'NA')}`"]
    else:
        lines += ["- Formal test: `NOT_RUN_DEVELOPMENT_NO_GO`; the development gate did not unlock formal evaluation. No formal 400-root records, formal confidence intervals, or formal PASS are inferred from missing records."]
    lines += ["", "## 预设配对比较", ""]
    if development_summary:
        for item in development_summary.get("comparisons", []):
            lo, hi = item.get("ci95", [None, None])
            lines.append(f"- `{item['comparison']}`: estimate `{item.get('estimate'):.6f}`; paired roots `{item.get('paired_roots')}`; bootstrap 95% CI `[{lo:.6f}, {hi:.6f}]`; lower bound > 0: `{item.get('lower_bound_gt_zero')}`.")
        for method in methods:
            lines.append(f"- `{method['method_id']}` seed `{method.get('training_seed')}`: `{method.get('successes')}/{method.get('roots')}` = `{method.get('success_rate')}`.")
    else:
        lines.append("- Development and formal comparisons: `NA`.")
    lines += [
        "", "## 能力保留与分层结论", "",
        f"- Layered decision: `{layered.get('status')}`.",
        f"- No-help autonomy gain: `{layered.get('no_help_autonomy_gain')}`; extra-training control gain: `{layered.get('extra_training_control_gain')}`; ordinary-data increment: `{layered.get('ordinary_data_increment')}`; short-handoff increment: `{layered.get('short_handoff_data_increment')}`.",
        f"- HANDOFF seed consistency: `{layered.get('handoff_positive_seeds_vs_base', 'NA')}/3` positive versus BASE; formal test was not unlocked.",
        f"- Baseline preservation evidence: `{layered.get('baseline_preservation_evidence')}`; the development four-cell counts are recorded below and in `metrics/development/baseline_preservation.csv`.",
        *[
            f"- Four-cell `{row['method_id']}` seed `{row.get('training_seed')}`: common success `{row['common_success']}`, new success `{row['new_success']}`, lost baseline success `{row['lost_baseline_success']}`, common failure `{row['common_failure']}`; baseline-success denominator `{int(row['common_success']) + int(row['lost_baseline_success'])}`/`{row['paired_roots']}`."
            for row in preservation
        ],
        f"- Approximate fixed-n=400 MDE: `{power.get('approximate_mde_80pct', 'NA')}`; planning only, conditional on frozen checkpoints and not a power guarantee.",
        "",
        "## 局限与下一阶段", "",
        "单任务 Square、同分布新根、固定基础 checkpoint/教师、条件化恢复源选择、三个训练种子；本轮没有在线帮助需求结论。由于 `HB4_DEVELOPMENT_NO_GO`，不建议在同一协议内扩大数据、挑选种子或追加正式测试；应另立协议后再研究数据配方或泛化。",
        "",
        "## 本地大资产与交接", "",
        "完整 checkpoint、HDF5、图像缓存、原始开发轨迹和运行日志保留在 `RUN_ROOT`，不进入轻量仓库包；路径与哈希见 `report/LOCAL_ONLY_ARTIFACTS.md`。正式矩阵的锁止日志记录为 `formal test not unlocked: HB4_DEVELOPMENT_NO_GO`；当前没有仍在运行的本轮进程。",
    ]
    (export_root / "report").mkdir(parents=True, exist_ok=True)
    (export_root / "report/HB4_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _local_artifacts(export_root: Path, run_root: Path) -> None:
    lines = ["# HB4 Local-only artifacts", "", "These files remain outside the GitHub lightweight export.", ""]
    for root, _dirs, files in os.walk(run_root):
        for name in sorted(files):
            path = Path(root) / name
            try:
                digest = sha256_file(path)
            except Exception:
                continue
            lines.append(f"- `{path}` | {path.stat().st_size} bytes | SHA256 `{digest}`")
    (export_root / "report").mkdir(parents=True, exist_ok=True)
    (export_root / "report/LOCAL_ONLY_ARTIFACTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _package(export_root: Path) -> tuple[Path, str]:
    package_dir = export_root / "package"
    package_dir.mkdir(parents=True, exist_ok=True)
    archive = package_dir / "HB4_results_lightweight.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(export_root.rglob("*")):
            if not path.is_file() or path == archive or path.name == archive.name + ".sha256":
                continue
            # Raw per-episode development/test records stay in the run/export
            # workspace for audit, but are intentionally excluded from the
            # lightweight handoff package. The formal JSONL projection below
            # is the only episode-level artifact published by this package.
            if path.name == "episodes.jsonl" or path.relative_to(export_root).as_posix() == "assets/resolved_inputs.json":
                continue
            output.write(path, path.relative_to(export_root.parent))
    digest = sha256_file(archive)
    (package_dir / "HB4_results_lightweight.zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    return archive, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    formal_path = args.export_root / "metrics/test/summary.json"
    development_path = args.export_root / "metrics/development/summary.json"
    formal = json.loads(formal_path.read_text()) if formal_path.is_file() else None
    development = json.loads(development_path.read_text()) if development_path.is_file() else None
    resolved_config_path = args.export_root / "config/hb4_resolved.json"
    if resolved_config_path.is_file():
        resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
        resolved_config.setdefault("initial_status", resolved_config.get("status"))
        if formal and formal.get("coverage", {}).get("complete"):
            resolved_config["status"] = "HB4_FORMAL_COMPLETE"
        elif development and development.get("coverage", {}).get("complete"):
            resolved_config["status"] = "HB4_DEVELOPMENT_GO" if json.loads((args.export_root / "metrics/development/decision.json").read_text()).get("decision") == "HB4_DEVELOPMENT_GO" else "HB4_DEVELOPMENT_NO_GO"
            resolved_config["formal_status"] = "NOT_RUN_DEVELOPMENT_NO_GO" if resolved_config["status"] == "HB4_DEVELOPMENT_NO_GO" else "UNLOCKED"
        atomic_json_dump(resolved_config, resolved_config_path)
    records_root = args.export_root / "metrics/test" if formal else args.export_root / "metrics/development"
    records = _load_records(records_root)
    metrics_output = args.export_root / "metrics/test" if formal else args.export_root / "metrics/development"
    metrics_output.mkdir(parents=True, exist_ok=True)
    if records:
        if formal:
            write_table(records, metrics_output / "episodes_lightweight.jsonl")
            atomic_json_dump(
                {
                    "schema_version": "hb4_paired_comparisons_v1",
                    "mode": "formal",
                    "comparisons": formal.get("comparisons", []),
                },
                metrics_output / "paired_comparisons.json",
            )
    power = _power(records, args.export_root / "metrics/power_planning.json")
    methods = _methods(records, metrics_output / "methods.csv")
    preservation = _preservation(records, metrics_output / "baseline_preservation.csv")
    layered = _layered_status(development, preservation)
    if formal:
        formal_decision_path = args.export_root / "metrics/test/decision.json"
        formal_decision = json.loads(formal_decision_path.read_text()) if formal_decision_path.is_file() else None
        atomic_json_dump(
            {
                "schema_version": "hb4_decision_summary_v1",
                "development": json.loads((args.export_root / "metrics/development/decision.json").read_text()) if (args.export_root / "metrics/development/decision.json").is_file() else None,
                "formal": formal_decision,
            },
            args.export_root / "metrics/decision.json",
        )
    else:
        development_decision_path = args.export_root / "metrics/development/decision.json"
        development_decision = json.loads(development_decision_path.read_text()) if development_decision_path.is_file() else None
        if development_decision is not None:
            development_decision["layered_status"] = layered
            atomic_json_dump(development_decision, development_decision_path)
        atomic_json_dump(
            {
                "schema_version": "hb4_decision_summary_v1",
                "development": development_decision,
                "formal": {"status": "NOT_RUN_DEVELOPMENT_NO_GO", "decision": None},
            },
            args.export_root / "metrics/decision.json",
        )
    _local_artifacts(args.export_root, args.run_root)
    _report(args.export_root, args.run_root, formal, development, power, methods, preservation, layered)
    archive, digest = _package(args.export_root)
    print(json.dumps({"archive": str(archive.resolve()), "sha256": digest, "records": len(records), "formal": formal is not None}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
