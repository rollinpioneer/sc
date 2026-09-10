"""Build HB4 lightweight metrics, report, and reproducible result package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import zipfile
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


def _report(export_root: Path, run_root: Path, formal_summary: dict | None, development_summary: dict | None, power: dict, methods: list[dict], preservation: list[dict]) -> None:
    cfg = json.loads((export_root / "config/hb4_resolved.json").read_text(encoding="utf-8"))
    if formal_summary and formal_summary.get("coverage", {}).get("complete"):
        execution_status = "FORMAL_COMPLETE"
    elif development_summary and development_summary.get("coverage", {}).get("complete"):
        execution_status = "DEVELOPMENT_COMPLETE_FORMAL_NOT_RUN"
    else:
        execution_status = "INCOMPLETE"
    lines = ["# HB4 Square 基础策略恢复经验吸收报告", "", "## Material Passport", "", f"- Execution status: `{execution_status}`", f"- Source commit: {cfg.get('source_commit', 'NA')}", f"- Scope: no-help Square evaluation after matched-budget offline finetuning", f"- Run root: `{run_root}`", "", "## 历史状态", "", "保留 HB3 `FORMAL_NONINFERIORITY_PASS` 的边界；本报告不将其改写为学习型退出的效用优越性。", "", "## 数据与训练", "", "- HB4-C matching audit: see `data/matching_audit.json`.", "- Training matrix: four arms x three seeds, 4000 optimizer updates, final step only used for evaluation.", "- Student input: two RGB cameras plus 9-D proprioception; helper policy and selector are excluded from the no-help evaluator.", "", "## 开发门槛"]
    if development_summary:
        gate = development_summary.get("gate", {})
        lines += ["", f"- Development coverage complete: `{development_summary.get('coverage', {}).get('complete')}`", f"- Development decision: `{json.loads((export_root / 'metrics/development/decision.json').read_text()).get('decision', 'NA')}`", f"- Gate details: `{json.dumps(gate, ensure_ascii=False, sort_keys=True)}`"]
    else:
        lines += ["", "- Development results: `NOT_AVAILABLE`"]
    lines += ["", "## 正式测试", ""]
    if formal_summary:
        cov = formal_summary.get("coverage", {})
        lines += [f"- Coverage: `{cov.get('records', 0)}` records, `{cov.get('expected_roots', 400)}` roots, engineering failures `{cov.get('engineering_failures', 'NA')}`.", f"- Main comparisons: `{json.dumps(formal_summary.get('comparisons', []), ensure_ascii=False)}`", f"- Formal decision: `{json.loads((export_root / 'metrics/test/decision.json').read_text()).get('decision', 'NA')}`"]
    else:
        lines += ["- Formal test: `NOT_RUN_DEVELOPMENT_NO_GO`; the development gate did not unlock formal evaluation, and no PASS is inferred from missing records."]
    lines += ["", "## 功效规划与能力保留", "", f"- Approximate fixed-n=400 MDE: `{power.get('approximate_mde_80pct', 'NA')}`; this is planning only.", f"- Methods summary: `{json.dumps(methods, ensure_ascii=False)}`", f"- Baseline preservation rows: `{len(preservation)}`", "", "## 局限", "", "单任务 Square、同分布新根、固定基础 checkpoint/教师、条件化恢复源选择、三个训练种子；本轮没有在线帮助需求结论。", "", "## 本地大资产", "", "完整 checkpoint、HDF5、图像缓存和运行日志保留在 `RUN_ROOT`，不进入轻量仓库包；其路径与哈希见 `report/LOCAL_ONLY_ARTIFACTS.md`。"]
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
        atomic_json_dump(
            {
                "schema_version": "hb4_decision_summary_v1",
                "development": json.loads((args.export_root / "metrics/development/decision.json").read_text()) if (args.export_root / "metrics/development/decision.json").is_file() else None,
                "formal": {"status": "NOT_RUN_DEVELOPMENT_NO_GO", "decision": None},
            },
            args.export_root / "metrics/decision.json",
        )
    _local_artifacts(args.export_root, args.run_root)
    _report(args.export_root, args.run_root, formal, development, power, methods, preservation)
    archive, digest = _package(args.export_root)
    print(json.dumps({"archive": str(archive.resolve()), "sha256": digest, "records": len(records), "formal": formal is not None}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
