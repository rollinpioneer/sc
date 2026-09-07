"""Audit HB1-A through HB1-G deliverables against the frozen protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_handback.common import atomic_json_dump, read_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    config = json.loads((root / "config/hb1.json").read_text(encoding="utf-8"))
    checks = []

    def add(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    for relative in (
        "config/hb1.json", "config/runtime.txt", "config/source_commit.txt",
        "assets/assets.json", "metrics/metric_fixture.json",
    ):
        add(f"exists:{relative}", (root / relative).is_file(), relative)
    fixture_path = root / "metrics/metric_fixture.json"
    if fixture_path.is_file():
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        add("metric_fixture_passed", fixture.get("passed") is True, fixture.get("passed"))

    qualified_tasks = []
    for task in config["tasks"]:
        base_path = root / f"base/{task}/base_policy.json"
        pair_path = root / f"assets/policy_pair_{task}.json"
        validation = root / f"base/{task}/validation_fixed"
        add(f"{task}:base_policy", base_path.is_file(), str(base_path))
        summaries = sorted(validation.glob("*/summary.json")) if validation.is_dir() else []
        add(f"{task}:base_candidates_evaluated", len(summaries) >= 1, len(summaries))
        if summaries:
            counts = [json.loads(path.read_text(encoding="utf-8")).get("roots") for path in summaries]
            add(f"{task}:base_val_20_each", all(value == 20 for value in counts), counts)
        add(f"{task}:policy_pair", pair_path.is_file(), str(pair_path))
        if not pair_path.is_file():
            continue
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        pair_status = pair.get("status") or pair.get("qualification", {}).get("status")
        if pair_status == "NEED_BASE_POLICY":
            add(f"{task}:capability_stop", pair.get("repair") is None, pair_status)
            continue

        train_dir = root / f"repair/{task}/train"
        for step in config["repair_training"]["checkpoint_steps"]:
            add(f"{task}:sac_{step}", (train_dir / f"sac_{step}.zip").is_file(), step)
            add(f"{task}:normalizer_{step}", (train_dir / f"vecnormalize_{step}.pkl").is_file(), step)
        training_summary_path = train_dir / "training_summary.json"
        add(f"{task}:training_summary", training_summary_path.is_file(), str(training_summary_path))
        if training_summary_path.is_file():
            training = json.loads(training_summary_path.read_text(encoding="utf-8"))
            add(
                f"{task}:fixed_training_budget",
                int(training.get("sac_transitions", -1)) == int(config["repair_training"]["total_steps"]),
                training.get("sac_transitions"),
            )
        qualification_path = root / f"repair/{task}/qualification/summary.json"
        add(f"{task}:qualification", qualification_path.is_file(), str(qualification_path))
        if not qualification_path.is_file():
            continue
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        status = qualification.get("status")
        add(
            f"{task}:qualification_terminal",
            status in {"QUALIFIED", "NEED_STRONGER_REPAIRER"},
            status,
        )
        if status != "QUALIFIED":
            continue
        qualified_tasks.append(task)
        minimal_path = root / f"metrics/{task}_minimal_checks.json"
        add(f"{task}:minimal_checks", minimal_path.is_file(), str(minimal_path))
        if minimal_path.is_file():
            minimal = json.loads(minimal_path.read_text(encoding="utf-8"))
            add(f"{task}:minimal_checks_passed", minimal.get("passed") is True, minimal.get("passed"))
        anchors_path = root / f"roots/{task}/probe/anchors.parquet"
        branch_path = root / f"branches/{task}/branch_results.parquet"
        anchors = read_table(anchors_path) if anchors_path.is_file() else []
        branches = read_table(branch_path) if branch_path.is_file() else []
        expected = len(anchors) * 5
        keys = {(row.get("anchor_id"), row.get("branch_name")) for row in branches}
        add(f"{task}:five_branches_per_anchor", len(branches) == expected and len(keys) == expected, {"expected": expected, "actual": len(branches)})
        add(f"{task}:all_branches_engineering_ok", bool(branches) and all(bool(row.get("engineering_ok")) for row in branches), len(branches))
        archive_manifest = root / f"branches/{task}/trajectories/manifest.json"
        add(f"{task}:root_hdf5_archives", archive_manifest.is_file(), str(archive_manifest))

    for relative in (
        "metrics/coverage.json", "metrics/curves_by_task.csv",
        "metrics/curves_by_anchor_time.csv", "metrics/hb1_decision.json",
        "metrics/paired_results.parquet", "metrics/handoff_opportunities.parquet",
        "metrics/handoff_curve_dataset.parquet", "metrics/selected_examples.csv",
        "report/HB1_REPORT.md", "package/HB1_results_lightweight.zip",
        "package/HB1_results_lightweight.zip.sha256",
    ):
        add(f"exists:{relative}", (root / relative).is_file(), relative)
    figures = sorted((root / "figures").glob("*.png"))
    add("four_figures", len(figures) == 4, [path.name for path in figures])
    selected_path = root / "metrics/selected_examples.csv"
    if selected_path.is_file():
        selected = read_table(selected_path) if selected_path.suffix == ".parquet" else selected_path.read_text(encoding="utf-8").splitlines()[1:]
        add("at_most_eight_cases", len(selected) <= int(config["outputs"]["max_case_videos"]), len(selected))

    coverage_path = root / "metrics/coverage.json"
    if coverage_path.is_file():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        expected_complete = bool(qualified_tasks)
        complete = coverage.get("coverage_fraction") == 1.0
        add("qualified_branch_coverage_complete", complete if expected_complete else True, coverage.get("coverage_fraction"))

    payload = {
        "schema_version": "hb1_completion_audit_v1",
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "qualified_tasks": qualified_tasks,
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps({"passed": payload["passed"], "checks": len(checks), "qualified_tasks": qualified_tasks}, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
