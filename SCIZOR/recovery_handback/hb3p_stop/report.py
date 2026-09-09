"""Write the bounded HB3-P learned stop/continue pilot report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _num(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def write_report(config_path: Path, protocol_path: Path, dataset_summary_path: Path, training_summary_path: Path,
                 coverage_path: Path, evaluation_summary_path: Path, decision_path: Path, output: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    dataset = json.loads(Path(dataset_summary_path).read_text(encoding="utf-8"))
    training = json.loads(Path(training_summary_path).read_text(encoding="utf-8"))
    coverage = json.loads(Path(coverage_path).read_text(encoding="utf-8"))
    summary = json.loads(Path(evaluation_summary_path).read_text(encoding="utf-8"))
    decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
    method_by_id = {row["method_id"]: row for row in summary["methods"]}
    lines = [
        "# HB3-P Stop/Continue Pilot",
        "",
        "This report covers one frozen fixed-entry pilot. It is not a formal non-inferiority result.",
        "",
        "## Frozen Protocol",
        "",
        f"- Entry takeover: `t={protocol['anchor_t']}`.",
        f"- Selector query: exactly once at `t={protocol['decision_t']}` when the episode reaches it.",
        f"- STOP: helper interval `[20,80)`, permanent handback at `t=80`.",
        f"- CONTINUE: helper interval `[20,100)`, permanent handback at `t=100`.",
        f"- Methods: `{', '.join(protocol['methods'])}`.",
        f"- Declared absolute success-rate non-inferiority margin: `{config['noninferiority_margin_absolute']}`.",
        f"- Non-inferiority comparator: `{protocol['noninferiority_comparator']}`; short-exit comparator: `{protocol['short_exit_comparator']}`.",
        f"- Independent test roots: `{protocol['new_test_roots']}`; this run is explicitly `{ 'PILOT' if protocol['pilot'] else 'FORMAL' }`.",
        "",
        "## Label And Model",
        "",
        f"- Paired source roots: `{dataset['roots']}`; labels STOP/CONTINUE = `{dataset['stop_labels']}/{dataset['continue_labels']}`.",
        "- Labels were generated from realized paired utility of the existing FIXED_L60 and FIXED_L80 runs; no old H/M1 probability was thresholded.",
        f"- OOF threshold: `{training['threshold_selection']['threshold']:.8f}`; OOF accuracy: `{_pct(training['oof_label_accuracy'])}`.",
        "- Input: four frames of 9-D proprioception, four 7-D base-action suggestions, and current normalized time.",
        "- Excluded: reward, success, object state, privileged state, future result, images, root ID, seed, repair action.",
        "",
        "## Coverage And Engineering Checks",
        "",
        f"- Complete logical records: `{coverage['complete_records']}/{coverage['expected_records']}`.",
        f"- Unique rollouts including baseline: `{coverage['unique_rollouts']}`.",
        f"- Prefix checks: `{coverage['prefix_verified_roots']}/{coverage['expected_roots']}`.",
        f"- Learned-to-selected-fixed full trajectory parity: `{coverage['branch_parity_verified_roots']}/{coverage['expected_roots']}`.",
        f"- Missing records / engineering failures: `{len(coverage['missing_records'])}/{len(coverage['engineering_failures'])}`.",
        "- Required online invariants: one base-policy call per executed step, at most one takeover, no repair call after handback.",
        "",
        "## Results",
        "",
        "| Method | System success | Autonomous completion | Mean U | Mean helper steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in protocol["methods"]:
        row = method_by_id[method]
        suffix = ""
        if method == "LEARNED_STOP_CONTINUE":
            suffix = f"; STOP/CONTINUE={row.get('stop_decisions', 0)}/{row.get('continue_decisions', 0)}"
        lines.append(f"| {method} | {_pct(row['system_success_rate'])} ({row['system_success_count']}/{row['roots']}) | {_pct(row['autonomous_completion_rate'])} ({row['autonomous_completion_count']}/{row['roots']}) | {_num(row['mean_utility'], 6)} | {_num(row['mean_helper_steps'], 3)}{suffix} |")
    lines.extend(["", "## Paired Comparisons", ""])
    for key in ("LEARNED_STOP_CONTINUE_minus_FIXED_L60", "LEARNED_STOP_CONTINUE_minus_FIXED_L80", "LEARNED_STOP_CONTINUE_minus_NONE"):
        item = summary["comparisons"][key]
        lines.append(f"- `{key}` utility delta: `{_num(item['utility']['point_difference'], 6)}`, 95% root-bootstrap CI `[{_num(item['utility']['ci95_percentile'][0], 6)}, {_num(item['utility']['ci95_percentile'][1], 6)}]`.")
        lines.append(f"  System-success delta: `{_num(item['system_success']['point_difference'], 6)}`, CI `[{_num(item['system_success']['ci95_percentile'][0], 6)}, {_num(item['system_success']['ci95_percentile'][1], 6)}]`; autonomous delta: `{_num(item['autonomous_completion']['point_difference'], 6)}`, CI `[{_num(item['autonomous_completion']['ci95_percentile'][0], 6)}, {_num(item['autonomous_completion']['ci95_percentile'][1], 6)}]`.")
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Diagnostic system-success non-inferiority check versus {decision['noninferiority_comparator']}: `{decision['system_success_noninferiority_diagnostic']}`.",
        f"- Diagnostic autonomous-completion non-inferiority check versus {decision['noninferiority_comparator']}: `{decision['autonomous_noninferiority_diagnostic']}`.",
        "- These checks do not establish non-inferiority: the sample is 40 roots, while the pre-run estimate was roughly 354 roots for about 0.05 precision (worst-case binary proportion approximately 384 roots).",
        "- The result should be used to decide whether a larger preregistered run is warranted, not as a deployment guarantee.",
        "",
        "## Provenance",
        "",
        f"- Source probe protocol SHA256: `{protocol['source_probe_protocol_sha256']}`.",
        f"- Frozen execution code commit: `{protocol['code_commit']}`.",
        f"- Selector checkpoint SHA256: `{protocol['model']['checkpoint_sha256']}`.",
        f"- Frozen protocol SHA256: `{Path(protocol_path).with_name('frozen.sha256').read_text().split()[0]}`.",
        "- Large trajectory, model, image, log, and IPC assets remain local and are excluded from the lightweight Git package.",
    ])
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output.resolve()), "status": decision["status"], "methods": len(protocol["methods"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset-summary", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--evaluation-summary", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_report(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
