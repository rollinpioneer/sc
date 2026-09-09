"""Write the bounded fixed-entry exit probe report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    summary = json.loads((root / "metrics/exit_probe/summary.json").read_text(encoding="utf-8"))
    decision = json.loads((root / "metrics/exit_probe/decision.json").read_text(encoding="utf-8"))
    lines = [
        "# HB3-P Intermediate Exit Probe",
        "",
        "This is one bounded, fixed-entry probe after the HB3-P exit diagnosis. It does not train a stopping model.",
        "",
        "## Frozen Scope",
        "",
        f"- Entry: t={protocol['candidate_times'][0]}.",
        "- Methods: NONE, FIXED_L40, FIXED_L60, FIXED_L80.",
        f"- Independent roots: {protocol['new_test_roots']} (seeds {protocol['test_seed_start']}-{protocol['test_seeds'][-1]}).",
        "- Maximum complete method records: 160; maximum environment steps: 64,000.",
        "- New roots were sampled before labels and were not selected from HB3-P test roots.",
        "",
        "## Results",
        "",
        "| Method | System success | Autonomous completion | Mean utility | Mean helper steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["methods"]:
        lines.append(
            f"| {row['method_id']} | {row['system_success_count']}/{row['roots']} ({row['system_success_rate']:.4f}) | "
            f"{row['autonomous_completion_count']}/{row['roots']} ({row['autonomous_completion_rate']:.4f}) | "
            f"{row['mean_utility']:.6f} | {row['mean_helper_steps_actual']:.3f} |"
        )
    lines.extend([
        "",
        "## Stop Versus Continue",
        "",
        f"- Prefix verification: {decision['prefix_verified_roots']}/{protocol['new_test_roots']}.",
        f"- Short-exit autonomous successes: `{decision['short_exit_autonomous_success_counts']}`.",
        f"- Short-failure / L80-success cases: `{decision['continue_to_l80_autonomous_success_cases']}`.",
        f"- Retrospective fixed-grid oracle mean utility delta versus L80: `{decision['oracle_mean_delta_vs_fixed_l80']:.6f}`.",
        "- The oracle is retrospective and is not a deployable selector or a formal success-rate upper bound.",
        "",
        "## Decision",
        "",
        f"- **{decision['decision']}**.",
        f"- Training allowed by this gate: `{decision['training_allowed']}`.",
        "- Training performed: `False`.",
        "- Any model training would require a separate frozen protocol and is outside this probe.",
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "decision": decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
