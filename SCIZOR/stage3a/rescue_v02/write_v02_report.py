from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--decision", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args(); root = a.root
    decision = read(a.decision); pilot = read(root / "metrics/paired_clean_pilot_summary.json"); audit = read(root / "metrics/benchmark_v0.2_train_val_audit.json"); ceiling = read(root / "metrics/oracle_ceiling_v0.2_validation.json")
    pm, fm = ceiling["paired_clean"]["metrics"], ceiling["primary_feasible"]["metrics"]
    lines = ["# Stage 3A v0.2 replay-locked rescue report", "", "## Decision", "", f"`{decision['decision']}`.  Stage 3A-E continuation is `{decision['continue_stage3a_e']}`.", "", "## Evidence", "", "1. The previous stop remains scoped to v0.1 recovery: its legacy runtime is unavailable; it does not invalidate the independently replay-locked v0.2 path.", f"2. The current mimicgen runtime is frozen by `{decision['runtime_fingerprint_sha256']}`.", f"3. Pilot selection contains 10 clean-success base demos and produced {decision['pilot']['num_pairs']} pairs.", f"4. Pilot paired-clean AUROC: {decision['pilot']['paired_clean_auroc']}; pilot exact branch/reference/clean rates: {pilot['engineering']}", f"5. Full v0.2 contains {audit['pair_count']} train/validation perturbed pairs; split counts: {audit.get('split_counts', {})}.", f"6. Full paired-clean validation AUROC/AUPRC: {pm['auroc']} / {pm['auprc']}.", f"7. Primary feasible validation AUROC/AUPRC: {fm['auroc']} / {fm['auprc']}.", f"8. Best-of-4 feasible validation AUROC: {ceiling['best_of_4_feasible']['metrics']['auroc']} (upper-bound diagnostic only).", f"9. Proposer transfer evaluated: {decision['proposer_transfer']['evaluated']}; full/action-only Top-5 recall: {decision['proposer_transfer']['full_top5_recall']} / {decision['proposer_transfer']['action_only_top5_recall']}.", f"10. Engineering pass: {ceiling['engineering_pass']}; method pass: {ceiling['method_pass']}; failed rules: {decision['failed_rules']}.", "", "## Frozen gates", "", "- Validation paired-clean AUROC >= 0.70", "- Validation primary feasible replacement AUROC >= 0.70", "- Exact twin-prefix/replay requirements recorded in `metrics/oracle_ceiling_v0.2_validation.json`", ""]
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text("\n".join(lines), encoding="utf-8")
    print(str(a.output))


if __name__ == "__main__": main()
