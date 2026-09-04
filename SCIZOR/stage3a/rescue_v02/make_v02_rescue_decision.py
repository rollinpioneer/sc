from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--proposer-transfer", type=Path); a = p.parse_args(); root = a.root
    pilot_audit = read(root / "pilot/benchmark_v0.2_pilot_audit.json"); pilot_oracle = read(root / "metrics/paired_clean_pilot_summary.json")
    full_audit_path, ceiling_path = root / "metrics/benchmark_v0.2_train_val_audit.json", root / "metrics/oracle_ceiling_v0.2_validation.json"
    if not (root / "pilot/benchmark_v0.2_pilot.hdf5").is_file():
        raise RuntimeError("v0.2 fallback was not executed; STOP is not a valid final decision")
    if not full_audit_path.is_file() or not ceiling_path.is_file():
        raise RuntimeError("full v0.2 audit and oracle ceiling are required before a final decision")
    full_audit, ceiling = read(full_audit_path), read(ceiling_path)
    proposer = read(a.proposer_transfer) if a.proposer_transfer and a.proposer_transfer.is_file() else {"evaluated": False, "full_top5_recall": None, "action_only_top5_recall": None}
    pilot_eng = pilot_oracle.get("engineering", {})
    pilot_pass = bool(pilot_audit.get("audit_pass") and all(pilot_eng.get(k) == 1.0 for k in ("branch_pre_state_equal_rate", "reference_exact_all_horizons_rate", "paired_clean_exact_all_horizons_rate", "finite_target_rate")))
    full_engineering = bool(full_audit.get("audit_pass") and ceiling.get("engineering_pass")); ceiling_pass = bool(ceiling.get("oracle_ceiling_pass"))
    failed = []
    if not pilot_pass: failed.append("pilot_engineering")
    if not full_audit.get("audit_pass"): failed.append("full_benchmark_audit")
    if not ceiling.get("engineering_pass"): failed.append("full_oracle_engineering")
    if full_engineering and not ceiling.get("method_pass"): failed.append("oracle_method_ceiling")
    decision = "STOP_STAGE3A_V02_ENGINEERING_FAILURE" if not full_engineering else ("STOP_STAGE3A_V02_ORACLE_CEILING_FAILURE" if not ceiling_pass else "REGENERATE_AND_RESUME_STAGE3A_ON_V02")
    pilot_auc = pilot_oracle.get("metrics", {}).get("overall_auroc")
    pmetrics, fmetrics, bmetrics = ceiling["paired_clean"]["metrics"], ceiling["primary_feasible"]["metrics"], ceiling["best_of_4_feasible"]["metrics"]
    result = {"decision": decision, "v01_status": "FAILED_AND_FROZEN", "v02_generation_attempted": True, "runtime_fingerprint_sha256": sha256(root / "config/runtime_fingerprint.json"), "pilot": {"num_pairs": pilot_audit.get("pair_count", 0), "determinism_pass": pilot_pass, "paired_clean_auroc": pilot_auc}, "full_v02": {"num_pairs": full_audit.get("pair_count", 0), "engineering_pass": full_engineering}, "oracle_ceiling": {"paired_clean_auroc": pmetrics.get("auroc"), "primary_feasible_auroc": fmetrics.get("auroc"), "best_of_4_auroc": bmetrics.get("auroc"), "paired_clean_auprc": pmetrics.get("auprc"), "primary_feasible_auprc": fmetrics.get("auprc")}, "proposer_transfer": proposer, "continue_stage3a_e": decision == "REGENERATE_AND_RESUME_STAGE3A_ON_V02", "failed_rules": failed}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
