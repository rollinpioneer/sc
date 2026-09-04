"""Finalize the bounded repair without ever consulting blind data on failure."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--v1-decision",type=Path,required=True); p.add_argument("--development",type=Path,required=True); p.add_argument("--validation-only",action="store_true"); p.add_argument("--blind"); p.add_argument("--blind-teacher-forced"); p.add_argument("--benchmark-check"); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    v1=json.loads(a.v1_decision.read_text()); dev=json.loads(a.development.read_text()); passed=bool(dev.get("development_gate_pass",False)); blind_generated=not a.validation_only and bool(a.blind)
    decision="GO_STAGE4_POLICY_VALIDATION" if passed and blind_generated else "SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR"
    result={"schema":"stage3f_r_final_decision_v1","decision":decision,"stage3_v1_decision_preserved":v1.get("decision")=="SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR","aggregation_repair_attempted":True,"blind_generated":blind_generated,"development_gate_pass":passed,"failed_rules":(["development_aggregation_gate"] if not passed else []),"selected_method":dev.get("selected_method"),"selected_source":dev.get("selected_source"),"selected_threshold":dev.get("selected_threshold")}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)); print(json.dumps(result,indent=2,sort_keys=True))
if __name__ == "__main__": main()
