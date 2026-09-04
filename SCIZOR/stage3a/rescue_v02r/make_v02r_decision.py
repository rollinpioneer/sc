from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--confirmation-metrics",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();m=json.loads(a.confirmation_metrics.read_text());pa=m["paired_clean"]["metrics"]["auroc"];fa=m["primary_feasible"]["metrics"]["auroc"];eng=bool(m["engineering_pass"]);method=pa is not None and fa is not None and pa>=.7 and fa>=.7
 if not eng: decision,failed="STOP_STAGE3A_V02R_ENGINEERING_FAILURE",["confirmation_engineering_gate"]
 elif not method: decision,failed="STOP_STAGE3A_AFTER_V02R_CONFIRMATION",([x for x,v in (("paired_clean_confirmation_auroc",pa),("primary_feasible_confirmation_auroc",fa)) if v is None or v<.7])
 else:decision,failed="RESUME_STAGE3A_E_ON_V02R",[]
 result={"decision":decision,"engineering_pass":eng,"method_pass":method,"paired_clean_confirmation_auroc":pa,"primary_feasible_confirmation_auroc":fa,"thresholds":{"paired_clean_auroc":.7,"primary_feasible_auroc":.7},"failed_rules":failed,"old_v02_stop_status":"SUPERSEDED_BY_V02R_CONFIRMATION"};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__":main()
