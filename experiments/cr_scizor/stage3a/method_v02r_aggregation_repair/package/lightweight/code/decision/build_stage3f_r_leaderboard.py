"""Combine preserved v1 rows and fixed development-method rows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from stage3a.method_v02r.evaluation.metrics import binary

COLS=["split","method_id","source","pair_auroc","pair_auprc","prevalence","no_effect_far","effective_recall","top1_within_1","region_hit","mean_abs_delay","recovery_false_attribution","can_auroc","square_auroc","mean_candidates","selected"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--development",type=Path,required=True); p.add_argument("--v1-pairs",type=Path,required=True); p.add_argument("--v1-metrics",type=Path,required=True); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    dev=json.loads(a.development.read_text()); protocol=json.loads(a.protocol.read_text()); rows=[]
    for r in dev["methods"]:
        rows.append({"split":"development", "method_id":r["method_id"], "source":r["source"], **{k:r.get(k) for k in COLS[3:-1]}, "selected":r["method_id"]==protocol.get("selected_method")})
    old=pd.read_parquet(a.v1_pairs); vm=json.loads(a.v1_metrics.read_text());
    for source,g in old.groupby("proposer_source"):
        for name,col in (("stage3_v1_action_fused","fused_pair_score"),("stage3_v1_counterfactual_only","counterfactual_only_pair_score"),("stage3_v1_raw","raw_proposer_pair_score")):
            b=binary(g.is_effective_intervention.astype(int),g[col]); task={t:binary(x.is_effective_intervention.astype(int),x[col]) for t,x in g.groupby("task")}; rows.append({"split":"development","method_id":name,"source":source,"pair_auroc":b["auroc"],"pair_auprc":b["auprc"],"prevalence":b["prevalence"],"no_effect_far":None,"effective_recall":None,"top1_within_1":None,"region_hit":None,"mean_abs_delay":None,"recovery_false_attribution":None,"can_auroc":task.get("can",{}).get("auroc"),"square_auroc":task.get("square",{}).get("auroc"),"mean_candidates":float(g.mean_candidates_per_pair.mean()) if "mean_candidates_per_pair" in g else float(g.mean_candidates_per_pair.mean()) if False else None,"selected":False})
    out=pd.DataFrame(rows)
    for c in COLS:
        if c not in out: out[c]=None
    a.output.parent.mkdir(parents=True,exist_ok=True); out[COLS].to_csv(a.output,index=False)

if __name__=="__main__": main()
