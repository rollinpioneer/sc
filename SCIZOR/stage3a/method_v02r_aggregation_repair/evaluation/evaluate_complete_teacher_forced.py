"""Evaluate the complete 256-pair teacher-forced verifier diagnostic."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from stage3a.method_v02r.evaluation.metrics import binary

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--ensemble",type=Path,required=True); p.add_argument("--action-only",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--csv",type=Path,required=True); a=p.parse_args()
    full=pd.read_parquet(a.ensemble); action=pd.read_parquet(a.action_only)[["replacement_id","pred_score","pred_positive_probability"]].rename(columns={"pred_score":"action_pred_score","pred_positive_probability":"action_pred_positive"}); f=full.merge(action,on="replacement_id",validate="one_to_one")
    if f.replacement_id.duplicated().any(): raise RuntimeError("duplicate teacher-forced replacement IDs")
    y=f.is_effective_intervention.astype(int); result={"schema":"stage3f_r_complete_teacher_metrics_v1","pair_count":int(f.pair_id.nunique()),"effective_pair_count":int(f.drop_duplicates("pair_id").is_effective_intervention.sum()),"full":{"auroc":binary(y,f.pred_positive_mean)["auroc"],"auprc":binary(y,f.pred_positive_mean)["auprc"],"score_mae":float(np.abs(f.pred_score_mean-f.counterfactual_improvement_long).mean()),"score_spearman":float(f.pred_score_mean.corr(f.counterfactual_improvement_long,method="spearman"))},"action_only":binary(y,f.action_pred_positive),"by_task":{}}
    for task,g in f.groupby("task"): result["by_task"][str(task)]={"full":binary(g.is_effective_intervention.astype(int),g.pred_positive_mean),"action_only":binary(g.is_effective_intervention.astype(int),g.action_pred_positive)}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)); pd.DataFrame([{**{"task":"all","method":"full"},**result["full"]},{**{"task":"all","method":"action_only"},**result["action_only"]}]).to_csv(a.csv,index=False); print(json.dumps(result,indent=2,sort_keys=True))
if __name__ == "__main__": main()
