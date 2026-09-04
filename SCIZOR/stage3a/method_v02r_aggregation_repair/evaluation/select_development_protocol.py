"""Select one fixed aggregation protocol using the development split only."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, pandas as pd
from stage3a.method_v02r.evaluation.metrics import binary, threshold_metrics

def _threshold(pair: pd.DataFrame) -> dict:
    y=pair.is_effective_intervention.astype(bool).to_numpy(); s=pair.pair_score.astype(float).to_numpy(); candidates=[]
    for t in np.unique(s):
        m=threshold_metrics(y,s,float(t))
        if (m["no_effect_far"] or 0)<=.20: candidates.append({**m,"localized_recall":float(((pair.loc[y,"predicted_t"]-pair.loc[y,"intervention_t"]).abs()<=1).mean()) if y.any() else 0.0})
    if not candidates: return {"threshold":None,"no_effect_far":None,"effective_recall":None,"localized_recall":None,"predicted_positive_count":0}
    return sorted(candidates,key=lambda x:(x["effective_recall"],x["localized_recall"],x["threshold"]),reverse=True)[0]

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--pair-scores",type=Path,required=True); p.add_argument("--transition-scores",type=Path,required=True); p.add_argument("--teacher-forced-complete",type=Path,required=True); p.add_argument("--verifier-learning",type=Path,required=True); p.add_argument("--proposer-transfer",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--output-metrics",type=Path,required=True); p.add_argument("--output-protocol",type=Path,required=True); p.add_argument("--output-csv",type=Path,required=True); a=p.parse_args()
    pairs=pd.read_parquet(a.pair_scores); trans=pd.read_parquet(a.transition_scores); cfg=json.loads(a.config.read_text()); tf=json.loads(a.teacher_forced_complete.read_text()); vl=json.loads(a.verifier_learning.read_text()); pt=json.loads(a.proposer_transfer.read_text()).get("validation",{}); rows=[]
    for method,g in pairs.groupby("method_id",sort=True):
        source=str(g.source.iloc[0]); bm=binary(g.is_effective_intervention.astype(int),g.pair_score); operating=_threshold(g); ts=trans[trans.method_id.eq(method)]
        transfer=pt.get(source,{}).get("overall",{})
        task_metrics={}
        for task,tg in g.groupby("task"): task_metrics[str(task)]=binary(tg.is_effective_intervention.astype(int),tg.pair_score)
        pred=g.pair_score.astype(float)>=float(operating["threshold"] if operating["threshold"] is not None else np.inf); no=~g.is_effective_intervention.astype(bool)
        region=float((((g.loc[g.is_effective_intervention,"predicted_t"]>=g.loc[g.is_effective_intervention,"responsible_start"]) & (g.loc[g.is_effective_intervention,"predicted_t"]<=g.loc[g.is_effective_intervention,"responsible_end"]))).mean()) if g.is_effective_intervention.any() else 0.0
        rows.append({"method_id":method,"source":source,"pair_auroc":bm["auroc"],"pair_auprc":bm["auprc"],"prevalence":bm["prevalence"],"no_effect_far":operating["no_effect_far"],"effective_recall":operating["effective_recall"],"top1_within_1":operating["localized_recall"],"region_hit":region,"mean_abs_delay":float((g.loc[g.is_effective_intervention,"predicted_t"]-g.loc[g.is_effective_intervention,"intervention_t"]).abs().mean()) if g.is_effective_intervention.any() else None,"recovery_false_attribution":int((pred & no & g.failure_type.astype(str).str.contains("recovery")).sum()),"can_auroc":task_metrics.get("can",{}).get("auroc"),"square_auroc":task_metrics.get("square",{}).get("auroc"),"mean_candidates":float(g.candidate_count.mean()) if len(g) else 0.0,"missing_candidate_pairs":int((~g.has_valid_candidate).sum()),"proposal_region_recall":float(transfer.get("responsibility_region_recall",0.0)),"threshold":operating["threshold"],"task_metrics":task_metrics})
    eligible=[]; gate_rules=cfg.get("development_gate",{}); candidate_full=float(vl.get("candidate_replacement",{}).get("full",{}).get("auroc") or 0.0); teacher_full=float(tf.get("full",{}).get("auroc") or 0.0)
    for r in rows:
        checks={"proposal_region_recall":r["proposal_region_recall"]>=gate_rules.get("min_proposal_region_recall",.5),"pair_auroc":(r["pair_auroc"] or 0)>=gate_rules.get("min_pair_auroc",.7),"threshold":r["threshold"] is not None,"effective_recall":(r["effective_recall"] or 0)>=gate_rules.get("min_effective_recall",.4),"can_auroc":(r["can_auroc"] or 0)>=gate_rules.get("min_task_pair_auroc",.6),"square_auroc":(r["square_auroc"] or 0)>=gate_rules.get("min_task_pair_auroc",.6)}
        # Raw proposer rows are diagnostics only; the protocol can select
        # exclusively from the five fixed repaired methods.
        if method not in {"action_cf_only", "action_current_fused", "action_defect_gated", "action_defect_contrast", "union_defect_contrast"}:
            checks["eligible_method"] = False
        r["eligible"] = all(checks.values()); r["failed_rules"]=[k for k,v in checks.items() if not v]; eligible.append(r) if r["eligible"] else None
    selected=None
    if eligible:
        best=max(r["pair_auprc"] or -1 for r in eligible); pool=[r for r in eligible if best-(r["pair_auprc"] or -1)<=.01]; low=min(r["no_effect_far"] for r in pool); pool=[r for r in pool if r["no_effect_far"]-low<=.02]; selected=sorted(pool,key=lambda r:(-(r["effective_recall"] or 0),-(r["top1_within_1"] or 0),r["mean_candidates"],r["method_id"]))[0]
    global_failed=[]
    if candidate_full<.70: global_failed.append("candidate_replacement_full_auroc")
    if teacher_full<.70: global_failed.append("complete_teacher_forced_full_auroc")
    if global_failed: selected=None
    protocol={"schema":"stage3f_r_development_protocol_v1","development_gate_pass":selected is not None,"decision":"RESUME_STAGE3_BLIND_AFTER_AGGREGATION_REPAIR" if selected else "SWITCH_TO_COVERAGE_CONSTRAINED_SOFT_SCIZOR","selected_method":selected["method_id"] if selected else None,"selected_source":selected["source"] if selected else None,"selected_threshold":selected["threshold"] if selected else None,"selected_formula":("fixed aggregation matrix method "+selected["method_id"]) if selected else None,"background_path":"background/train_background.json","failed_rules":global_failed or (["no_valid_fixed_aggregation_method"] if not eligible else []),"candidate_replacement_full_auroc":candidate_full,"complete_teacher_forced_full_auroc":teacher_full}
    metrics={"schema":"stage3f_r_development_metrics_v1","methods":rows,"eligible_methods":[r["method_id"] for r in eligible],"selected":selected["method_id"] if selected else None,"global_failed_rules":global_failed}
    a.output_metrics.parent.mkdir(parents=True,exist_ok=True); a.output_metrics.write_text(json.dumps(metrics,indent=2,default=str)); a.output_protocol.write_text(json.dumps(protocol,indent=2,default=str)); pd.DataFrame([{k:v for k,v in r.items() if k!="task_metrics"} for r in rows]).to_csv(a.output_csv,index=False); print(json.dumps(protocol,indent=2))
if __name__ == "__main__": main()
