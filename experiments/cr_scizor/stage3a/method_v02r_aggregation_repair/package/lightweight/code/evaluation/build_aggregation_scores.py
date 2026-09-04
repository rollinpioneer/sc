"""Build the fixed raw/counterfactual/deficit aggregation matrix."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from .aggregation_utils import attach_local_deficit, build_pair_universe, fill_missing_pairs, replacement_summary, source_columns
DIAGNOSTIC=["action_raw","union_raw","full_raw"]; ELIGIBLE=["action_cf_only","action_current_fused","action_defect_gated","action_defect_contrast","union_defect_contrast"]

def main() -> None:
    p=argparse.ArgumentParser()
    for name in ("ensemble","samples","proposals","labels","chunk-evidence","background","config"): p.add_argument("--"+name,dest=name.replace("-","_"),type=Path,required=True)
    p.add_argument("--split",required=True); p.add_argument("--only-method"); p.add_argument("--output-replacements",type=Path,required=True); p.add_argument("--output-transitions",type=Path,required=True); p.add_argument("--output-pairs",type=Path,required=True); p.add_argument("--summary",type=Path,required=True); a=p.parse_args()
    bg=json.loads(a.background.read_text()); ens=pd.read_parquet(a.ensemble); proposals=pd.read_parquet(a.proposals); labels=pd.read_parquet(a.labels); evidence=pd.read_parquet(a.chunk_evidence); ens=ens[(ens.split==a.split)&(~ens.is_teacher_forced.fillna(False))].copy()
    cols=["pair_id","t","task","demo_id","in_full_top5","in_action_top5","in_union_top5","full_rank","action_rank","union_rank","raw_full_score","raw_action_score","raw_union_score","proposal_rank_weight"]; prop=proposals[cols].drop_duplicates(["pair_id","t"]); repl=attach_local_deficit(ens.merge(prop,left_on=["pair_id","query_t"],right_on=["pair_id","t"],how="left",suffixes=("","_proposal")),evidence)
    requested=[a.only_method] if a.only_method else DIAGNOSTIC+ELIGIBLE; universe=build_pair_universe(labels,a.split); trs=[]; prs=[]
    for source in ("action_top5","union_top5","full_top5"):
        membership,raw_col,_=source_columns(source); cand=repl[repl[membership].fillna(False)]; local=[]
        for (pair_id,t),g in cand.groupby(["pair_id","query_t"],sort=False):
            first=g.iloc[0]; s=replacement_summary(g); task=str(first.task); b=bg.get("sources",{}).get(source,{}).get(task,{}); bmax=float(b.get("cf_max_q80",0)); bcon=float(b.get("cf_contrast_q80",0)); d=float(first.local_deficit); raw=float(first.get(raw_col,0) or 0); w=float(first.get("proposal_rank_weight",0) or 0)
            vals={"action_raw":raw,"union_raw":raw,"full_raw":raw,"action_cf_only":s["cf_max"],"action_current_fused":s["cf_max"]*w,"action_defect_gated":d*max(s["cf_max"]-bmax,0),"action_defect_contrast":d*max(s["cf_contrast"]-bcon,0),"union_defect_contrast":d*max(s["cf_contrast"]-bcon,0)}
            for method in requested:
                diagnostic_source = {"action_raw": "action_top5", "union_raw": "union_top5", "full_raw": "full_top5"}.get(method)
                if not ((diagnostic_source == source) or (method.startswith("action_") and source == "action_top5") or (method == "union_defect_contrast" and source == "union_top5")): continue
                row={"method_id":method,"source":source,"pair_id":pair_id,"task":task,"split":a.split,"query_t":int(t),"transition_score":float(vals[method]),**s,"background_max":bmax,"background_contrast":bcon,"local_deficit":d,"proposal_rank_weight":w,"raw_proposer_score":raw,"replacement_count":int(len(g)),"is_effective_intervention":bool(first.is_effective_intervention),"intervention_t":first.intervention_t,"responsible_start":first.responsible_start,"responsible_end":first.responsible_end,"is_recovery":bool(first.get("is_recovery",False))}; trs.append(row); local.append(row)
        td=pd.DataFrame(local)
        for method in requested:
            part=td[td.method_id.eq(method)] if len(td) else td
            for pair_id,g in part.groupby("pair_id",sort=False):
                scores=g.transition_score.astype(float).to_numpy(); best=int(np.argmax(scores)) if len(scores) else -1; first=g.iloc[best] if len(g) else universe[universe.pair_id.eq(pair_id)].iloc[0]; meta=universe[universe.pair_id.eq(pair_id)].iloc[0]; score=float(scores[best]) if len(scores) else 0.0
                if method.endswith("contrast") and len(scores): score=max(score-float(np.median(scores)),0)
                prs.append({"method_id":method,"source":source,"pair_id":pair_id,"task":meta.task,"split":a.split,"pair_score":score,"predicted_t":int(first.query_t) if len(g) else -1,"candidate_count":int(len(g)),"has_valid_candidate":bool(len(g)),"is_effective_intervention":bool(meta.is_effective_intervention),"intervention_t":meta.intervention_t,"responsible_start":meta.responsible_start,"responsible_end":meta.responsible_end,"failure_type":meta.failure_type})
    trans=pd.DataFrame(trs); pairs=pd.DataFrame(prs); complete=[]
    for method in requested:
        source="union_top5" if method=="union_defect_contrast" else ("action_top5" if method not in DIAGNOSTIC else {"action_raw":"action_top5","union_raw":"union_top5","full_raw":"full_top5"}[method]); complete.append(fill_missing_pairs(pairs[pairs.method_id.eq(method)] if len(pairs) else pairs,universe,method,source))
    pairs=pd.concat(complete,ignore_index=True) if complete else pd.DataFrame(); a.output_replacements.parent.mkdir(parents=True,exist_ok=True); repl.to_parquet(a.output_replacements,index=False); trans.to_parquet(a.output_transitions,index=False); pairs.to_parquet(a.output_pairs,index=False); summary={"split":a.split,"methods":requested,"replacement_rows":int(len(repl)),"transition_rows":int(len(trans)),"pair_rows":int(len(pairs)),"pair_counts":pairs.groupby("method_id").size().to_dict() if len(pairs) else {}}; a.summary.parent.mkdir(parents=True,exist_ok=True); a.summary.write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))

if __name__ == "__main__": main()
