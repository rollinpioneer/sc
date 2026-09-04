"""Estimate train-only no-effect replacement backgrounds."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from .aggregation_utils import attach_local_deficit, replacement_summary, source_columns

def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("ensemble", "samples", "proposals", "labels", "chunk-evidence", "config"):
        p.add_argument("--" + name, dest=name.replace("-", "_"), type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--transition-output", type=Path, required=True)
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); ens = pd.read_parquet(a.ensemble); proposals = pd.read_parquet(a.proposals); evidence = pd.read_parquet(a.chunk_evidence)
    ens = ens[(ens.split == "train") & (~ens.is_teacher_forced)].copy()
    cols = ["pair_id", "t", "task", "demo_id", "in_full_top5", "in_action_top5", "in_union_top5", "full_rank", "action_rank", "union_rank", "raw_full_score", "raw_action_score", "raw_union_score", "proposal_rank_weight"]
    prop = proposals[cols].drop_duplicates(["pair_id", "t"])
    frame = attach_local_deficit(ens.merge(prop, left_on=["pair_id", "query_t"], right_on=["pair_id", "t"], how="left", suffixes=("", "_proposal")), evidence)
    frame = frame[~frame.is_effective_intervention.fillna(False)].copy(); rows=[]; out={"schema":"stage3f_r_train_background_v1", "quantile":float(cfg.get("replacement_background_quantile", .8)), "sources":{}, "local_deficit":{"formula":"clip(max covering V_c,0,1)", "coverage_rate":float(frame.deficit_coverage.mean()) if len(frame) else 0.0}}
    q=float(cfg.get("replacement_background_quantile", .8))
    for source in ("action_top5", "union_top5", "full_top5"):
        membership, raw_col, _ = source_columns(source); part=frame[frame[membership].fillna(False)]; stats=[]
        for _, group in part.groupby(["pair_id","query_t"], sort=False):
            s=replacement_summary(group); first=group.iloc[0]; stats.append({**s,"pair_id":first.pair_id,"task":first.task,"source":source,"query_t":int(first.query_t),"local_deficit":float(first.local_deficit),"raw_proposer_score":float(first.get(raw_col,0.0) or 0.0),"proposal_rank_weight":float(first.get("proposal_rank_weight",0.0) or 0.0)})
        trans=pd.DataFrame(stats); rows.extend(trans.to_dict("records") if len(trans) else []); out["sources"][source]={}
        for task in ("can","square"):
            tr=trans[trans.task.eq(task)] if len(trans) else trans; out["sources"][source][task]={"cf_max_q80":float(tr.cf_max.quantile(q)) if len(tr) else 0.0,"cf_contrast_q80":float(tr.cf_contrast.quantile(q)) if len(tr) else 0.0,"n":int(len(tr))}
    trans=pd.DataFrame(rows); a.transition_output.parent.mkdir(parents=True, exist_ok=True); trans.to_parquet(a.transition_output,index=False); a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)); print(json.dumps(out,indent=2,sort_keys=True))

if __name__ == "__main__": main()
