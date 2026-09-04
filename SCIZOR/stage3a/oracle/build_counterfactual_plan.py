"""Turn proposer transitions into simulator counterfactual queries."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from stage3a.library.query_action_library import load_library, retrieve_replacements

EFFECTIVE = {"direct_failure", "delayed_failure", "recovery_failure", "recovery_success"}
KEYS = ["pair_id", "demo_id", "task", "base_demo_id", "split", "t"]

def parse_args():
    p=argparse.ArgumentParser()
    for n in ("proposal-candidates","pair-metadata","transition-labels","feature-index","action-library-dir","config","output","summary"): p.add_argument("--"+n, type=Path, required=True)
    p.add_argument("--split", required=True); return p.parse_args()

def main():
    a=parse_args(); cfg=json.loads(a.config.read_text()); proposals=pd.read_parquet(a.proposal_candidates); proposals=proposals.drop_duplicates(KEYS,keep="first"); proposals=proposals[proposals.split.eq(a.split)].copy()
    labels=pd.read_parquet(a.transition_labels); labels=labels[(labels.variant=="perturbed")&(labels.split==a.split)].copy(); labels=labels.sort_values(KEYS).drop_duplicates(KEYS,keep="first")
    fi=pd.read_parquet(a.feature_index).set_index("demo_id"); metadata={}
    with a.pair_metadata.open() as f:
        for line in f:
            if line.strip():
                m=json.loads(line); metadata[m["pair_id"]]=m
    lib=load_library(a.action_library_dir); queries={}
    def add_query(row, source):
        key=(str(row.pair_id),int(row.t)); q=queries.get(key)
        if q is None:
            q={"pair_id":str(row.pair_id),"demo_id":str(row.demo_id),"task":str(row.task),"base_demo_id":str(row.base_demo_id),"split":str(row.split),"query_t":int(row.t),"episode_length":int(row.episode_length),"query_source":source,"in_full_top5":bool(row.in_full_top5),"in_action_top5":bool(row.in_action_top5),"in_union_top5":bool(row.in_union_top5),"full_rank":None if pd.isna(row.full_rank) else int(row.full_rank),"action_rank":None if pd.isna(row.action_rank) else int(row.action_rank),"proposal_rank_weight":float(row.proposal_rank_weight),"replacement_rows":[]}; queries[key]=q
        elif source=="teacher_forced_intervention": q["query_source"]="proposal_and_teacher_forced_intervention"
    for row in proposals.itertuples(index=False): add_query(row,"proposal")
    allowed_no_effect=(labels.failure_type.eq("no_effect") & labels.label_status.ne("ambiguous"))
    teacher=labels[(labels.failure_type.isin(EFFECTIVE)|allowed_no_effect)&labels.intervention_t.notna()].drop_duplicates("pair_id")
    for row in teacher.itertuples(index=False):
        # use the proposer row schema when available; otherwise recover metadata
        cand=proposals[proposals.pair_id.eq(row.pair_id)]
        if len(cand):
            base=cand.iloc[0].copy(); base["t"]=int(row.intervention_t); base["in_full_top5"]=bool(((cand.t==int(row.intervention_t)) & cand.in_full_top5).any()); base["in_action_top5"]=bool(((cand.t==int(row.intervention_t)) & cand.in_action_top5).any()); base["in_union_top5"]=bool(((cand.t==int(row.intervention_t)) & cand.in_union_top5).any()); base["full_rank"]=None; base["action_rank"]=None; base["proposal_rank_weight"]=0.0; add_query(base,"teacher_forced_intervention")
        else: continue
    out=[]; ood=0
    for q in queries.values():
        fr=fi.loc[q["demo_id"]]; feat=np.load(fr.feature_path); states=feat["states"].astype(np.float32); actions=feat["actions"].astype(np.float32); t=q["query_t"]; meta=metadata.get(q["pair_id"],{}); repl=retrieve_replacements(q["task"],states[t],actions[t],q["base_demo_id"],float(t/max(q["episode_length"]-1,1)),float(actions[max(0,t-1),-1]),lib,cfg)
        for n,c in enumerate(repl):
            q["replacement_rows"].append(c); out.append({"query_id":f'{q["pair_id"]}|t{t}',"replacement_id":f'{q["pair_id"]}|t{t}|r{n}|{c["replacement_source"]}',"split":q["split"],"pair_id":q["pair_id"],"demo_id":q["demo_id"],"task":q["task"],"base_demo_id":q["base_demo_id"],"hdf5_group":str(meta.get("hdf5_group",f'/data/{q["demo_id"]}')),"query_t":t,"episode_length":q["episode_length"],"query_source":q["query_source"],"in_full_top5":q["in_full_top5"],"in_action_top5":q["in_action_top5"],"in_union_top5":q["in_union_top5"],"full_rank":q["full_rank"],"action_rank":q["action_rank"],"proposal_rank_weight":q["proposal_rank_weight"],"current_action":actions[t].astype(float).tolist(),"replacement_action":c["replacement_action"],"replacement_source":c["replacement_source"],"library_demo_id":c["library_demo_id"],"library_base_demo_id":c["library_base_demo_id"],"library_t":c["library_t"],"state_distance":c["state_distance"],"action_delta_l2":c["action_delta_l2"],"state_in_domain":c["state_in_domain"],"action_in_domain":c["action_in_domain"],"oracle_only":False,"continuation_end_t":min(q["episode_length"],t+int(cfg["oracle"]["max_horizon"]))})
        if t==int(meta.get("intervention_t",-999)) and cfg["oracle"].get("include_paired_clean_upper_bound",True) and meta.get("original_action") is not None:
            n=len(repl); out.append({"query_id":f'{q["pair_id"]}|t{t}',"replacement_id":f'{q["pair_id"]}|t{t}|r{n}|paired_clean_upper_bound',"split":q["split"],"pair_id":q["pair_id"],"demo_id":q["demo_id"],"task":q["task"],"base_demo_id":q["base_demo_id"],"hdf5_group":str(meta.get("hdf5_group",f'/data/{q["demo_id"]}')),"query_t":t,"episode_length":q["episode_length"],"query_source":q["query_source"],"in_full_top5":q["in_full_top5"],"in_action_top5":q["in_action_top5"],"in_union_top5":q["in_union_top5"],"full_rank":q["full_rank"],"action_rank":q["action_rank"],"proposal_rank_weight":q["proposal_rank_weight"],"current_action":actions[t].astype(float).tolist(),"replacement_action":meta["original_action"],"replacement_source":"paired_clean_upper_bound","library_demo_id":None,"library_base_demo_id":None,"library_t":None,"state_distance":None,"action_delta_l2":float(np.linalg.norm(np.asarray(meta["original_action"],dtype=np.float32)-actions[t])),"state_in_domain":False,"action_in_domain":False,"oracle_only":True,"continuation_end_t":min(q["episode_length"],t+int(cfg["oracle"]["max_horizon"]))})
        if not repl: ood+=1
        feat.close()
    out_df=pd.DataFrame(out)
    pair_fields=labels[["pair_id","failure_type","label_status","is_effective_intervention","intervention_t","responsible_t","responsible_start","responsible_end"]].drop_duplicates("pair_id")
    out_df=out_df.merge(pair_fields,on="pair_id",how="left",validate="many_to_one")
    a.output.parent.mkdir(parents=True,exist_ok=True); out_df.to_json(a.output,orient="records",lines=True)
    summary={"split":a.split,"queries":len(queries),"replacement_rows":len(out),"ood_queries":ood,"ood_rate":float(ood/max(len(queries),1)),"source_counts":{str(k):int(v) for k,v in out_df.replacement_source.value_counts().items()} if len(out_df) else {},"proposal_queries":sum(q["query_source"].startswith("proposal") for q in queries.values()),"teacher_forced_queries":sum("teacher_forced" in q["query_source"] for q in queries.values())}; a.summary.parent.mkdir(parents=True,exist_ok=True); a.summary.write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
