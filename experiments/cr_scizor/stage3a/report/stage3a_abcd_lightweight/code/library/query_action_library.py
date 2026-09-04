"""Deterministic train-only replacement-action retrieval."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

def load_library(directory: str | Path) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict, dict]:
    d=Path(directory); index=pd.read_parquet(d/"action_library_index.parquet"); arrays={}
    for task in ("can","square"):
        with np.load(d/f"library_{task}.npz") as z: arrays[task]={k:z[k].copy() for k in z.files}
        arrays[task]["faiss_index"] = faiss.read_index(str(d / f"state_{task}.faiss"))
    thresholds=json.loads((d/"support_thresholds.json").read_text()); codebooks={t:json.loads((d/f"codebook_{t}.json").read_text()) for t in ("can","square")}; return index,arrays,thresholds,codebooks

def retrieve_replacements(task, query_state, current_action, query_base_demo_id, query_relative_position, previous_gripper_action, library, config):
    index, arrays, thresholds, codebooks = library; index=index[index.task.eq(task)].reset_index(drop=True); ar=arrays[task]; raw=np.asarray(query_state,dtype=np.float32); raw=np.clip((raw-ar["state_mean"])/np.maximum(ar["state_std"],1e-6),-10.0,10.0); q=np.concatenate([raw,[query_relative_position,previous_gripper_action]]).astype(np.float32); q/=max(float(np.linalg.norm(q)),1e-8); k=min(int(config["library"]["state_query_neighbors"]), ar["vectors"].shape[0]); faiss_dist, faiss_order=ar["faiss_index"].search(np.ascontiguousarray(q[None,:]), k); faiss_dist, faiss_order=faiss_dist[0], faiss_order[0]; order=faiss_order; dist_by_row={int(j):float(d) for j,d in zip(faiss_order,faiss_dist) if int(j)>=0}
    lim=thresholds[task]; max_delta=float(lim["action_delta_q99"]); min_delta=float(config["library"]["min_action_delta_l2"]); state_lim=float(lim["state_distance_q95"]); chosen=[]; seen_demo=set(); seen_clusters=set()
    for j in order:
        if int(j) < 0: continue
        r=index.iloc[int(j)]; delta=float(np.linalg.norm(ar["actions"][j]-current_action));
        if config["library"]["exclude_same_base_demo"] and str(r.base_demo_id)==str(query_base_demo_id): continue
        if float(dist_by_row[int(j)])>state_lim or delta<min_delta or delta>max_delta: continue
        if any(float(np.linalg.norm(ar["actions"][j]-ar["actions"][x]))<float(config["library"]["action_diversity_l2"]) for x in chosen): continue
        chosen.append(int(j)); seen_demo.add(str(r.base_demo_id)); seen_clusters.add(int(r.action_cluster_id))
        if len(chosen)>=3: break
    if len(chosen)<int(config["library"]["replacements_per_query"]):
        med_by_cluster={int(m["cluster_id"]):m for m in codebooks[task]["medoids"]}
        for m in sorted(codebooks[task]["medoids"],key=lambda x:(-x["cluster_size"],x["cluster_id"])):
            if len(chosen)>=int(config["library"]["replacements_per_query"]): break
            if m["cluster_id"] in seen_clusters: continue
            j=int(index.index[index.library_id==m["library_id"]][0]); r=index.iloc[j]; delta=float(np.linalg.norm(ar["actions"][j]-current_action));
            if str(r.base_demo_id)==str(query_base_demo_id) or delta<min_delta or delta>max_delta: continue
            if any(float(np.linalg.norm(ar["actions"][j]-ar["actions"][x]))<float(config["library"]["action_diversity_l2"]) for x in chosen): continue
            chosen.append(j); seen_clusters.add(m["cluster_id"])
    out=[]
    for n,j in enumerate(chosen):
        r=index.iloc[j]; delta=float(np.linalg.norm(ar["actions"][j]-current_action)); row_dist=float(dist_by_row.get(int(j), np.sum((ar["vectors"][j]-q)**2))); out.append({"replacement_id":n,"replacement_source":"nn_real" if n<3 else "codebook_medoid","library_id":str(r.library_id),"library_demo_id":str(r.demo_id),"library_base_demo_id":str(r.base_demo_id),"library_t":int(r.t),"replacement_action":ar["actions"][j].astype(float).tolist(),"state_distance":row_dist,"action_delta_l2":delta,"action_cluster_id":int(r.action_cluster_id),"state_in_domain":bool(row_dist<=state_lim),"action_in_domain":bool(min_delta<=delta<=max_delta)})
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--proposal-candidates",type=Path,required=True); p.add_argument("--feature-index",type=Path,required=True); p.add_argument("--library-dir",type=Path,required=True); p.add_argument("--config",type=Path,required=True); p.add_argument("--max-queries",type=int,default=8); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); cfg=json.loads(a.config.read_text()); lib=load_library(a.library_dir); fi=pd.read_parquet(a.feature_index).set_index("demo_id"); props=pd.read_parquet(a.proposal_candidates).drop_duplicates(["pair_id","t"]).head(a.max_queries); rows=[]
    for r in props.itertuples(index=False):
        f=np.load(fi.loc[r.demo_id].feature_path); states=f["states"].astype(np.float32); acts=f["actions"].astype(np.float32); with_valid=f["state_valid_mask"].astype(bool); ns=states.copy(); rows_state=ns[int(r.t)];
        repl=retrieve_replacements(str(r.task),rows_state,acts[int(r.t)],str(r.base_demo_id),float(r.t/max(int(r.episode_length)-1,1)),float(acts[max(0,int(r.t)-1),-1]),lib,cfg); rows.append({"pair_id":str(r.pair_id),"demo_id":str(r.demo_id),"task":str(r.task),"base_demo_id":str(r.base_demo_id),"t":int(r.t),"replacements":repl,"ood":not bool(repl)})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(json.dumps(x) for x in rows)+"\n",encoding="utf-8"); print(json.dumps({"queries":len(rows),"replacement_counts":[len(x["replacements"]) for x in rows]}))
if __name__=="__main__": main()
