from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def metric(data, key):
    y=np.asarray([bool(x["is_effective_intervention"]) for x in data],int);s=np.asarray([float(x[key]) for x in data],float);good=np.isfinite(s);y,s=y[good],s[good];pos,neg=y==1,y==0
    if not pos.any() or not neg.any():return {"auroc":None,"auprc":None,"n":len(y),"positive":int(pos.sum())}
    order=np.argsort(s,kind="mergesort");rank=np.empty(len(s));i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:j+=1
        rank[order[i:j]]=(i+j+1)/2;i=j
    auc=float((rank[pos].sum()-pos.sum()*(pos.sum()+1)/2)/(pos.sum()*neg.sum()));yy=y[np.argsort(-s,kind="mergesort")];ap=float((np.cumsum(yy)/np.arange(1,len(yy)+1)*yy).sum()/pos.sum())
    return {"auroc":auc,"auprc":ap,"n":len(y),"positive":int(pos.sum())}


def distribution(data,key):
    out={}
    for name,value in (("effective",True),("no_effect",False)):
        x=np.asarray([float(r[key]) for r in data if bool(r["is_effective_intervention"])==value],float);out[name]={"n":len(x),"mean":float(x.mean()) if len(x) else None,"median":float(np.median(x)) if len(x) else None,"q25":float(np.quantile(x,.25)) if len(x) else None,"q75":float(np.quantile(x,.75)) if len(x) else None}
    return out


def task(data,key):return {t:{"metrics":metric([r for r in data if r["task"]==t],key),"distribution":distribution([r for r in data if r["task"]==t],key)} for t in ("can","square")}


def main():
    p=argparse.ArgumentParser();p.add_argument("--paired-clean",type=Path,required=True);p.add_argument("--feasible",type=Path,required=True);p.add_argument("--split",choices=["train","validation","confirmation"],required=True);p.add_argument("--score-key",default="counterfactual_improvement_long");p.add_argument("--output",type=Path,required=True);a=p.parse_args();key=a.score_key
    paired=[r for r in read(a.paired_clean) if r["split"]==a.split];feasible=[r for r in read(a.feasible) if r["split"]==a.split];primary=[r for r in feasible if int(r["replacement_rank"])==0];by=defaultdict(list)
    for r in feasible:by[r["pair_id"]].append(r)
    best=[max(v,key=lambda r:float(r[key])) for v in by.values()]
    pe={f"{k}_rate":float(np.mean([r[k] for r in paired])) if paired else 0. for k in ("branch_pre_state_equal","reference_exact","paired_clean_exact","finite_target")};fe={f"{k}_rate":float(np.mean([r[k] for r in feasible])) if feasible else 0. for k in ("branch_pre_state_equal","reference_exact","finite_target")}
    engineering=pe["branch_pre_state_equal_rate"]==1 and pe["reference_exact_rate"]>=.999 and pe["paired_clean_exact_rate"]>=.999 and pe["finite_target_rate"]>=.99 and fe["branch_pre_state_equal_rate"]==1 and fe["reference_exact_rate"]>=.999 and fe["finite_target_rate"]>=.99
    result={"split":a.split,"score_key":key,"paired_clean":{"metrics":metric(paired,key),"distribution":distribution(paired,key),"task_metrics":task(paired,key),"engineering":pe},"primary_feasible":{"metrics":metric(primary,key),"distribution":distribution(primary,key),"task_metrics":task(primary,key)},"best_of_4_feasible_diagnostic":{"metrics":metric(best,key),"distribution":distribution(best,key),"task_metrics":task(best,key)},"feasible_engineering":fe,"engineering_pass":bool(engineering),"pair_count":len(paired),"feasible_row_count":len(feasible)}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()
