"""Convert raw simulator deltas into the frozen continuous oracle targets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser()
    for n in ("train-raw","validation-raw","train-plan","validation-plan","transition-labels","config","output-normalizer","output-train","output-validation"): p.add_argument("--"+n,type=Path,required=True)
    return p.parse_args()

def main():
    a=parse_args(); cfg=json.loads(a.config.read_text()); frames=[]
    labels=pd.read_parquet(a.transition_labels)
    # The plan already carries the pair labels.  Keep the benchmark labels as
    # a consistency source, but merge them with explicit suffix handling so
    # that the target table has one canonical field per label (rather than the
    # accidental ``*_x``/``*_y`` columns produced by a default merge).
    pair_labels=labels[labels.variant.eq("perturbed")][["pair_id","failure_type","label_status","is_effective_intervention","intervention_t","responsible_t","responsible_start","responsible_end"]].drop_duplicates("pair_id")
    for raw,plan,split in ((a.train_raw,a.train_plan,"train"),(a.validation_raw,a.validation_plan,"validation")):
        r=pd.read_json(raw,lines=True); p=pd.read_json(plan,lines=True)
        if len(r):
            keep=[c for c in p.columns if c not in r.columns or c in {"replacement_id"}]
            r=r.merge(p[keep],on="replacement_id",how="left",validate="one_to_one")
            r=r.merge(pair_labels,on="pair_id",how="left",validate="many_to_one",suffixes=("", "_labels"))
            for col in pair_labels.columns:
                if col == "pair_id":
                    continue
                label_col = f"{col}_labels"
                if label_col not in r.columns:
                    continue
                if col not in r.columns:
                    r[col] = r[label_col]
                else:
                    # Prefer the value frozen in the plan, filling only an
                    # absent value from the benchmark label table.
                    r[col] = r[col].combine_first(r[label_col])
                r.drop(columns=[label_col], inplace=True)
        r["split"]=split; frames.append(r)
    train,val=frames; train_valid=train[(~train.oracle_only.fillna(False)) & train.reference_replay_ok.fillna(False) & (train.actual_horizon>=10)] if len(train) else train
    scales={}
    for task in ("can","square"):
        scales[task]={}
        for h in (10,20,40):
            sub=train_valid[train_valid.task.eq(task)] if len(train_valid) else train_valid
            scales[task][str(h)]={}
            for name in ("dense_mean_delta","stage_mean_delta"):
                x=pd.to_numeric(sub.get(f"{name}_h{h}",pd.Series(dtype=float)),errors="coerce").to_numpy(dtype=float); x=x[np.isfinite(x)]
                scales[task][str(h)][name.replace("_delta","")]=float(max(np.quantile(np.abs(x),float(cfg["target"]["robust_scale_quantile"])),1e-6)) if len(x) else 1e-6
    def add(df):
        if not len(df): return df
        out=df.copy(); ih=[]
        for _,row in out.iterrows():
            task=str(row.get("task","can")); im=[]
            for h in (10,20,40):
                sc=scales.get(task,{}).get(str(h),{}); dn=float(np.clip(float(row.get(f"dense_mean_delta_h{h}",0.0))/sc.get("dense_mean",1e-6),-1,1)); sn=float(np.clip(float(row.get(f"stage_mean_delta_h{h}",0.0))/sc.get("stage_mean",1e-6),-1,1)); sd=float(row.get(f"success_delta_h{h}",0.0)); im.append(float(cfg["target"]["dense_weight"])*dn+float(cfg["target"]["stage_weight"])*sn+float(cfg["target"]["success_weight"])*sd)
            ih.append(im)
        arr=np.asarray(ih,float); w=np.asarray(cfg["target"]["horizon_weights"],float); out["improvement_h10"],out["improvement_h20"],out["improvement_h40"]=arr[:,0],arr[:,1],arr[:,2]; out["oracle_improvement"]=(arr*w[None,:]).sum(1); out["oracle_positive"]=(out.oracle_improvement>=float(cfg["target"]["positive_improvement_threshold"])).astype(bool); out["target_valid"]=out.reference_replay_ok.fillna(False)&(pd.to_numeric(out.actual_horizon,errors="coerce")>=10); out["verifier_eligible"]=out.target_valid&(~out.oracle_only.fillna(False))&out.state_in_domain.fillna(False)&out.action_in_domain.fillna(False); return out
    train,val=add(train),add(val)
    normalizer={"scales":scales,"formula":{"dense_weight":0.4,"stage_weight":0.5,"success_weight":0.1,"horizon_weights":[0.2,0.3,0.5],"positive_threshold":0.05},"train_rows_for_scale":int(len(train_valid))}
    a.output_normalizer.parent.mkdir(parents=True,exist_ok=True); a.output_normalizer.write_text(json.dumps(normalizer,indent=2),encoding="utf-8"); a.output_train.parent.mkdir(parents=True,exist_ok=True); train.to_parquet(a.output_train,index=False); a.output_validation.parent.mkdir(parents=True,exist_ok=True); val.to_parquet(a.output_validation,index=False); print(json.dumps({"train_rows":len(train),"validation_rows":len(val),"train_valid":int(train.target_valid.sum()) if len(train) else 0,"validation_valid":int(val.target_valid.sum()) if len(val) else 0,"scales":scales},indent=2))
if __name__=="__main__": main()
