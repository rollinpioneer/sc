from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from stage3a.rescue_v02.common import compare_rollouts, env_for_dataset, replay


def number(name):
    try: return int(name.rsplit("_", 1)[-1])
    except ValueError: return 10**9


def used(v01, v02):
    out={"can":set(),"square":set()}
    for line in v01.read_text().splitlines():
        if line.strip():
            x=json.loads(line);out[x["task"]].add(x["base_demo_id"])
    for task, rows in json.loads(v02.read_text()).items():
        for row in rows:out[task].add(row["demo_id"])
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--can-source",type=Path,required=True);p.add_argument("--square-source",type=Path,required=True);p.add_argument("--v01-metadata",type=Path,required=True);p.add_argument("--v02-base-manifest",type=Path,required=True);p.add_argument("--num-per-task",type=int,default=8);p.add_argument("--output",type=Path,required=True);p.add_argument("--details",type=Path,required=True);a=p.parse_args();taken=used(a.v01_metadata,a.v02_base_manifest);sources={"can":a.can_source,"square":a.square_source};selected={"can":[],"square":[]};details=[];envs={};handles={}
    try:
        for task,source in sources.items():
            handles[task]=h5py.File(source,"r");envs[task]=(env_for_dataset(str(source))[0],env_for_dataset(str(source))[0]);model=envs[task][0].env.model.get_xml()
            for demo in sorted(handles[task]["data"].keys(),key=number):
                if demo in taken[task]:continue
                g=handles[task][f"data/{demo}"];initial=np.asarray(g["states"][0]).copy();actions=np.asarray(g["actions"]).copy();left=replay(envs[task][0],initial,actions,render_images=False,model_xml=model);right=replay(envs[task][1],initial,actions,render_images=False,model_xml=model);comparison=compare_rollouts(left,right);row={"task":task,"demo_id":demo,"steps":len(actions),"determinism_pass":bool(comparison["pass"]),"final_success":bool(left["success"][-1]),"comparison":comparison,"split":"confirmation","source_dataset":str(source)};details.append(row)
                if row["determinism_pass"] and row["final_success"]:selected[task].append(row)
                if len(selected[task])==a.num_per_task:break
            if len(selected[task])<a.num_per_task:raise RuntimeError(f"{task}: only {len(selected[task])} selected")
    finally:
        for x,y in envs.values():x.close();y.close()
        for x in handles.values():x.close()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.details.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(selected,indent=2));a.details.write_text("\n".join(json.dumps(x) for x in details)+"\n");print(json.dumps({k:[x["demo_id"] for x in v] for k,v in selected.items()},indent=2))
if __name__=="__main__":main()
