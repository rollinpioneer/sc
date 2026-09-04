from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from stage3a.rescue_v02.common import _staged, _state, env_for_dataset, text
from .outcome_score_long import load_json, score_outcomes


def reset_prefix(env, initial, actions, model):
    env.reset(); env.reset_to({"states":np.asarray(initial).copy(),"model":model})
    for action in actions: env.step(action)
    return _state(env)


def rollout(env, actions, h):
    ss=[]; rr=[]; gg=[]; yy=[]
    for action in np.asarray(actions)[:h]:
        _,reward,_,_=env.step(action); ss.append(_state(env));rr.append(float(reward));gg.append(_staged(env));yy.append(bool(env.is_success().get("task",False)))
    return {"states_post":np.asarray(ss),"rewards":np.asarray(rr,np.float32),"staged_rewards":np.asarray(gg,np.float32),"success":np.asarray(yy,np.bool_)}


def exact(run, group, start):
    maxima=[]
    for key in ("states_post","rewards","staged_rewards","success"):
        a,b=np.asarray(run[key]),np.asarray(group[key])[start:start+len(run[key])]
        maxima.append(float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape and a.size else (0. if a.shape==b.shape else float("inf")))
    return bool(all(v==0 for v in maxima)),float(max(maxima))


def main():
    p=argparse.ArgumentParser();p.add_argument("--benchmark",type=Path,required=True);p.add_argument("--plans",type=Path,required=True);p.add_argument("--normalizer",type=Path,required=True);p.add_argument("--spec",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--summary",type=Path,required=True);p.add_argument("--part-index",type=int,default=0);p.add_argument("--num-parts",type=int,default=1);a=p.parse_args()
    grouped=defaultdict(list); order=[]
    for line in a.plans.read_text().splitlines():
        if not line.strip():continue
        r=json.loads(line); key=r["pair_id"]
        if key not in grouped:order.append(key)
        grouped[key].append(r)
    selected=[x for i,x in enumerate(order) if i%a.num_parts==a.part_index]; norm=load_json(a.normalizer)["normalizer"];spec=load_json(a.spec);envs={};sources={};models={};out=[]
    with h5py.File(a.benchmark,"r") as f:
        for pair_id in selected:
            plans=grouped[pair_id];g=f[f"data/{plans[0]['perturbed_demo_id']}"];task=text(g.attrs["task"]);t=int(plans[0]["query_t"]);actions=np.asarray(g["actions"])
            if task not in envs:
                source=text(g.attrs["source_dataset"]);envs[task]=(env_for_dataset(source)[0],env_for_dataset(source)[0]);sources[task]=h5py.File(source,"r");models[task]=envs[task][0].env.model.get_xml()
            initial=np.asarray(sources[task][f"data/{text(g.attrs['base_demo_id'])}"]["states"][0]).copy();pre_ref=reset_prefix(envs[task][0],initial,actions[:t],models[task]);h=min(int(spec["max_horizon"]),len(actions)-t);ref=rollout(envs[task][0],actions[t:],h);ref_ok,ref_max=exact(ref,g,t)
            for plan in plans:
                pre_repl=reset_prefix(envs[task][1],initial,actions[:t],models[task]);repl_actions=actions[t:].copy();repl_actions[0]=np.asarray(plan["replacement_action"],dtype=actions.dtype);repl=rollout(envs[task][1],repl_actions,h)
                row={k:plan.get(k) for k in ("replacement_id","query_id","pair_id","task","base_demo_id","split","perturbed_demo_id","clean_demo_id","query_t","query_source","replacement_rank","replacement_source","replacement_action","library_id","library_base_demo_id","library_clean_demo_id","library_t","state_distance","action_delta_l2","state_in_domain","action_in_domain","failure_type","is_effective_intervention")};row.update({"pair_id":pair_id,"task":task,"intervention_t":t,"branch_pre_state_equal":bool(np.array_equal(pre_ref,pre_repl)),"branch_pre_state_max_abs":float(np.max(np.abs(pre_ref-pre_repl))),"reference_exact":ref_ok,"reference_max_abs":ref_max});row.update(score_outcomes(task=task,reference_rewards=ref["rewards"],replacement_rewards=repl["rewards"],reference_staged=ref["staged_rewards"],replacement_staged=repl["staged_rewards"],reference_success=ref["success"],replacement_success=repl["success"],normalizer=norm,spec=spec));out.append(row)
    for x,y in envs.values():x.close();y.close()
    for x in sources.values():x.close()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text("\n".join(json.dumps(x) for x in out)+"\n");summary={"row_count":len(out),"pair_count":len(selected),**{f"{key}_rate":float(np.mean([r[key] for r in out])) if out else 0. for key in ("branch_pre_state_equal","reference_exact","finite_target")}};a.summary.parent.mkdir(parents=True,exist_ok=True);a.summary.write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__ == "__main__":main()
