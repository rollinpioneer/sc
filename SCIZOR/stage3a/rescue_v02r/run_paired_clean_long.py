from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from stage3a.rescue_v02.common import env_for_dataset, replay, text
from .outcome_score_long import load_json, score_outcomes


def exact(left, right):
    maxima = []
    for a, b in zip(left, right):
        a, b = np.asarray(a), np.asarray(b)
        maxima.append(float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.shape == b.shape and a.size else (0.0 if a.shape == b.shape else float("inf")))
    return bool(all(v == 0 for v in maxima)), float(max(maxima))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--benchmark", type=Path, required=True); p.add_argument("--metadata", type=Path, required=True); p.add_argument("--normalizer", type=Path, required=True); p.add_argument("--spec", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--summary", type=Path, required=True); p.add_argument("--part-index", type=int, default=0); p.add_argument("--num-parts", type=int, default=1); a=p.parse_args()
    meta = {r["pair_id"]: r for r in (json.loads(x) for x in a.metadata.read_text().splitlines() if x.strip())}; norm = load_json(a.normalizer)["normalizer"]; spec = load_json(a.spec); output=[]; envs={}; sources={}; models={}; seen=0
    with h5py.File(a.benchmark, "r") as f:
        for g in f["data"].values():
            if text(g.attrs.get("variant")) != "perturbed": continue
            selected = seen % a.num_parts == a.part_index; seen += 1
            if not selected: continue
            task, pair_id = text(g.attrs["task"]), text(g.attrs["pair_id"])
            if task not in envs:
                source = text(g.attrs["source_dataset"]); envs[task] = (env_for_dataset(source)[0], env_for_dataset(source)[0]); sources[task] = h5py.File(source, "r"); models[task] = envs[task][0].env.model.get_xml()
            clean = f[f"data/{text(g.attrs['clean_demo_id'])}"]; initial=np.asarray(sources[task][f"data/{text(g.attrs['base_demo_id'])}"]["states"][0]).copy()
            pert_roll = replay(envs[task][0], initial, np.asarray(g["actions"]), render_images=False, model_xml=models[task]); clean_roll=replay(envs[task][1], initial, np.asarray(clean["actions"]), render_images=False, model_xml=models[task]); t=int(g.attrs["perturb_t"])
            ref_ok, ref_max=exact((pert_roll["states_post"],pert_roll["rewards"],pert_roll["staged_rewards"],pert_roll["success"]),(g["states_post"],g["rewards"],g["staged_rewards"],g["success"])); clean_ok, clean_max=exact((clean_roll["states_post"],clean_roll["rewards"],clean_roll["staged_rewards"],clean_roll["success"]),(clean["states_post"],clean["rewards"],clean["staged_rewards"],clean["success"]))
            row={"pair_id":pair_id,"task":task,"base_demo_id":text(g.attrs["base_demo_id"]),"perturb_t":t,"failure_type":text(g.attrs.get("failure_type","")),"is_effective_intervention":bool(g.attrs.get("is_effective_intervention",False)),"split":str(meta[pair_id]["split"]),"branch_pre_state_equal":bool(np.array_equal(pert_roll["states_pre"][t],clean_roll["states_pre"][t])),"branch_pre_state_max_abs":float(np.max(np.abs(pert_roll["states_pre"][t]-clean_roll["states_pre"][t]))),"reference_exact":ref_ok,"reference_max_abs":ref_max,"paired_clean_exact":clean_ok,"paired_clean_max_abs":clean_max}
            row.update(score_outcomes(task=task,reference_rewards=pert_roll["rewards"][t:],replacement_rewards=clean_roll["rewards"][t:],reference_staged=pert_roll["staged_rewards"][t:],replacement_staged=clean_roll["staged_rewards"][t:],reference_success=pert_roll["success"][t:],replacement_success=clean_roll["success"][t:],normalizer=norm,spec=spec)); output.append(row)
    for x,y in envs.values(): x.close(); y.close()
    for x in sources.values(): x.close()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text("\n".join(json.dumps(x) for x in output)+"\n")
    summary={"pair_count":len(output),**{f"{key}_rate":float(np.mean([r[key] for r in output])) if output else 0.0 for key in ("branch_pre_state_equal","reference_exact","paired_clean_exact","finite_target")}}; a.summary.parent.mkdir(parents=True,exist_ok=True); a.summary.write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
