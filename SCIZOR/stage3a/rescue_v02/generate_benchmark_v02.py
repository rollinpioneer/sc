from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from stage1.benchmark.intervention_times import select_intervention_times
from stage1.benchmark.outcome_labels import label_outcome
from .common import env_for_dataset, env_metadata, perturb_action, replay, sha256_bytes, sha256_file, text


PERTURBATIONS = ("zero_motion", "reverse_motion", "flip_gripper", "axis_impulse")


def write_rollout(parent, rollout, include_images=True):
    for key in ("actions", "states_pre", "states_post", "rewards", "staged_rewards", "success"):
        parent.create_dataset(key, data=rollout[key], compression="gzip")
    if include_images:
        obs = parent.create_group("obs")
        obs.create_dataset("agentview_image_pre", data=rollout["images_pre"], compression="gzip")
        obs.create_dataset("agentview_image_post", data=rollout["images_post"], compression="gzip")


def magnitude(task, kind, pair_meta):
    vals = [float(r["magnitude"]) for r in pair_meta if r.get("task") == task and r.get("perturbation_type") == kind]
    if not vals:
        raise ValueError(f"no frozen magnitude for {task}/{kind}")
    return float(vals[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-manifest", type=Path, required=True)
    p.add_argument("--can-source", type=Path, required=True)
    p.add_argument("--square-source", type=Path, required=True)
    p.add_argument("--stage1-pair-meta", type=Path, required=True)
    p.add_argument("--runtime-fingerprint", type=Path, required=True)
    p.add_argument("--output-hdf5", type=Path, required=True)
    p.add_argument("--output-meta", type=Path, required=True)
    p.add_argument("--num-intervention-points", type=int, default=4)
    p.add_argument("--perturbations", default=",".join(PERTURBATIONS))
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--no-images", action="store_true")
    a = p.parse_args()
    bases = json.loads(a.base_manifest.read_text(encoding="utf-8"))
    pair_meta = [json.loads(line) for line in a.stage1_pair_meta.read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = tuple(x.strip() for x in a.perturbations.split(","))
    runtime_hash = sha256_file(a.runtime_fingerprint)
    sources = {"can": a.can_source, "square": a.square_source}
    if a.output_hdf5.exists():
        raise FileExistsError(a.output_hdf5)
    a.output_hdf5.parent.mkdir(parents=True, exist_ok=True); a.output_meta.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    total = 0
    with h5py.File(a.output_hdf5, "w") as out, a.output_meta.open("w", encoding="utf-8") as mf:
        data = out.create_group("data")
        data.attrs["benchmark_version"] = "v0.2_replay_locked"
        data.attrs["created_from"] = "current_runtime_self_consistent_replay"
        data.attrs["v01_lineage_required"] = False
        data.attrs["generator_git_commit"] = "recorded_in_runtime_fingerprint"
        data.attrs["generator_source_sha256"] = sha256_file(Path(__file__))
        data.attrs["runtime_fingerprint_sha256"] = runtime_hash
        for task in ("can", "square"):
            env, _ = env_for_dataset(sources[task])
            try:
                runtime = env_metadata(env, runtime_hash)
                with h5py.File(sources[task], "r") as src:
                    for candidate in bases[task]:
                        demo_id = str(candidate["demo_id"])
                        demo = src[f"data/{demo_id}"]
                        actions = np.asarray(demo["actions"][:]).copy(); initial = np.asarray(demo["states"][0]).copy()
                        model_xml = env.env.model.get_xml()
                        clean_a = replay(env, initial, actions, render_images=not a.no_images, model_xml=model_xml)
                        clean_b = replay(env, initial, actions, render_images=not a.no_images, model_xml=model_xml)
                        for key in ("states_pre", "states_post", "rewards", "staged_rewards", "success"):
                            if not np.array_equal(clean_a[key], clean_b[key]):
                                raise RuntimeError(f"clean determinism failed for {task}/{demo_id}/{key}")
                        if not bool(clean_a["success"][-1]):
                            raise RuntimeError(f"clean final success failed for {task}/{demo_id}")
                        base_attrs = {
                            "task": task, "base_demo_id": demo_id, "source_dataset": str(sources[task]),
                            "source_model_file": text(demo.attrs.get("model_file", "")),
                            "source_model_sha256": hashlib.sha256(str(demo.attrs.get("model_file", "")).encode()).hexdigest(),
                            "source_initial_state_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                            "source_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                            "state_layout": "explicit_pre_and_post", "split": str(candidate.get("split", "pilot")), **runtime,
                        }
                        clean_name = f"{task}_{demo_id}_clean"
                        clean_group = data.create_group(clean_name); write_rollout(clean_group, clean_a, not a.no_images)
                        for key, value in base_attrs.items(): clean_group.attrs[key] = value
                        clean_group.attrs.update({"variant": "clean", "pair_id": f"{task}:{demo_id}:clean", "perturb_t": -1, "failure_type": "clean", "final_success": True})
                        plan = select_intervention_times(actions, num_times=a.num_intervention_points)
                        low, high = env.env.action_spec
                        for perturb_t, reason in plan:
                            for kind in kinds:
                                mag = magnitude(task, kind, pair_meta)
                                pert_actions = actions.copy(); pert_actions[int(perturb_t)] = perturb_action(actions[int(perturb_t)], kind, mag, rng, low, high)
                                if int(np.count_nonzero(np.any(pert_actions != actions, axis=1))) != 1:
                                    raise RuntimeError(f"single-action check failed for {task}/{demo_id}/{perturb_t}/{kind}")
                                pert = replay(env, initial, pert_actions, render_images=not a.no_images, model_xml=model_xml)
                                label = label_outcome(clean_a, pert, int(perturb_t))
                                pair_id = f"{task}:{demo_id}:{kind}:t{int(perturb_t)}:m{mag:.2f}"
                                name = f"{task}_{demo_id}_{kind}_t{int(perturb_t)}_m{mag:.2f}"
                                g = data.create_group(name); write_rollout(g, pert, not a.no_images)
                                attrs = {**base_attrs, "variant": "perturbed", "pair_id": pair_id, "perturb_t": int(perturb_t), "selection_reason": reason, "perturbation_type": kind, "magnitude": mag, "original_action": json.dumps(actions[int(perturb_t)].tolist()), "perturbed_action": json.dumps(pert_actions[int(perturb_t)].tolist()), "action_diff_count": 1, "failure_onset": -1 if label["failure_onset"] is None else int(label["failure_onset"]), "failure_type": label["failure_type"], "recovery_start": -1 if label["recovery_start"] is None else int(label["recovery_start"]), "recovery_end": -1 if label["recovery_end"] is None else int(label["recovery_end"]), "final_success": bool(pert["success"][-1]), "is_effective_intervention": label["failure_type"] in {"direct_failure", "delayed_failure", "recovery_failure", "recovery_success"}, "clean_demo_id": clean_name, "episode_length": len(pert_actions)}
                                for key, value in attrs.items(): g.attrs[key] = value
                                mf.write(json.dumps({"pair_id": pair_id, "task": task, "base_demo_id": demo_id, "split": str(candidate.get("split", "pilot")), "clean_demo_id": clean_name, "perturbed_demo_id": name, "perturb_t": int(perturb_t), "perturbation_type": kind, "magnitude": mag, **{k: label.get(k) for k in ("failure_onset", "failure_type", "recovery_start", "recovery_end", "label_status")}, "final_success_clean": True, "final_success_perturbed": bool(pert["success"][-1]), "is_effective_intervention": attrs["is_effective_intervention"]}, ensure_ascii=False) + "\n")
                                total += 1
                        # A simulator crash must not leave a structurally
                        # unreadable HDF5 shard. The complete shard is still
                        # required for merge, but this checkpoint makes a
                        # failed attempt diagnosable and bounded to one demo.
                        mf.flush()
                        out.flush()
            finally:
                env.close()
        data.attrs["total_pairs"] = total
    print(json.dumps({"pair_count": total, "output_hdf5": str(a.output_hdf5), "output_meta": str(a.output_meta)}, indent=2))


if __name__ == "__main__":
    main()
