import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .intervention_times import select_intervention_times
from .outcome_labels import label_outcome
from .simulator_replay import create_shaped_env, replay_episode


PERTURBATION_TYPES = ["zero_motion", "reverse_motion", "flip_gripper", "axis_impulse"]


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _load_magnitude_config(path):
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset_for_task(datasets, tasks, task):
    if len(datasets) == 1:
        return datasets[0]
    idx = tasks.index(task)
    return datasets[idx]


def _selected_magnitude(config, task, perturbation_type, candidate_magnitude=None):
    task_cfg = config.get(task, {}) if isinstance(config, dict) else {}
    if isinstance(config, dict) and "selected" in config:
        selected = config["selected"].get(task, {})
        if perturbation_type in selected:
            chosen = selected[perturbation_type]
            if isinstance(chosen, dict) and "magnitude" in chosen:
                return float(chosen["magnitude"])
            return float(chosen)
    if perturbation_type in task_cfg:
        value = task_cfg[perturbation_type]
        if isinstance(value, dict) and "magnitude" in value:
            return float(value["magnitude"])
        return float(value)
    if "default" in task_cfg:
        return float(task_cfg["default"])
    if perturbation_type in config:
        value = config[perturbation_type]
        if isinstance(value, dict) and "magnitude" in value:
            return float(value["magnitude"])
        return float(value)
    if "default" in config:
        return float(config["default"])
    if candidate_magnitude is not None:
        return float(candidate_magnitude)
    return 0.5


def _maybe_flip_images(images, flip):
    return images[:, ::-1] if flip else images


def _write_rollout_group(data_root, name, rollout, attrs, render_flip, include_images=True):
    grp = data_root.create_group(name)
    grp.create_dataset("actions", data=np.asarray(rollout["actions"]), compression="gzip")
    grp.create_dataset("original_actions", data=np.asarray(rollout["original_actions"]), compression="gzip")
    grp.create_dataset("states", data=np.asarray(rollout["states"]), compression="gzip")
    grp.create_dataset("rewards", data=np.asarray(rollout["rewards"], dtype=np.float32), compression="gzip")
    grp.create_dataset("success", data=np.asarray(rollout["success"], dtype=np.bool_), compression="gzip")
    if include_images and rollout.get("images") is not None:
        obs = grp.require_group("obs")
        obs.create_dataset(
            "agentview_image",
            data=np.asarray(_maybe_flip_images(rollout["images"], render_flip), dtype=np.uint8),
            compression="gzip",
        )
    if rollout.get("staged_rewards") is not None and len(rollout["staged_rewards"]):
        grp.create_dataset("staged_rewards", data=np.asarray(rollout["staged_rewards"], dtype=np.float32), compression="gzip")
    for key, value in attrs.items():
        grp.attrs[key] = value
    return grp.name.split("/")[-1]


def _write_metadata_line(handle, record):
    handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")


def _episode_records(task, dataset_path, mode, episodes_per_task, interventions_per_episode, perturbation_types, magnitude_config, candidate_magnitudes, seed, output_hdf5, output_metadata, demo_start=0, demo_end=None):
    rng = np.random.default_rng(seed)
    with h5py.File(dataset_path, "r") as src, h5py.File(output_hdf5, "a") as out:
        data = out.require_group("data")
        env_args = json.loads(src["data"].attrs["env_args"])
        data.attrs["env_args"] = src["data"].attrs["env_args"]
        data.attrs["source_dataset"] = dataset_path
        env, _ = create_shaped_env(dataset_path)
        render_flip = None
        pair_count = 0
        demo_keys = [k for k in sorted(src["data"]) if "states" in src["data"][k] and "actions" in src["data"][k]]
        demo_keys = demo_keys[int(demo_start):int(demo_end) if demo_end is not None else None]
        with output_metadata.open("a", encoding="utf-8") as meta_f:
            clean_kept = 0
            for demo_key in demo_keys:
                if clean_kept >= episodes_per_task:
                    break
                demo = src["data"][demo_key]
                initial_state = {"states": demo["states"][0]}
                include_images = mode != "calibration"
                clean = replay_episode(env, initial_state, demo["actions"][:], collect_images=include_images)
                if not bool(clean["success"][-1]):
                    continue
                clean_kept += 1
                if render_flip is None and include_images:
                    stored = np.asarray(demo["obs/agentview_image"][0]) if "obs/agentview_image" in demo else None
                    if stored is None:
                        render_flip = False
                    else:
                        raw = float(np.mean(np.abs(clean["images"][0].astype(float) - stored.astype(float))))
                        flip = float(np.mean(np.abs(clean["images"][0][::-1].astype(float) - stored.astype(float))))
                        render_flip = bool(flip < raw)
                if render_flip is None:
                    render_flip = False
                intervention_plan = select_intervention_times(clean["original_actions"], num_times=interventions_per_episode)
                clean_demo_id = f"{task}_{demo_key}_clean"
                if clean_demo_id not in data:
                    clean_attrs = {
                        "pair_id": f"{task}:{demo_key}:clean",
                        "variant": "clean",
                        "base_demo_id": demo_key,
                        "clean_demo_id": clean_demo_id,
                        "perturbed_demo_id": "",
                        "task": task,
                        "source_dataset": dataset_path,
                        "perturb_t": -1,
                        "selection_reason": "clean",
                        "perturbation_type": "",
                        "magnitude": -1.0,
                        "failure_onset": -1,
                        "failure_type": "clean",
                        "responsible_t": -1,
                        "responsible_start": -1,
                        "responsible_end": -1,
                        "recovery_start": -1,
                        "recovery_end": -1,
                        "gap_threshold": -1.0,
                        "final_success_clean": bool(clean["success"][-1]),
                        "final_success_perturbed": bool(clean["success"][-1]),
                        "label_status": "clean",
                        "control_freq": int(env_args.get("env_kwargs", {}).get("control_freq", 20)),
                        "episode_length": int(len(clean["actions"])),
                        "render_flip_applied": bool(render_flip),
                    }
                    _write_rollout_group(
                        data,
                        clean_demo_id,
                        clean,
                        clean_attrs,
                        render_flip,
                        include_images=include_images,
                    )
                for perturb_t, reason in intervention_plan:
                    for perturbation_type in perturbation_types:
                        if mode == "calibration":
                            mags = list(candidate_magnitudes)
                        else:
                            mags = [_selected_magnitude(magnitude_config, task, perturbation_type, candidate_magnitudes[0] if candidate_magnitudes else None)]
                        for magnitude in mags:
                            perturbed = replay_episode(
                                env,
                                initial_state,
                                demo["actions"][:],
                                perturb_t=perturb_t,
                                perturb_kind=perturbation_type,
                                magnitude=magnitude,
                                rng=rng,
                                collect_images=include_images,
                            )
                            label = label_outcome(clean, perturbed, perturb_t)
                            pair_id = f"{task}:{demo_key}:{perturbation_type}:t{int(perturb_t)}:m{float(magnitude):.2f}"
                            perturbed_demo_id = f"{task}_{demo_key}_{perturbation_type}_t{int(perturb_t)}_m{float(magnitude):.2f}"
                            attrs = {
                                "pair_id": pair_id,
                                "variant": "perturbed",
                                "base_demo_id": demo_key,
                                "clean_demo_id": clean_demo_id,
                                "perturbed_demo_id": perturbed_demo_id,
                                "task": task,
                                "source_dataset": dataset_path,
                                "perturb_t": int(perturb_t),
                                "selection_reason": reason,
                                "perturbation_type": perturbation_type,
                                "magnitude": float(magnitude),
                                "failure_onset": -1 if label["failure_onset"] is None else int(label["failure_onset"]),
                                "failure_type": label["failure_type"],
                                "responsible_t": int(label["responsible_t"]),
                                "responsible_start": int(label["responsible_start"]),
                                "responsible_end": int(label["responsible_end"]),
                                "recovery_start": -1 if label["recovery_start"] is None else int(label["recovery_start"]),
                                "recovery_end": -1 if label["recovery_end"] is None else int(label["recovery_end"]),
                                "gap_threshold": float(label.get("gap_threshold", -1.0)),
                                "final_success_clean": bool(label["final_success_clean"]),
                                "final_success_perturbed": bool(label["final_success_perturbed"]),
                                "label_status": label["label_status"],
                                "control_freq": int(env_args.get("env_kwargs", {}).get("control_freq", 20)),
                                "episode_length": int(len(perturbed["actions"])),
                                "render_flip_applied": bool(render_flip),
                            }
                            _write_rollout_group(
                                data,
                                perturbed_demo_id,
                                perturbed,
                                attrs,
                                render_flip,
                                include_images=include_images,
                            )
                            record = {
                                "pair_id": pair_id,
                                "task": task,
                                "source_dataset": dataset_path,
                                "base_demo_id": demo_key,
                                "clean_demo_id": clean_demo_id,
                                "perturbed_demo_id": perturbed_demo_id,
                                "perturb_t": int(perturb_t),
                                "selection_reason": reason,
                                "perturbation_type": perturbation_type,
                                "magnitude": float(magnitude),
                                "original_action": np.asarray(demo["actions"][int(perturb_t)]),
                                "perturbed_action": np.asarray(perturbed["actions"][int(perturb_t)]),
                                "action_delta_norm": float(np.linalg.norm(np.asarray(perturbed["actions"][int(perturb_t)]) - np.asarray(demo["actions"][int(perturb_t)]))),
                                "control_freq": int(env_args.get("env_kwargs", {}).get("control_freq", 20)),
                                "episode_length": int(len(perturbed["actions"])),
                                "gap_threshold": float(label.get("gap_threshold", -1.0)),
                                "failure_onset": label["failure_onset"],
                                "failure_type": label["failure_type"],
                                "responsible_t": int(label["responsible_t"]),
                                "responsible_start": int(label["responsible_start"]),
                                "responsible_end": int(label["responsible_end"]),
                                "recovery_start": label["recovery_start"],
                                "recovery_end": label["recovery_end"],
                                "final_success_clean": bool(label["final_success_clean"]),
                                "final_success_perturbed": bool(label["final_success_perturbed"]),
                                "render_flip_applied": bool(render_flip),
                                "label_status": label["label_status"],
                            }
                            _write_metadata_line(meta_f, record)
                            pair_count += 1
        data.attrs["total"] = int(len(data.keys()))
        env.close()
    return pair_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--tasks", nargs="+", required=True)
    p.add_argument("--episodes-per-task", type=int, required=True)
    p.add_argument("--interventions-per-episode", type=int, default=4)
    p.add_argument("--perturbation-types", nargs="*", default=PERTURBATION_TYPES)
    p.add_argument("--candidate-magnitudes", nargs="*", type=float, default=[0.25, 0.5, 0.75])
    p.add_argument("--magnitude-config", default=None)
    p.add_argument("--mode", choices=["calibration", "full"], default="full")
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--output-hdf5", required=True)
    p.add_argument("--output-metadata", required=True)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--max-extra-pairs", type=int, default=100)
    p.add_argument("--demo-start", type=int, default=0)
    p.add_argument("--demo-end", type=int, default=None)
    args = p.parse_args()

    if len(args.datasets) not in {1, len(args.tasks)}:
        raise SystemExit("--datasets must contain either one path or the same number of paths as --tasks")

    Path(args.output_hdf5).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_metadata).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.output_metadata).exists():
        Path(args.output_metadata).unlink()
    if Path(args.output_hdf5).exists():
        Path(args.output_hdf5).unlink()
    config = _load_magnitude_config(args.magnitude_config)

    total_pairs = 0
    for task in args.tasks:
        dataset_path = _dataset_for_task(args.datasets, args.tasks, task)
        total_pairs += _episode_records(
            task=task,
            dataset_path=dataset_path,
            mode=args.mode,
            episodes_per_task=args.episodes_per_task,
            interventions_per_episode=args.interventions_per_episode,
            perturbation_types=args.perturbation_types,
            magnitude_config=config,
            candidate_magnitudes=args.candidate_magnitudes,
            seed=args.seed + (args.tasks.index(task) * 10000),
            output_hdf5=args.output_hdf5,
            output_metadata=Path(args.output_metadata),
            demo_start=args.demo_start,
            demo_end=args.demo_end,
        )

    print(json.dumps({"mode": args.mode, "pair_count": total_pairs, "output_hdf5": args.output_hdf5, "output_metadata": args.output_metadata}, indent=2))


if __name__ == "__main__":
    main()
