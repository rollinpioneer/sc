"""Run fixed L60/L80 and learned stop/continue episodes from t=0."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_file
from recovery_handback.execution.label_reference import apply_success_labels
from recovery_handback.execution.paired_branch import build_repair_adapter
from recovery_handback.hb3p_stop.model import load_predictor


def _load_payload(path: Path) -> dict:
    stem = Path(path).with_suffix("")
    with np.load(path, allow_pickle=False) as handle:
        states = np.asarray(handle["states"], dtype=np.float64)
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return {"states": states, "model": stem.with_suffix(".xml").read_text(encoding="utf-8"), "seed": int(meta["seed"])}


def _observation_tape(rollout_path: Path) -> dict:
    with np.load(rollout_path, allow_pickle=False) as handle:
        states = np.asarray(handle["states_pre"], dtype=np.float64)
        image_keys = [key.removeprefix("policy_obs__") for key in handle.files if key.startswith("policy_obs__")]
        images = {key: np.asarray(handle[f"policy_obs__{key}"]) for key in image_keys}
    if not images or any(len(value) != len(states) for value in images.values()):
        raise ValueError("invalid persistent observation tape")
    tape = {}
    for index, state in enumerate(states):
        tape.setdefault(state.tobytes(order="C"), {key: value[index].copy() for key, value in images.items()})
    return tape


def _obs_frame(obs: dict) -> dict:
    allowed = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    return {key: np.asarray(obs[key]).copy() for key in allowed}


def _save_handoff(history: list[dict], path: Path) -> None:
    arrays = {}
    meta = []
    for index, frame in enumerate(history[-4:]):
        for key, value in frame["obs"].items():
            arrays[f"{index}_{key}"] = np.asarray(value)
        arrays[f"{index}_base_action"] = np.asarray(frame["base_action"], dtype=np.float32)
        meta.append({"absolute_t": int(frame["absolute_t"]), "memory_capture_phase": "after_suggest_before_action"})
    if len(meta) != 4:
        raise RuntimeError("handoff history requires exactly four frames")
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _method_spec(method: str) -> tuple[str, int | None]:
    if method == "NONE":
        return "NONE", None
    if method == "FIXED_L60":
        return "STOP", 60
    if method == "FIXED_L80":
        return "CONTINUE", 80
    if method == "LEARNED_STOP_CONTINUE":
        return "LEARNED", None
    raise ValueError(f"unknown stop/continue method: {method}")


def _validate(result: dict, helper_mask: list[bool]) -> None:
    end = int(result["episode_end_state_index"])
    if int(result["base_policy_calls"]) != end:
        raise RuntimeError("base policy was not called exactly once per executed environment step")
    if int(result["repair_policy_calls"]) != int(result["helper_steps_actual"]):
        raise RuntimeError("repair calls disagree with helper steps")
    if int(result["repair_calls_after_handoff"]) != 0:
        raise RuntimeError("repair was called after permanent handback")
    if int(result["takeover_count"]) not in (0, 1):
        raise RuntimeError("more than one takeover")
    if not result["takeover_count"]:
        if any(helper_mask) or result["handoff_t"] is not None or int(result["selected_length"]) != 0:
            raise RuntimeError("helper activity exists without takeover")
        return
    if result["selected_length"] not in (60, 80):
        raise RuntimeError("takeover has no frozen stop/continue length")
    start = int(result["takeover_t"])
    expected = list(range(start, min(start + int(result["selected_length"]), end)))
    active = [index for index, value in enumerate(helper_mask) if value]
    if active != expected:
        raise RuntimeError("helper ownership is not one exact interval")
    if result["handoff_executed"] and int(result["handoff_t"]) != start + int(result["selected_length"]):
        raise RuntimeError("handoff time disagrees with selected length")


class OnlineRunner:
    def __init__(self, config_path: Path, protocol_path: Path, *, device: str = "cuda",
                 base_device: str | None = None):
        self.config_path = Path(config_path)
        self.protocol_path = Path(protocol_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.protocol_hash = sha256_file(self.protocol_path)
        self.pair = json.loads(Path(self.config["policy_pair_path"]).read_text(encoding="utf-8"))
        self.base = BasePolicyAdapter(
            self.pair["base"]["checkpoint"],
            device=base_device or device,
        )
        self.repair = build_repair_adapter(self.pair, device=device)
        model = self.protocol["model"]
        predictor_device = device if device == "cuda" else "cpu"
        self.predict_stop_continue = load_predictor(Path(model["checkpoint"]), Path(model["normalizer"]), device=predictor_device)

    def run(self, root: dict, method: str, output_path: Path) -> dict:
        started = time.time()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_path.with_suffix(".npz")
        trace_path = output_path.with_name(output_path.stem + "_decision_trace.json")
        handoff_path = output_path.with_name(output_path.stem + "_handoff.npz")
        kind, fixed_length = _method_spec(method)
        root_id = str(root["root_id"])
        source = Path(self.config["assets_file"])
        assets = json.loads(source.read_text(encoding="utf-8"))
        source_hdf5 = Path(assets["tasks"]["square"]["source_hdf5"])
        env = EnvAdapter(source_hdf5, **load_observation_spec(source_hdf5), observation_tape=_observation_tape(Path(root["rollout_path"])))
        result = {
            "schema_version": "hb3p_stop_continue_online_episode_v1",
            "task": "square", "role": str(root.get("role", self.config["role"])), "root_id": root_id,
            "stat_group_id": str(root.get("stat_group_id", root_id)), "method_id": method,
            "protocol_hash": self.protocol_hash, "semantic_pair_id": self.protocol["semantic_pair_id"],
            "engineering_ok": False, "exception_reason": None,
            "system_success": False, "first_raw_success_state": None, "stable_success_state": None,
            "takeover_count": 0, "takeover_t": None, "selected_length": 0, "repair_length": 0,
            "handoff_t": None, "handoff_state_index": None, "handoff_executed": False,
            "success_seen_under_helper": False, "helper_completed_task": False,
            "helper_steps_actual": 0, "changed_action_steps": 0, "repair_policy_calls": 0,
            "repair_calls_after_handoff": 0, "base_policy_calls": 0, "episode_end_state_index": 0,
            "decision_t": None, "stop_continue_decision": None, "continue_score": None,
            "continue_threshold": None, "query_count": 0, "queried_times": [],
            "inference_wall_seconds": 0.0, "simulation_wall_seconds": 0.0, "wall_seconds": 0.0,
            "root_seed": int(root["seed"]), "initial_state_hash": root.get("initial_state_hash"),
            "checkpoint_sha256": {"base": self.pair["base"]["checkpoint_sha256"], "repair": self.pair["repair"]["checkpoint_sha256"], "selector": self.protocol["model"]["checkpoint_sha256"] if method == "LEARNED_STOP_CONTINUE" else None},
            "trajectory_path": str(trajectory_path.resolve()), "decision_trace_path": str(trace_path.resolve()),
            "handoff_history_path": None, "end_reason": "exception", "raw_return": 0.0,
        }
        actions, base_actions, helper_mask, states, successes, rewards, trace = [], [], [], [], [], [], []
        recent = deque(maxlen=4)
        stable = deque(maxlen=5)
        planned_length = fixed_length
        planned_end = 20 + fixed_length if fixed_length is not None else None
        if kind == "LEARNED":
            # Helper ownership is active from t=20 while the only selector
            # query is deferred until t=80. The provisional upper bound is
            # replaced by the frozen STOP/CONTINUE result at that time.
            planned_length = 80
            planned_end = 100
        inference_seconds = 0.0
        simulation_started = time.perf_counter()
        try:
            payload = _load_payload(Path(root["canonical_payload_path"]))
            obs = env.reset_canonical(payload, payload["seed"])
            self.base.start_episode()
            self.repair.start_episode(int(self.protocol["horizon_steps"]))
            states.append(env.physical_state())
            low, high = env.action_bounds()
            scale = np.asarray(self.pair.get("residual_scale", self.config.get("residual_scale", [0.2] * 6 + [2.0])), dtype=np.float32)
            for t in range(int(self.protocol["horizon_steps"])):
                base_action = self.base.suggest_once(obs, t, root_id)
                base_memory = self.base.memory_snapshot()
                frame = {"absolute_t": t, "obs": _obs_frame(obs), "base_action": base_action.copy()}
                recent.append(frame)
                if kind == "LEARNED" and t == 80 and result["takeover_count"] == 1 and result["stop_continue_decision"] is None:
                    predict_started = time.perf_counter()
                    prediction = self.predict_stop_continue(list(recent), t, int(self.protocol["horizon_steps"]))
                    inference_seconds += time.perf_counter() - predict_started
                    result.update({
                        "decision_t": 80, "stop_continue_decision": prediction["decision"],
                        "continue_score": prediction["continue_score"], "continue_threshold": prediction["threshold"],
                        "query_count": 1, "queried_times": [80],
                    })
                    planned_length = 80 if prediction["decision"] == "CONTINUE" else 60
                    planned_end = 20 + planned_length
                    result["selected_length"] = int(planned_length)
                    result["repair_length"] = int(planned_length)
                    trace.append({"absolute_t": 80, "model": "STOP_CONTINUE_MLP", **prediction})
                if t == 20 and result["first_raw_success_state"] is None and kind != "NONE":
                    result["takeover_count"] = 1
                    result["takeover_t"] = 20
                    if planned_length is not None:
                        result["selected_length"] = int(planned_length)
                        result["repair_length"] = int(planned_length)
                if result["takeover_count"] and planned_length is not None and t == planned_end:
                    result["selected_length"] = int(planned_length)
                    result["repair_length"] = int(planned_length)
                    result["handoff_t"] = t
                    result["handoff_state_index"] = t
                    result["handoff_executed"] = True
                    _save_handoff(list(recent), handoff_path)
                    result["handoff_history_path"] = str(handoff_path.resolve())
                active = bool(result["takeover_count"] and planned_end is not None and t < planned_end)
                if active:
                    privileged = env.privileged_features()
                    if getattr(self.repair, "requires_stage_rewards", False):
                        privileged["__stage_rewards__"] = env.staged_rewards()
                    proposal = self.repair.residual(obs, privileged, base_action, base_memory,
                                                    int(self.protocol["horizon_steps"]) - t, root_id, t)
                    result["repair_policy_calls"] += 1
                    result["helper_steps_actual"] += 1
                    if getattr(self.repair, "preserve_base_action", False):
                        action = base_action.copy()
                    elif getattr(self.repair, "action_mode", "residual") == "direct":
                        action = np.clip(proposal, low, high).astype(np.float32)
                    else:
                        action = np.clip(base_action + scale * proposal, low, high).astype(np.float32)
                else:
                    action = base_action.copy()
                if result["handoff_executed"] and active:
                    result["repair_calls_after_handoff"] += 1
                if not np.allclose(action, base_action, atol=1e-7, rtol=0.0):
                    result["changed_action_steps"] += 1
                obs_next, reward, raw_success, _ = env.step(action)
                if hasattr(self.repair, "observe_executed_action"):
                    self.repair.observe_executed_action(action)
                next_state = t + 1
                if raw_success and result["first_raw_success_state"] is None:
                    result["first_raw_success_state"] = next_state
                if raw_success and active:
                    result["success_seen_under_helper"] = True
                stable.append(bool(raw_success))
                actions.append(action.copy()); base_actions.append(base_action.copy()); helper_mask.append(active)
                successes.append(bool(raw_success)); rewards.append(float(reward)); states.append(env.physical_state())
                result["raw_return"] += float(reward); result["episode_end_state_index"] = next_state
                if len(stable) == stable.maxlen and all(stable):
                    result["stable_success_state"] = next_state
                    result["system_success"] = True
                    result["helper_completed_task"] = active
                    result["end_reason"] = "stable_success"
                    break
                obs = obs_next
            else:
                result["end_reason"] = "deadline"
            if result["takeover_count"] and result["selected_length"] == 0 and planned_length is not None:
                result["selected_length"] = int(planned_length); result["repair_length"] = int(planned_length)
            result["base_policy_calls"] = self.base._calls
            _validate(result, helper_mask)
            result["engineering_ok"] = True
        except Exception as exc:
            result["exception_reason"] = f"{type(exc).__name__}: {exc}"
            result["base_policy_calls"] = self.base._calls
        finally:
            result["observation_tape_hits"] = env.observation_tape_hits
            result["observation_tape_misses"] = env.observation_tape_misses
            env.close()
        result["inference_wall_seconds"] = inference_seconds
        result["simulation_wall_seconds"] = time.perf_counter() - simulation_started - inference_seconds
        result["wall_seconds"] = time.time() - started
        apply_success_labels(result, int(self.protocol["minimum_autonomous_steps"]))
        np.savez_compressed(trajectory_path,
                            actions=np.asarray(actions, np.float32), base_actions=np.asarray(base_actions, np.float32),
                            helper_mask=np.asarray(helper_mask, np.bool_), states=np.asarray(states, np.float64),
                            success=np.asarray(successes, np.bool_), rewards=np.asarray(rewards, np.float32))
        atomic_json_dump({"schema_version": "hb3p_stop_continue_decision_trace_v1", "records": trace}, trace_path)
        atomic_json_dump(result, output_path)
        return result
