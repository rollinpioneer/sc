"""Execute one complete HB3-P episode from t=0 with live fixed-grid decisions."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.common import atomic_json_dump, sha256_file, sha256_json
from recovery_handback.execution.label_reference import apply_success_labels
from recovery_handback.execution.paired_branch import build_repair_adapter
from recovery_handback.hb2.metrics import choose_length
from recovery_handback.hb3p.controller import Rule, SingleIntervention
from recovery_handback.hb3p.predictors import M0Predictor, M1Predictor
from recovery_handback.hb3p.rpc_client import FileQueueClient


def _load_payload(path: Path) -> dict:
    stem = path.with_suffix("")
    with np.load(path, allow_pickle=False) as handle:
        states = np.asarray(handle["states"], dtype=np.float64)
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return {
        "states": states,
        "model": stem.with_suffix(".xml").read_text(encoding="utf-8"),
        "ep_meta": meta.get("ep_meta"),
        "seed": int(meta["seed"]),
    }


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


def _rule(method_id: str, protocol: dict) -> Rule:
    spec = protocol["methods"][method_id]
    if spec["kind"] == "none":
        return Rule("none")
    if spec["kind"] == "scheduled":
        return Rule("scheduled", scheduled_t=int(spec["scheduled_t"]), scheduled_length=int(spec["length"]))
    if spec["kind"] == "risk":
        return Rule("risk", model=spec["model"], risk_threshold=float(spec["threshold"]), risk_length=int(spec["length"]))
    return Rule("model", model=spec["model"], penalty=float(protocol["decision"]["primary_lambda"]))


def _obs_frame(obs: dict) -> dict:
    allowed = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    return {key: np.asarray(obs[key]).copy() for key in allowed}


def _save_handoff(history: list[dict], path: Path) -> None:
    arrays = {}
    meta = []
    for index, frame in enumerate(history[-4:]):
        for key, value in frame["obs"].items():
            arrays[f"{index}_{key}"] = np.asarray(value)
        arrays[f"{index}_base_action"] = np.asarray(frame["base_action"], np.float32)
        meta.append({"absolute_t": int(frame["absolute_t"]), "memory_capture_phase": "after_suggest_before_action"})
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _validate_execution(result: dict, helper_mask: list[bool]) -> None:
    end = int(result["episode_end_state_index"])
    if int(result["base_policy_calls"]) != end:
        raise RuntimeError("base policy was not called exactly once per executed environment step")
    if int(result["repair_policy_calls"]) != int(result["helper_steps_actual"]):
        raise RuntimeError("repair call count disagrees with helper step count")
    if int(result["repair_calls_after_handoff"]) != 0:
        raise RuntimeError("repair policy was called after permanent handback")
    if int(result["takeover_count"]) not in (0, 1):
        raise RuntimeError("more than one takeover was recorded")
    if int(result["query_count"]) != len(result["queried_times"]):
        raise RuntimeError("query count disagrees with decision trace")
    if any(int(t) not in (20, 80, 160) for t in result["queried_times"]):
        raise RuntimeError("selector was queried outside the frozen grid")
    active = [index for index, value in enumerate(helper_mask) if value]
    if not result["takeover_count"]:
        if active or result["takeover_t"] is not None or int(result["selected_length"]) != 0:
            raise RuntimeError("helper activity exists without a takeover")
        return
    takeover = int(result["takeover_t"])
    planned = int(result["selected_length"])
    expected = list(range(takeover, min(takeover + planned, end)))
    if active != expected:
        raise RuntimeError("helper ownership is not one exact contiguous frozen interval")
    if any(int(t) > takeover for t in result["queried_times"]):
        raise RuntimeError("selector was queried after the first takeover")
    if result["handoff_executed"]:
        if int(result["handoff_t"]) != takeover + planned:
            raise RuntimeError("handoff time disagrees with takeover plus fixed length")


class OnlineRunner:
    def __init__(self, config_path: Path, protocol_path: Path, queue_dir: Path | None, *, device: str = "cuda"):
        self.config_path = Path(config_path)
        self.protocol_path = Path(protocol_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.protocol_hash = sha256_file(self.protocol_path)
        pair_path = Path(self.config["hb2"]["selected_policy_pair_path"])
        self.pair = json.loads(pair_path.read_text(encoding="utf-8"))
        self.base = BasePolicyAdapter(self.pair["base"]["checkpoint"], device=device)
        self.repair = build_repair_adapter(self.pair, device=device)
        self.m0 = M0Predictor(self.protocol)
        self.m1 = M1Predictor(self.protocol, device=device)
        self.m4 = FileQueueClient(queue_dir, self.protocol_path) if queue_dir else None

    def _predict(self, model: str, history: list[dict], t: int, method_id: str, root_id: str) -> dict:
        if model == "M0_time":
            return self.m0.predict(history, t)
        if model == "M1_proprio":
            return self.m1.predict(history, t)
        if model == "M4_paired":
            if self.m4 is None:
                raise RuntimeError("M4 queue is not configured")
            return self.m4.predict(history, t, method_id, root_id)
        raise ValueError(f"unknown online predictor: {model}")

    def run(self, root: dict, method_id: str, output_path: Path) -> dict:
        started = time.time()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path = output_path.with_suffix(".npz")
        trace_path = output_path.with_name(output_path.stem + "_decision_trace.json")
        handoff_path = output_path.with_name(output_path.stem + "_handoff.npz")
        root_id = str(root["root_id"])
        controller = SingleIntervention(
            _rule(method_id, self.protocol),
            int(self.protocol["horizon_steps"]),
            tolerance=float(self.protocol["decision"]["tie_tolerance"]),
        )
        source = Path(json.loads(Path(self.config["assets_file"]).read_text())["tasks"]["square"]["source_hdf5"])
        env = EnvAdapter(source, **load_observation_spec(source), observation_tape=_observation_tape(Path(root["rollout_path"])))
        result = {
            "schema_version": "hb3p_online_episode_v1",
            "task": str(root.get("task", "square")),
            "role": str(root.get("role", root.get("exposure_role", "unknown"))),
            "root_id": root_id,
            "stat_group_id": str(root.get("stat_group_id") or root_id),
            "method_id": method_id,
            "protocol_hash": self.protocol_hash,
            "semantic_pair_id": self.protocol["semantic_pair_id"],
            "engineering_ok": False,
            "exception_reason": None,
            "system_success": False,
            "first_raw_success_state": None,
            "stable_success_state": None,
            "takeover_count": 0,
            "takeover_t": None,
            "selected_length": 0,
            "repair_length": 0,
            "handoff_t": None,
            "handoff_state_index": None,
            "handoff_executed": False,
            "success_seen_under_helper": False,
            "helper_completed_task": False,
            "helper_steps_actual": 0,
            "changed_action_steps": 0,
            "repair_policy_calls": 0,
            "repair_calls_after_handoff": 0,
            "base_policy_calls": 0,
            "episode_end_state_index": 0,
            "query_count": 0,
            "queried_times": [],
            "inference_wall_seconds": 0.0,
            "simulation_wall_seconds": 0.0,
            "root_seed": int(root["seed"]),
            "initial_state_hash": root.get("initial_state_hash"),
            "horizon_steps": int(self.protocol["horizon_steps"]),
            "checkpoint_sha256": {
                "base": self.pair["base"]["checkpoint_sha256"],
                "repair": self.pair["repair"]["checkpoint_sha256"],
                "selector": self.protocol["models"].get(self.protocol["methods"][method_id].get("model", ""), {}).get("checkpoint_sha256"),
            },
            "trajectory_path": str(trajectory_path.resolve()),
            "decision_trace_path": str(trace_path.resolve()),
            "handoff_history_path": None,
            "end_reason": "exception",
            "raw_return": 0.0,
        }
        actions = []
        base_actions = []
        helper_mask = []
        states = []
        successes = []
        rewards = []
        trace = []
        recent = deque(maxlen=4)
        stable = deque(maxlen=int(self.protocol["success_consecutive_steps"]))
        inference_seconds = 0.0
        simulation_started = time.perf_counter()
        try:
            payload = _load_payload(Path(root["canonical_payload_path"]))
            obs = env.reset_canonical(payload, payload["seed"])
            self.base.start_episode()
            self.repair.start_episode(int(self.protocol["horizon_steps"]))
            states.append(env.physical_state())
            low, high = env.action_bounds()
            for t in range(int(self.protocol["horizon_steps"])):
                base_action = self.base.suggest_once(obs, t, root_id)
                base_memory = self.base.memory_snapshot()
                frame = {"absolute_t": t, "obs": _obs_frame(obs), "base_action": base_action.copy()}
                recent.append(frame)
                probabilities = None
                if controller.needs_prediction(t, result["first_raw_success_state"] is not None):
                    model = self.protocol["methods"][method_id]["model"]
                    prediction_started = time.perf_counter()
                    probabilities = self._predict(model, list(recent), t, method_id, root_id)
                    elapsed = time.perf_counter() - prediction_started
                    inference_seconds += elapsed
                    trace.append({
                        "absolute_t": t,
                        "method_id": method_id,
                        "model": model,
                        "probabilities": {key: value for key, value in probabilities.items() if key.startswith("p")},
                        "reported_selected_length": probabilities.get("selected_length"),
                        "client_selected_length": probabilities.get("client_selected_length"),
                        "history_sha256": probabilities.get("history_sha256"),
                        "inference_seconds": elapsed,
                        "backbone_seconds": probabilities.get("backbone_seconds", 0.0),
                        "prediction_head_seconds": probabilities.get("prediction_head_seconds", 0.0),
                        "ipc_roundtrip_seconds": probabilities.get("ipc_roundtrip_seconds", 0.0),
                    })
                    if self.protocol["methods"][method_id]["kind"] == "model":
                        selected = choose_length(
                            probabilities,
                            float(self.protocol["decision"]["primary_lambda"]),
                            denominator=float(self.protocol["decision"]["cost_denominator"]),
                            tolerance=float(self.protocol["decision"]["tie_tolerance"]),
                        )
                        if int(probabilities.get("selected_length", -1)) != selected:
                            raise RuntimeError("predictor selected length disagrees with frozen formula")
                owner, event = controller.owner_before_action(
                    t, probabilities, result["first_raw_success_state"] is not None
                )
                if event == "takeover":
                    result["takeover_count"] = 1
                    result["takeover_t"] = t
                    result["selected_length"] = controller.length
                    result["repair_length"] = controller.length
                if event == "handoff":
                    result["handoff_t"] = t
                    result["handoff_state_index"] = t
                    result["handoff_executed"] = True
                    _save_handoff(list(recent), handoff_path)
                    result["handoff_history_path"] = str(handoff_path.resolve())
                if owner == "repair":
                    privileged = env.privileged_features()
                    if getattr(self.repair, "requires_stage_rewards", False):
                        privileged["__stage_rewards__"] = env.staged_rewards()
                    proposal = self.repair.residual(obs, privileged, base_action, base_memory, int(self.protocol["horizon_steps"]) - t, root_id, t)
                    result["repair_policy_calls"] += 1
                    result["helper_steps_actual"] += 1
                    if getattr(self.repair, "preserve_base_action", False):
                        action = base_action.copy()
                    elif getattr(self.repair, "action_mode", "residual") == "direct":
                        action = np.clip(proposal, low, high).astype(np.float32)
                    else:
                        scale = np.asarray(self.config["repair_training"]["residual_scale"], np.float32)
                        action = np.clip(base_action + scale * proposal, low, high).astype(np.float32)
                else:
                    action = base_action.copy()
                if result["handoff_executed"] and owner == "repair":
                    result["repair_calls_after_handoff"] += 1
                if not np.allclose(action, base_action, atol=1e-7, rtol=0.0):
                    result["changed_action_steps"] += 1
                obs_next, reward, raw_success, _ = env.step(action)
                if controller.takeovers and hasattr(self.repair, "observe_executed_action"):
                    self.repair.observe_executed_action(action)
                next_state = t + 1
                if raw_success and result["first_raw_success_state"] is None:
                    result["first_raw_success_state"] = next_state
                if raw_success and owner == "repair":
                    result["success_seen_under_helper"] = True
                stable.append(bool(raw_success))
                actions.append(action.copy())
                base_actions.append(base_action.copy())
                helper_mask.append(owner == "repair")
                successes.append(bool(raw_success))
                rewards.append(float(reward))
                result["raw_return"] += float(reward)
                states.append(env.physical_state())
                result["episode_end_state_index"] = next_state
                if len(stable) == stable.maxlen and all(stable):
                    result["stable_success_state"] = next_state
                    result["system_success"] = True
                    result["helper_completed_task"] = owner == "repair"
                    result["end_reason"] = "stable_success"
                    break
                obs = obs_next
            else:
                result["end_reason"] = "deadline"
            result["base_policy_calls"] = self.base._calls
            result["query_count"] = controller.query_count
            result["queried_times"] = [int(row["absolute_t"]) for row in trace]
            _validate_execution(result, helper_mask)
            result["engineering_ok"] = True
        except Exception as exc:
            result["exception_reason"] = f"{type(exc).__name__}: {exc}"
            result["base_policy_calls"] = self.base._calls
            result["query_count"] = controller.query_count
            result["queried_times"] = [int(row["absolute_t"]) for row in trace]
        finally:
            result["observation_tape_hits"] = env.observation_tape_hits
            result["observation_tape_misses"] = env.observation_tape_misses
            env.close()
        result["inference_wall_seconds"] = inference_seconds
        result["simulation_wall_seconds"] = time.perf_counter() - simulation_started - inference_seconds
        result["wall_seconds"] = time.time() - started
        apply_success_labels(result, int(self.protocol["minimum_autonomous_steps"]))
        np.savez_compressed(
            trajectory_path,
            actions=np.asarray(actions, np.float32),
            base_actions=np.asarray(base_actions, np.float32),
            helper_mask=np.asarray(helper_mask, np.bool_),
            states=np.asarray(states, np.float64),
            success=np.asarray(successes, np.bool_),
            rewards=np.asarray(rewards, np.float32),
        )
        atomic_json_dump({"schema_version": "hb3p_decision_trace_v1", "records": trace}, trace_path)
        atomic_json_dump(result, output_path)
        return result
