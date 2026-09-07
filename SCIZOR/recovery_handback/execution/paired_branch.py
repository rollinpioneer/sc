"""Run one HB1 branch from a canonical root and fixed anchor."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from recovery_handback.adapters.base_policy import BasePolicyAdapter
from recovery_handback.adapters.env_adapter import EnvAdapter, load_observation_spec
from recovery_handback.adapters.repair_policy import build_repair_policy
from recovery_handback.common import atomic_json_dump, flatten_numeric, sha256_json
from recovery_handback.execution.label_reference import apply_success_labels, helper_active


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


def _load_observation_tape(rollout, states_pre: np.ndarray) -> dict:
    """Load root-time camera inputs keyed by the exact pre-action state."""
    image_keys = sorted(
        key.removeprefix("policy_obs__")
        for key in rollout.files
        if key.startswith("policy_obs__")
    )
    if not image_keys:
        raise RuntimeError("root rollout has no persistent policy observation tape")
    images = {
        key: np.asarray(rollout[f"policy_obs__{key}"])
        for key in image_keys
    }
    expected = len(states_pre)
    mismatched = {
        key: int(value.shape[0])
        for key, value in images.items()
        if value.ndim < 1 or value.shape[0] != expected
    }
    if mismatched:
        raise RuntimeError(
            f"policy observation tape length mismatch: expected={expected}, actual={mismatched}"
        )
    tape = {}
    for index, state in enumerate(states_pre):
        state_key = np.asarray(state, dtype=np.float64).tobytes(order="C")
        tape.setdefault(state_key, {
            key: value[index].copy() for key, value in images.items()
        })
    return tape


def _base_record(policy_pair: dict) -> dict:
    base = policy_pair.get("base", policy_pair)
    if "checkpoint" not in base:
        raise ValueError("policy pair has no base checkpoint")
    return base


def _repair_record(policy_pair: dict) -> dict:
    repair = policy_pair.get("repair")
    if not isinstance(repair, dict) or not repair.get("checkpoint"):
        raise ValueError("policy pair has no frozen repair checkpoint")
    return repair


def _memory_vector(snapshot: dict) -> np.ndarray:
    parts = []
    hidden = snapshot.get("_rnn_hidden_state")
    if hidden is not None:
        tensors = hidden if isinstance(hidden, (tuple, list)) else (hidden,)
        parts.extend(flatten_numeric(tensor) for tensor in tensors)
    if "_rnn_counter" in snapshot:
        parts.append(flatten_numeric(snapshot["_rnn_counter"]))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _expected_memory(path: Path, anchor_t: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as handle:
        keys = sorted(
            (key for key in handle.files if key.startswith(f"{anchor_t}_hidden_")),
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )
        parts = [np.asarray(handle[key], dtype=np.float64).reshape(-1) for key in keys]
        counter_key = f"{anchor_t}_counter"
        if counter_key in handle:
            parts.append(np.asarray(handle[counter_key], dtype=np.float64).reshape(-1))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right))) if left.size else 0.0


def _store_history(frames: list[dict], path: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for index, frame in enumerate(frames):
        keys = []
        for key, value in frame["obs"].items():
            arrays[f"{index}_{key}"] = np.asarray(value)
            keys.append(key)
        arrays[f"{index}_base_action"] = np.asarray(frame["base_action"], dtype=np.float32)
        arrays[f"{index}_action"] = np.asarray(frame["action"], dtype=np.float32)
        arrays[f"{index}_base_memory"] = flatten_numeric(frame.get("base_memory", {}))
        meta.append({
            "absolute_t": frame["absolute_t"], "helper_active": frame["helper_active"],
            "keys": keys, "memory_capture_phase": "after_suggest",
        })
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


class ZeroRepair:
    action_mode = "residual"
    preserve_base_action = True

    def start_episode(self, horizon_steps=400):
        self.horizon_steps = int(horizon_steps)

    def residual(self, raw_obs, privileged_state, base_action, base_memory,
                 remaining_steps, root_key, absolute_t):
        return np.zeros_like(np.asarray(base_action, dtype=np.float32))

    def observe_executed_action(self, action):
        return None


def build_repair_adapter(policy_pair: dict, device: str = "cuda"):
    repair = _repair_record(policy_pair)
    return build_repair_policy(repair, device=device)


def run_branch(config: dict, anchor: dict, policy_pair: dict, repair_length: int | str,
               output_path: str | Path, *, repair_override=None, device: str = "cuda",
               base_device: str | None = None, repair_device: str | None = None,
               observation_tape: dict | None = None) -> dict:
    """Run one 0/5/20/80/full branch and atomically save its result."""
    started = time.time()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_path.with_suffix(".npz")
    handoff_history_path = output_path.with_name(output_path.stem + "_handoff.npz")
    anchor_t = int(anchor["anchor_t"])
    horizon = int(config["horizon_steps"])
    root_key = str(anchor["root_id"])
    base_device = base_device or device
    repair_device = repair_device or device
    assets = json.loads(Path(config["assets_file"]).read_text(encoding="utf-8"))
    source = Path(assets["tasks"][anchor["task"]]["source_hdf5"])
    base_record = _base_record(policy_pair)
    rollout_path = Path(anchor["rollout_path"])
    with np.load(rollout_path, allow_pickle=False) as rollout:
        saved_actions = np.asarray(rollout["actions"], dtype=np.float32)
        saved_suggestions = np.asarray(rollout["suggestions"], dtype=np.float32)
        saved_states_pre = np.asarray(rollout["states_pre"], dtype=np.float64)
        persisted_tape = _load_observation_tape(rollout, saved_states_pre)
    if observation_tape is None:
        observation_tape = {}
    for state_key, images in persisted_tape.items():
        observation_tape.setdefault(state_key, images)
    env = EnvAdapter(
        source, **load_observation_spec(source), observation_tape=observation_tape
    )
    base = BasePolicyAdapter(base_record["checkpoint"], device=base_device)
    repair = repair_override
    if repair is None and repair_length not in (0, "0"):
        repair = build_repair_adapter(policy_pair, device=repair_device)
    if repair is not None:
        repair.start_episode(horizon)

    result: dict[str, Any] = {
        "schema_version": "hb1_branch_v1",
        "task": anchor["task"], "role": anchor.get("role", "probe"),
        "cohort": anchor.get("cohort", "main"), "root_id": root_key,
        "stat_group_id": anchor.get("stat_group_id", root_key),
        "anchor_id": anchor["anchor_id"], "anchor_t": anchor_t,
        "branch_name": "full" if repair_length == "full" else ("none" if int(repair_length) == 0 else f"l{int(repair_length)}"),
        "repair_length": repair_length, "horizon_steps": horizon,
        "control_freq": int(config.get("expected_control_freq", env.control_freq)),
        "base_device": base_device, "repair_device": repair_device,
        "observation_tape_enabled": True,
        "observation_tape_source": "rollout_policy_obs_by_exact_state_v1",
        "observation_tape_persisted_states": len(persisted_tape),
        "observation_tape_hits": 0, "observation_tape_misses": 0,
        "policy_pair_hash": sha256_json(policy_pair), "config_hash": sha256_json(config),
        "engineering_ok": False, "exception_reason": None,
        "prefix_env_steps": 0, "continuation_env_steps": 0,
        "base_policy_calls": 0, "repair_policy_calls": 0,
        "repair_calls_after_handoff": 0, "helper_steps_actual": 0,
        "changed_action_steps": 0, "handoff_executed": False,
        "handoff_state_index": None, "first_raw_success_state": None,
        "stable_success_state": None, "success_seen_under_helper": False,
        "helper_completed_task": False, "system_success": False,
        "first_success_wait_after_handoff": None,
        "autonomous_execution_steps_after_handoff": None,
        "episode_end_state_index": anchor_t, "end_reason": "exception",
        "raw_return": 0.0, "prefix_state_max_abs": float("inf"),
        "prefix_action_max_abs": float("inf"), "memory_max_abs": float("inf"),
        "trajectory_path": str(trajectory_path.resolve()),
        "anchor_history_path": anchor.get("anchor_history_path"),
        "handoff_history_path": None, "memory_capture_phase": "before_anchor_suggest",
        "handoff_memory_capture_phase": None,
    }
    actions: list[np.ndarray] = []
    base_actions: list[np.ndarray] = []
    helper_mask: list[bool] = []
    states: list[np.ndarray] = []
    successes: list[bool] = []
    rewards: list[float] = []
    recent = deque(maxlen=int(config["outputs"]["store_anchor_and_handoff_history"]))
    stable = deque(maxlen=int(config["success_consecutive_steps"]))
    try:
        payload = _load_payload(Path(anchor["canonical_payload_path"]))
        obs = env.reset_canonical(payload, payload["seed"])
        base.start_episode()
        prefix_action_error = 0.0
        prefix_state_error = 0.0
        for t in range(anchor_t):
            current_state = env.physical_state()
            prefix_state_error = max(prefix_state_error, _max_abs(current_state, saved_states_pre[t]))
            suggestion = base.suggest_once(obs, t, root_key)
            prefix_action_error = max(prefix_action_error, _max_abs(suggestion, saved_suggestions[t]))
            obs, _reward, _success, _info = env.step(saved_actions[t])
            result["prefix_env_steps"] += 1
        result["prefix_action_max_abs"] = prefix_action_error
        result["prefix_state_max_abs"] = max(
            prefix_state_error, _max_abs(env.physical_state(), saved_states_pre[anchor_t])
        )
        actual_memory = _memory_vector(base.memory_snapshot())
        expected_memory = _expected_memory(Path(anchor["policy_memory_check_path"]), anchor_t)
        result["memory_max_abs"] = _max_abs(actual_memory, expected_memory)
        limits = config["engineering"]
        if result["prefix_state_max_abs"] > float(limits["prefix_state_max_abs"]):
            raise RuntimeError(f"prefix state mismatch: {result['prefix_state_max_abs']}")
        if result["prefix_action_max_abs"] > float(limits["prefix_action_max_abs"]):
            raise RuntimeError(f"prefix action mismatch: {result['prefix_action_max_abs']}")
        if result["memory_max_abs"] > float(limits["policy_memory_max_abs"]):
            raise RuntimeError(f"policy memory mismatch: {result['memory_max_abs']}")

        states.append(env.physical_state())
        low, high = env.action_bounds()
        scale = np.asarray(config["repair_training"]["residual_scale"], dtype=np.float32)
        history_keep = int(config["outputs"]["store_anchor_and_handoff_history"])
        for t in range(anchor_t, horizon):
            base_action = base.suggest_once(obs, t, root_key)
            base_memory = base.memory_snapshot()
            active = helper_active(repair_length, anchor_t, t)
            if active:
                if repair is None:
                    raise RuntimeError("helper branch requested without a repair policy")
                privileged_state = env.privileged_features()
                if getattr(repair, "requires_stage_rewards", False):
                    privileged_state["__stage_rewards__"] = env.staged_rewards()
                proposal = repair.residual(
                    obs, privileged_state, base_action, base_memory,
                    horizon - t, root_key, t,
                )
                result["repair_policy_calls"] += 1
                result["helper_steps_actual"] += 1
                if getattr(repair, "preserve_base_action", False):
                    action = np.asarray(base_action, dtype=np.float32).copy()
                elif getattr(repair, "action_mode", "residual") == "direct":
                    action = np.clip(proposal, low, high).astype(np.float32)
                else:
                    action = np.clip(base_action + scale * proposal, low, high).astype(np.float32)
            else:
                action = np.asarray(base_action, dtype=np.float32)
            obs_before = {
                key: np.asarray(value).copy() for key, value in obs.items()
                if key.endswith("_image") or key in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
            }
            current_frame = {
                "absolute_t": t, "helper_active": bool(active), "obs": obs_before,
                "base_action": base_action, "action": action, "base_memory": base_memory,
            }
            if repair_length not in (0, "full") and t == anchor_t + int(repair_length):
                result["handoff_executed"] = True
                result["handoff_state_index"] = t
                result["handoff_memory_capture_phase"] = "after_suggest_before_action"
                _store_history((list(recent) + [current_frame])[-history_keep:], handoff_history_path)
                result["handoff_history_path"] = str(handoff_history_path.resolve())
            if result["handoff_executed"] and active:
                result["repair_calls_after_handoff"] += 1
            if not np.allclose(action, base_action, atol=1e-7, rtol=0.0):
                result["changed_action_steps"] += 1
            obs_next, reward, raw_success, _info = env.step(action)
            if repair is not None and hasattr(repair, "observe_executed_action"):
                repair.observe_executed_action(action)
            result["continuation_env_steps"] += 1
            result["raw_return"] += float(reward)
            next_state = t + 1
            if raw_success and result["first_raw_success_state"] is None:
                result["first_raw_success_state"] = next_state
            if raw_success and active:
                result["success_seen_under_helper"] = True
            stable.append(bool(raw_success))
            actions.append(action.copy())
            base_actions.append(np.asarray(base_action, dtype=np.float32).copy())
            helper_mask.append(bool(active))
            rewards.append(float(reward))
            successes.append(bool(raw_success))
            states.append(env.physical_state())
            recent.append(current_frame)
            result["episode_end_state_index"] = next_state
            if len(stable) == stable.maxlen and all(stable):
                result["stable_success_state"] = next_state
                result["system_success"] = True
                result["helper_completed_task"] = bool(active)
                result["end_reason"] = "stable_success"
                break
            obs = obs_next
        else:
            result["end_reason"] = "deadline"
        result["base_policy_calls"] = base._calls
        result["engineering_ok"] = True
    except Exception as exc:
        result["exception_reason"] = f"{type(exc).__name__}: {exc}"
        result["base_policy_calls"] = base._calls
    finally:
        result["observation_tape_hits"] = env.observation_tape_hits
        result["observation_tape_misses"] = env.observation_tape_misses
        env.close()
    result["wall_seconds"] = time.time() - started
    apply_success_labels(result, int(config["minimum_autonomous_steps"]))
    np.savez_compressed(
        trajectory_path,
        actions=np.asarray(actions, dtype=np.float32),
        base_actions=np.asarray(base_actions, dtype=np.float32),
        helper_mask=np.asarray(helper_mask, dtype=np.bool_),
        states=np.asarray(states, dtype=np.float64),
        success=np.asarray(successes, dtype=np.bool_),
        rewards=np.asarray(rewards, dtype=np.float32),
    )
    atomic_json_dump(result, output_path)
    return result
