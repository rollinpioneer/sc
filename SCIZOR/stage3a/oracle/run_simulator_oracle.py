"""Run local counterfactual rollouts against the frozen benchmark simulator."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark-hdf5", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--state-alignment", type=Path)
    p.add_argument("--state-alignment-output", type=Path)
    p.add_argument("--max-queries", type=int)
    p.add_argument("--query-offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _safe(v):
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    return v


class Worker:
    def __init__(self, benchmark: str):
        self.benchmark = benchmark
        self.envs = {}
        self.h5 = h5py.File(benchmark, "r")
        self.groups = {}
        self.reference = {}

    def env(self, task: str):
        if task in self.envs: return self.envs[task]
        # Import the repository's MuJoCo 2/3-compatible wrapper explicitly;
        # the simulator environment may be installed under a different name.
        from robomimic.utils import obs_utils
        from robomimic.envs.env_robosuite import EnvRobosuite
        if obs_utils.OBS_KEYS_TO_MODALITIES is None:
            obs_utils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": ["robot0_eef_pos"], "rgb": []}})
        dataset = "/data/" + ("can_demo_0_clean" if task == "can" else "square_demo_0_clean")
        with h5py.File(self.benchmark, "r") as f:
            # Benchmark attrs contain the canonical source path. Fall back to
            # the configured task dataset only for malformed legacy rows.
            source = f[dataset].attrs.get("source_dataset", "")
            if isinstance(source, bytes): source = source.decode()
            source = source or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(self.benchmark))), "data", "robomimic", task, "ph", "image.hdf5")
        with h5py.File(source, "r") as source_h5:
            meta = json.loads(source_h5["data"].attrs["env_args"])
        kw = dict(meta["env_kwargs"])
        kw.update(has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, camera_names=[])
        kw.pop("camera_depths", None)
        kw["reward_shaping"] = True
        env_name = "PickPlaceCan" if task == "can" else "NutAssemblySquare"
        # The benchmark metadata is authoritative for the env name when it is
        # available; square's dataset attrs use the same registered name.
        with h5py.File(source, "r") as source_h5:
            env_name = str(json.loads(source_h5["data"].attrs["env_args"]).get("env_name", env_name))
        self.envs[task] = EnvRobosuite(env_name=env_name, render=False, render_offscreen=False, use_image_obs=False, **kw)
        return self.envs[task]

    def close(self):
        for e in self.envs.values():
            try: e.close()
            except Exception: pass
        self.h5.close()

    def group_data(self, group: str):
        if group not in self.groups:
            g = self.h5[group]
            self.groups[group] = {"states": g["states"][:].astype(np.float32), "actions": g["actions"][:].astype(np.float32)}
        return self.groups[group]

    def rollout(self, task: str, states: np.ndarray, actions: np.ndarray, start: int, first_action: np.ndarray, end: int):
        env = self.env(task)
        env.reset_to({"states": states[int(start)]})
        rewards, staged, success, actual_states = [], [], [], []
        stage_fn = getattr(env.env, "staged_rewards", None)
        for action in np.concatenate([np.asarray(first_action, dtype=np.float32)[None], actions[int(start) + 1:int(end)]], axis=0):
            _, reward, _, _ = env.step(action)
            rewards.append(float(reward)); success.append(bool(env.is_success().get("task", False)))
            actual_states.append(np.asarray(env.get_state()["states"], dtype=np.float32).copy())
            staged.append(np.asarray(stage_fn(), dtype=np.float32).reshape(-1) if stage_fn is not None else np.zeros(1, dtype=np.float32))
        return {"rewards": np.asarray(rewards, dtype=np.float32), "staged": np.asarray(staged, dtype=np.float32), "success": np.asarray(success, dtype=bool), "states": np.asarray(actual_states, dtype=np.float32)}

    def one(self, row: dict):
        task, group, start = str(row["task"]), str(row["hdf5_group"]), int(row["query_t"])
        data = self.group_data(group); states, actions = data["states"], data["actions"]
        end = min(len(actions), start + 40)
        key = (task, group, start)
        ref = self.reference.get(key)
        if ref is None:
            ref = self.rollout(task, states, actions, start, actions[start], end); self.reference[key] = ref
        repl = self.rollout(task, states, actions, start, np.asarray(row["replacement_action"], dtype=np.float32), end)
        actual_horizon = min(len(ref["rewards"]), len(repl["rewards"]), 40)
        out = {k: row.get(k) for k in ("replacement_id", "query_id", "split", "pair_id", "task", "base_demo_id", "query_t", "query_source", "replacement_source", "oracle_only", "state_distance", "action_delta_l2", "state_in_domain", "action_in_domain")}
        out.update(actual_horizon=int(actual_horizon), reference_replay_ok=bool(len(ref["rewards"]) > 0), has_staged_rewards=bool(ref["staged"].shape[1] > 1 or np.any(ref["staged"])),)
        for h in (10, 20, 40):
            n = min(h, actual_horizon)
            if n <= 0:
                vals = {"dense_mean_delta": 0.0, "dense_return_delta": 0.0, "stage_mean_delta": 0.0, "stage_terminal_delta": 0.0, "success_delta": 0.0}
            else:
                dr = repl["rewards"][:n] - ref["rewards"][:n]
                rs = repl["staged"][:n].sum(axis=1); fs = ref["staged"][:n].sum(axis=1)
                vals = {"dense_mean_delta": float(dr.mean()), "dense_return_delta": float(dr.sum()), "stage_mean_delta": float((rs - fs).mean()), "stage_terminal_delta": float(rs[-1] - fs[-1]), "success_delta": float(repl["success"][:n].any()) - float(ref["success"][:n].any())}
            for name, value in vals.items(): out[f"{name}_h{h}"] = value
        return out


_WORKER = None
def _run_one(args):
    global _WORKER
    benchmark, row = args
    if _WORKER is None: _WORKER = Worker(benchmark)
    return _WORKER.one(row)


def _alignment(benchmark: Path, rows: list[dict], output: Path):
    output = Path(output)
    w = Worker(str(benchmark)); checks=[]
    try:
        # The plan is replacement-level, so the first two rows can belong to
        # the same query.  Select two distinct query IDs for the alignment
        # smoke as required by the Stage 3A-D protocol.
        seen_queries = set()
        alignment_rows = []
        for row in rows:
            qid = str(row.get("query_id"))
            if qid in seen_queries:
                continue
            seen_queries.add(qid)
            alignment_rows.append(row)
            if len(alignment_rows) >= 2:
                break
        for row in alignment_rows:
            data=w.group_data(str(row["hdf5_group"])); t=int(row["query_t"]); end=min(len(data["actions"]),t+1)
            if t+1>=len(data["states"]): continue
            rr=w.rollout(str(row["task"]),data["states"],data["actions"],t,data["actions"][t],end)
            err=float(np.linalg.norm(rr["states"][0]-data["states"][t+1]))
            checks.append({"query_id":row["query_id"],"task":row["task"],"query_t":t,"next_state_l2":err,"pass":bool(err<1e-4),"pre_action_state_index":t if err<1e-4 else None})
    finally: w.close()
    output.parent.mkdir(parents=True,exist_ok=True)
    median = float(np.median([x["next_state_l2"] for x in checks])) if checks else None
    output.write_text(json.dumps({
        "checks": checks,
        "median_l2": median,
        "all_pass": bool(checks and all(x["pass"] for x in checks)),
        "alignment_status": "passed" if checks and all(x["pass"] for x in checks) else "failed",
        "failure_reason": (
            "simulator next state did not reproduce benchmark states[t+1] "
            "within 1e-4; preserve recorded index and do not claim alignment"
        ) if not (checks and all(x["pass"] for x in checks)) else None,
    }, indent=2), encoding="utf-8")


def main():
    a=parse_args(); all_rows=[json.loads(x) for x in a.plan.read_text().splitlines() if x.strip()]
    grouped={}; order=[]
    for r in all_rows:
        qid=r["query_id"]
        if qid not in grouped: grouped[qid]=[]; order.append(qid)
        grouped[qid].append(r)
    start=max(0,int(a.query_offset)); stop=len(order) if a.max_queries is None else min(len(order),start+int(a.max_queries))
    rows=[r for qid in order[start:stop] for r in grouped[qid]]
    existing={}
    if a.resume and a.output.exists():
        for line in a.output.read_text().splitlines():
            if line.strip():
                r=json.loads(line); existing[r["replacement_id"]]=r
    todo=[r for r in rows if r["replacement_id"] not in existing]
    if a.state_alignment or a.state_alignment_output:
        _alignment(a.benchmark_hdf5, rows, a.state_alignment or a.state_alignment_output)
    if todo:
        # Keep each query's replacement rows together where possible so a
        # worker can reuse its reference rollout cache.
        chunk = max(1, min(64, len(todo) // max(1, int(a.workers))))
        with ProcessPoolExecutor(max_workers=max(1,int(a.workers))) as pool:
            for result in pool.map(_run_one, [(str(a.benchmark_hdf5), r) for r in todo], chunksize=chunk):
                existing[result["replacement_id"]]=result
                if a.resume:
                    a.output.parent.mkdir(parents=True, exist_ok=True)
                    with a.output.open("a") as append_handle:
                        append_handle.write(json.dumps({k:_safe(v) for k,v in result.items()},allow_nan=False)+"\n")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w") as f:
        for key in sorted(existing): f.write(json.dumps({k:_safe(v) for k,v in existing[key].items()},allow_nan=False)+"\n")
    print(json.dumps({"planned_rows":len(rows),"completed_rows":len(existing),"new_rows":len(todo),"output":str(a.output)},indent=2))

if __name__=="__main__": main()
