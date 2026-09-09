#!/usr/bin/env bash
set -Eeuo pipefail

OUT="${OUT:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_exit_probe_v1}"
WT="${WT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb3p-exit-probe-v1}"
CODE="${CODE:-$WT/SCIZOR}"
PY_SIM="${PY_SIM:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
CONFIG="$OUT/config/exit_probe.json"
PROTOCOL="$OUT/config/frozen_protocol.json"
ROOTS="$OUT/roots/hb3p_exit_probe"
EPISODES="$OUT/episodes"
STATUS="$OUT/status"
LOGS="$OUT/logs"
GPU_SIM0="${GPU_SIM0:-1}"
GPU_SIM1="${GPU_SIM1:-2}"

export PYTHONPATH="$CODE/robomimic:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY_SIM")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$STATUS" "$LOGS" "$EPISODES"
printf '%s\n' "$$" > "$STATUS/exit-probe-supervisor.pid"

stamp() { date '+%F %T %Z'; }
mark() { printf '%s %s\n' "$(stamp)" "$2" > "$STATUS/$1"; }
sim_pids=()
cleanup() {
  for pid in "${sim_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${sim_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
fail() {
  code=$?
  rm -f "$STATUS/exit-probe.running"
  printf '%s failed line=%s exit=%s\n' "$(stamp)" "$1" "$code" > "$STATUS/exit-probe.failed"
  exit "$code"
}
trap cleanup EXIT
trap 'fail $LINENO' ERR

test -f "$CONFIG"
test -f "$PROTOCOL"
rm -f "$STATUS/exit-probe.failed" "$STATUS/exit-probe.done"
mark exit-probe.running "pid=$$ sim_gpus=$GPU_SIM0,$GPU_SIM1"

"$PY_SIM" -m compileall -q "$CODE/recovery_handback/hb3p"
git -C "$WT" diff --check -- SCIZOR/recovery_handback/hb3p

mark exit-probe-collect.running "collect seeds 600000..600039 on gpu=$GPU_SIM0"
CUDA_VISIBLE_DEVICES="$GPU_SIM0" MUJOCO_EGL_DEVICE_ID="$GPU_SIM0" \
  "$PY_SIM" -m recovery_handback.hb3p.collect_test \
  --config "$CONFIG" --protocol "$PROTOCOL" --role hb3p_exit_probe --start-index 0 --count 40 \
  --output-dir "$ROOTS" --resume > "$LOGS/collect_exit_probe.log" 2>&1
mark exit-probe-collect.done "40 roots collected"

mark exit-probe-online.running "three fixed exits across two root shards"
CUDA_VISIBLE_DEVICES="$GPU_SIM0" MUJOCO_EGL_DEVICE_ID="$GPU_SIM0" \
  "$PY_SIM" -m recovery_handback.hb3p.run_online \
  --config "$CONFIG" --protocol "$PROTOCOL" --roots "$ROOTS/runtime_root_manifest.jsonl" \
  --num-shards 2 --shard-index 0 --methods FIXED_L40 FIXED_L60 FIXED_L80 \
  --output-dir "$EPISODES/shard0" --resume > "$LOGS/online_exit_probe0.log" 2>&1 &
sim_pids+=("$!")
CUDA_VISIBLE_DEVICES="$GPU_SIM1" MUJOCO_EGL_DEVICE_ID="$GPU_SIM1" \
  "$PY_SIM" -m recovery_handback.hb3p.run_online \
  --config "$CONFIG" --protocol "$PROTOCOL" --roots "$ROOTS/runtime_root_manifest.jsonl" \
  --num-shards 2 --shard-index 1 --methods FIXED_L40 FIXED_L60 FIXED_L80 \
  --output-dir "$EPISODES/shard1" --resume > "$LOGS/online_exit_probe1.log" 2>&1 &
sim_pids+=("$!")
printf '%s\n' "${sim_pids[@]}" > "$STATUS/exit-probe-online-workers.pid"
wait "${sim_pids[0]}"
wait "${sim_pids[1]}"
sim_pids=()
mark exit-probe-online.done "120 fixed-exit online records complete"

"$PY_SIM" -m recovery_handback.hb3p.aggregate \
  --config "$CONFIG" --protocol "$PROTOCOL" --roots "$ROOTS/roots.parquet" \
  --episodes-root "$EPISODES" --output-dir "$OUT/metrics/exit_probe" \
  > "$LOGS/aggregate_exit_probe.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p.evaluate_exit_probe \
  --protocol "$PROTOCOL" --episode-table "$OUT/metrics/exit_probe/episodes.parquet" \
  --output-dir "$OUT/metrics/exit_probe" > "$LOGS/evaluate_exit_probe.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p.report_exit_probe \
  --protocol "$PROTOCOL" --experiment-root "$OUT" --output "$OUT/report/HB3P_EXIT_PROBE.md" \
  > "$LOGS/report_exit_probe.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p.package_exit_probe \
  --experiment-root "$OUT" --code-root "$CODE" --output "$OUT/package/HB3P_exit_probe_lightweight.zip" \
  > "$LOGS/package_exit_probe.log" 2>&1
mark exit-probe.done "collection, online probe, evaluation, report, and package complete"
rm -f "$STATUS/exit-probe.running"
