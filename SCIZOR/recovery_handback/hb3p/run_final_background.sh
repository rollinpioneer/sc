#!/usr/bin/env bash
set -Eeuo pipefail

OUT="${OUT:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_v1}"
WT="${WT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb3p-square-v1}"
CODE="${CODE:-$WT/SCIZOR}"
PY_SIM="${PY_SIM:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
PY_FEAT="${PY_FEAT:-/home/xushijie/.conda/envs/lerobot/bin/python}"
CONFIG="$OUT/config/hb3p.json"
PROTOCOL="$OUT/config/frozen_protocol.json"
ROOTS="$OUT/roots/hb3p_test"
EPISODES="$OUT/episodes/test"
STATUS="$OUT/status"
LOGS="$OUT/logs"
GPU_SIM0="${GPU_SIM0:-1}"
GPU_SIM1="${GPU_SIM1:-2}"
GPU_FEAT0="${GPU_FEAT0:-3}"
GPU_FEAT1="${GPU_FEAT1:-6}"

export PYTHONPATH="$CODE/robomimic:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY_SIM")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$STATUS" "$LOGS" "$EPISODES"
printf '%s\n' "$$" > "$STATUS/hb3p-final-supervisor.pid"

stamp() { date '+%F %T %Z'; }
mark() { printf '%s %s\n' "$(stamp)" "$2" > "$STATUS/$1"; }
worker_pids=()
sim_pids=()
stop_files=("$STATUS/feature_test0.stop" "$STATUS/feature_test1.stop")
cleanup() {
  for pid in "${sim_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${sim_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  touch "${stop_files[@]}"
  for pid in "${worker_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
fail() {
  code=$?
  rm -f "$STATUS/hb3p-model.running"
  printf '%s failed line=%s exit=%s\n' "$(stamp)" "$1" "$code" > "$STATUS/hb3p-model.failed"
  exit "$code"
}
trap cleanup EXIT
trap 'fail $LINENO' ERR

test -f "$PROTOCOL"
test -f "$OUT/config/frozen.sha256"
rm -f "$STATUS/hb3p-model.failed" "$STATUS/hb3p-model.done" "${stop_files[@]}"
mark hb3p-model.running "pid=$$ sim_gpus=$GPU_SIM0,$GPU_SIM1 feature_gpus=$GPU_FEAT0,$GPU_FEAT1"

"$PY_SIM" -m compileall -q "$CODE/recovery_handback/hb3p"
git -C "$WT" diff --check -- SCIZOR/recovery_handback/hb3p

mark hb3p-E-collect.running "collect fixed seeds 500000..500079 on gpu=$GPU_SIM0"
CUDA_VISIBLE_DEVICES="$GPU_SIM0" MUJOCO_EGL_DEVICE_ID="$GPU_SIM0" \
  "$PY_SIM" -m recovery_handback.hb3p.collect_test \
  --config "$CONFIG" --protocol "$PROTOCOL" --start-index 0 --count 80 \
  --output-dir "$ROOTS" --resume > "$LOGS/collect_test.log" 2>&1
mark hb3p-E-collect.done "80 roots and label-free runtime manifest complete"

for index in 0 1; do
  queue="$OUT/ipc/test$index"
  ready="$queue/worker_ready.json"
  rm -f "$ready"
  mkdir -p "$queue"
  gpu="$GPU_FEAT0"; [[ "$index" == 1 ]] && gpu="$GPU_FEAT1"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY_FEAT" -m recovery_handback.hb3p.serve \
    --config "$CONFIG" --protocol "$PROTOCOL" --queue-dir "$queue" --device cuda \
    --stop-file "${stop_files[$index]}" > "$LOGS/feature_test$index.log" 2>&1 &
  worker_pids+=("$!")
  printf '%s\n' "$!" > "$STATUS/feature_test$index.pid"
done
for index in 0 1; do
  ready="$OUT/ipc/test$index/worker_ready.json"
  for _ in $(seq 1 120); do
    [[ -f "$ready" ]] && break
    kill -0 "${worker_pids[$index]}"
    sleep 5
  done
  test -f "$ready"
done

mark hb3p-E-online.running "three unique online methods across two root shards"
CUDA_VISIBLE_DEVICES="$GPU_SIM0" MUJOCO_EGL_DEVICE_ID="$GPU_SIM0" \
  "$PY_SIM" -m recovery_handback.hb3p.run_online \
  --config "$CONFIG" --protocol "$PROTOCOL" --roots "$ROOTS/runtime_root_manifest.jsonl" \
  --queue-dir "$OUT/ipc/test0" --num-shards 2 --shard-index 0 \
  --output-dir "$EPISODES/shard0" --resume > "$LOGS/online_test0.log" 2>&1 &
sim_pids+=("$!")
CUDA_VISIBLE_DEVICES="$GPU_SIM1" MUJOCO_EGL_DEVICE_ID="$GPU_SIM1" \
  "$PY_SIM" -m recovery_handback.hb3p.run_online \
  --config "$CONFIG" --protocol "$PROTOCOL" --roots "$ROOTS/runtime_root_manifest.jsonl" \
  --queue-dir "$OUT/ipc/test1" --num-shards 2 --shard-index 1 \
  --output-dir "$EPISODES/shard1" --resume > "$LOGS/online_test1.log" 2>&1 &
sim_pids+=("$!")
printf '%s\n' "${sim_pids[@]}" > "$STATUS/hb3p-online-workers.pid"
wait "${sim_pids[0]}"
wait "${sim_pids[1]}"
sim_pids=()
mark hb3p-E-online.done "all unique online executions complete"

touch "${stop_files[@]}"
wait "${worker_pids[0]}"
wait "${worker_pids[1]}"
worker_pids=()

"$PY_SIM" -m recovery_handback.hb3p.aggregate \
  --config "$CONFIG" --protocol "$PROTOCOL" --roots "$ROOTS/roots.parquet" \
  --episodes-root "$EPISODES" --output-dir "$OUT/metrics/test" > "$LOGS/aggregate_test.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p.evaluate \
  --config "$CONFIG" --protocol "$PROTOCOL" \
  --episode-table "$OUT/metrics/test/episodes.parquet" \
  --output-dir "$OUT/metrics/test" > "$LOGS/evaluate_test.log" 2>&1
mark hb3p-F.done "episode metrics and paired bootstrap complete"
mark hb3p-G.done "layered research decision generated"

"$PY_SIM" -m recovery_handback.hb3p.report \
  --config "$CONFIG" --experiment-root "$OUT" --output "$OUT/report/HB3P_REPORT.md" \
  > "$LOGS/report.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p.package \
  --experiment-root "$OUT" --output "$OUT/package/HB3P_results_lightweight.zip" \
  > "$LOGS/package.log" 2>&1
"$PY_SIM" -m recovery_handback.hb3p.export_lightweight \
  --experiment-root "$OUT" --destination "$WT/experiments/handback/hb3p_v1" \
  > "$LOGS/export_lightweight.log" 2>&1
mark hb3p-H.done "report, lightweight package, and Git export complete"
mark hb3p-model.done "test/evaluation/report/package/export complete"
rm -f "$STATUS/hb3p-model.running"
