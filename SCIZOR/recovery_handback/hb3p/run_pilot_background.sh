#!/usr/bin/env bash
set -Eeuo pipefail

OUT="${OUT:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb3p_v1}"
HB2="${HB2:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb2_v1}"
CODE="${CODE:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb3p-square-v1/SCIZOR}"
PY_SIM="${PY_SIM:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
PY_FEAT="${PY_FEAT:-/home/xushijie/.conda/envs/lerobot/bin/python}"
GPU_SIM="${GPU_SIM:-1}"
GPU_FEAT="${GPU_FEAT:-6}"
CONFIG="$OUT/config/hb3p.json"
PROTOCOL="$OUT/config/protocol.pre_freeze.json"
QUEUE="$OUT/ipc/pilot0"
STOP_FILE="$OUT/status/feature_pilot0.stop"
STATUS="$OUT/status"
LOGS="$OUT/logs"

export PYTHONPATH="$CODE/robomimic:$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY_SIM")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$STATUS" "$LOGS" "$QUEUE"
printf '%s\n' "$$" > "$STATUS/hb3p-pilot-supervisor.pid"

stamp() { date '+%F %T %Z'; }
mark() { printf '%s %s\n' "$(stamp)" "$2" > "$STATUS/$1"; }
worker_pid=""
cleanup() {
  touch "$STOP_FILE"
  if [[ -n "$worker_pid" ]] && kill -0 "$worker_pid" 2>/dev/null; then
    wait "$worker_pid" || true
  fi
}
fail() {
  code=$?
  printf '%s failed line=%s exit=%s\n' "$(stamp)" "$1" "$code" > "$STATUS/hb3p-pilot.failed"
  exit "$code"
}
trap cleanup EXIT
trap 'fail $LINENO' ERR

rm -f "$STOP_FILE" "$STATUS/hb3p-pilot.failed" "$STATUS/hb3p-pilot.done"
rm -f "$QUEUE/worker_ready.json"
mark hb3p-pilot.running "pid=$$ sim_gpu=$GPU_SIM feature_gpu=$GPU_FEAT"

CUDA_VISIBLE_DEVICES="$GPU_FEAT" "$PY_FEAT" -m recovery_handback.hb3p.serve \
  --config "$CONFIG" --protocol "$PROTOCOL" --queue-dir "$QUEUE" --device cuda \
  --stop-file "$STOP_FILE" > "$LOGS/feature_pilot0.log" 2>&1 &
worker_pid=$!
printf '%s\n' "$worker_pid" > "$STATUS/feature_pilot0.pid"

for _ in $(seq 1 120); do
  [[ -f "$QUEUE/worker_ready.json" ]] && break
  kill -0 "$worker_pid"
  sleep 5
done
test -f "$QUEUE/worker_ready.json"

CUDA_VISIBLE_DEVICES="$GPU_SIM" MUJOCO_EGL_DEVICE_ID="$GPU_SIM" \
  "$PY_SIM" -m recovery_handback.hb3p.pilot \
  --config "$CONFIG" --protocol "$PROTOCOL" \
  --pilot-roots "$HB2/roots/hb2_pilot" --queue-dir "$QUEUE" \
  --output-dir "$OUT/metrics/pilot" > "$LOGS/hb3p_pilot.log" 2>&1

mark hb3p-pilot.done "pilot parity complete"
