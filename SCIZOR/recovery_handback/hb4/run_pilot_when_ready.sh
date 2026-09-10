#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1}"
RUN_ROOT="${RUN_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1}"
PYTHON="${PYTHON:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
SUMMARY="$RUN_ROOT/data/standard_teacher/summary.json"
LOG="$RUN_ROOT/logs/pilot.log"

mkdir -p "$RUN_ROOT/logs"
exec >>"$LOG" 2>&1
printf 'started=%s\n' "$(date --iso-8601=seconds)"
while :; do
  if [[ -f "$SUMMARY" ]] && "$PYTHON" - "$SUMMARY" <<'PY'
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("processed_records") == 200 else 1)
PY
  then
    break
  fi
  sleep 30
done

env CUDA_VISIBLE_DEVICES=0,1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" \
  "$PYTHON" -m recovery_handback.hb4.pilot --device cuda:1 \
  --output-root "$RUN_ROOT/data/pilot"
printf 'finished=%s\n' "$(date --iso-8601=seconds)"
