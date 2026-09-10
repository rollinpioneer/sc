#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1}"
RUN_ROOT="${RUN_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1}"
EXPORT_ROOT="${EXPORT_ROOT:-$CODE_ROOT/experiments/handback/hb4_square_absorb_v1}"
PYTHON="${PYTHON:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
LOG="$RUN_ROOT/logs/build_training_blocks.log"

mkdir -p "$RUN_ROOT/logs"
exec >>"$LOG" 2>&1
printf 'started=%s\n' "$(date --iso-8601=seconds)"

while :; do
  audit="$EXPORT_ROOT/data/reconstruction_audit.json"
  summary="$RUN_ROOT/data/standard_teacher/summary.json"
  if [[ -f "$audit" && -f "$summary" ]]; then
    if "$PYTHON" - "$audit" "$summary" <<'PY'
import json, sys
audit = json.load(open(sys.argv[1]))
summary = json.load(open(sys.argv[2]))
raise SystemExit(0 if audit.get("status") in {"PASS_B_READY", "PASS_HB4_B_READY"} and summary.get("processed_records") == 200 else 1)
PY
    then
      break
    fi
  fi
  sleep 30
done

env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" \
"$PYTHON" -m recovery_handback.hb4.build_training_blocks \
  --export-root "$EXPORT_ROOT" --run-root "$RUN_ROOT"
env MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" \
  "$PYTHON" -m recovery_handback.hb4.build_training_hdf5 \
  --export-root "$EXPORT_ROOT" --run-root "$RUN_ROOT"
printf 'finished=%s\n' "$(date --iso-8601=seconds)"
