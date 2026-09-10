#!/usr/bin/env bash
set -Eeuo pipefail

CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb4-square-absorb-v1}"
RUN_ROOT="${RUN_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_hb4_runs/hb4_square_absorb_v1}"
EXPORT_ROOT="${EXPORT_ROOT:-$CODE_ROOT/experiments/handback/hb4_square_absorb_v1}"
PYTHON="${PYTHON:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
DECISION="$EXPORT_ROOT/metrics/development/decision.json"
FORMAL_SUMMARY="$EXPORT_ROOT/metrics/test/summary.json"

exec 9>"$RUN_ROOT/logs/finalize.lock"
flock -n 9 || { echo "finalize already running"; exit 0; }

while :; do
  if [[ -f "$FORMAL_SUMMARY" ]]; then
    "$PYTHON" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("coverage",{}).get("complete") else 1)' "$FORMAL_SUMMARY" && break
  fi
  if [[ -f "$DECISION" ]]; then
    decision="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1])).get("decision", ""))' "$DECISION")"
    [[ "$decision" == "HB4_DEVELOPMENT_GO" ]] || break
  fi
  sleep 60
done

env PYTHONPATH="$CODE_ROOT/SCIZOR:$CODE_ROOT/SCIZOR/robomimic" "$PYTHON" -m recovery_handback.hb4.postprocess --export-root "$EXPORT_ROOT" --run-root "$RUN_ROOT"
