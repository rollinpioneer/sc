#!/usr/bin/env bash
set -Eeuo pipefail

HB2_ROOT="${HB2_ROOT:-/home/__compress_data/xushijie/work/cr_scizor/experiments/handback/hb2_v1}"
CODE_ROOT="${CODE_ROOT:-/home/__compress_data/xushijie/work/cr_scizor_github_sc_worktrees/hb2-square-v1/SCIZOR}"
PY_FEAT="${PY_FEAT:-/home/xushijie/.conda/envs/lerobot/bin/python}"
PY_SIM="${PY_SIM:-/home/xushijie/.conda/envs/handback-hb1/bin/python}"
CONFIG="$HB2_ROOT/config/hb2.json"
STATUS="$HB2_ROOT/status"
LOGS="$HB2_ROOT/logs"
export PYTHONPATH="$CODE_ROOT/robomimic:$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY_SIM")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$STATUS" "$LOGS"
printf '%s\n' "$$" > "$LOGS/hb2-resume-supervisor.pid"
stamp() { date '+%F %T %Z'; }
mark() { printf '%s %s\n' "$(stamp)" "$2" > "$STATUS/$1"; }
fail() { code=$?; printf '%s failed line=%s exit=%s\n' "$(stamp)" "$1" "$code" > "$STATUS/hb2-model.failed"; exit "$code"; }
trap 'fail $LINENO' ERR
run_gpu() { local gpu="$1"; shift; CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" "$@"; }

test -f "$STATUS/hb2-data.done"
test -f "$HB2_ROOT/config/frozen_protocol.json"
test -f "$HB2_ROOT/config/frozen_handoff_protocol.json"
rm -f "$STATUS/hb2-model.failed" "$STATUS/runtime-probe.done" "$STATUS/test-collect.done" \
  "$STATUS/test-features.done" "$STATUS/hb2-model.done"
mark hb2-model-resume.running "pid=$$ from frozen F/H protocols"

mark runtime-probe.running "four-root offline/online parity and one-shot execution"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.runtime_probe \
  --config "$CONFIG" --protocol "$HB2_ROOT/config/frozen_protocol.json" \
  --handoff-protocol "$HB2_ROOT/config/frozen_handoff_protocol.json" \
  --pilot-roots "$HB2_ROOT/roots/hb2_pilot" --output-dir "$HB2_ROOT/metrics/runtime_probe" \
  > "$LOGS/runtime_probe.log" 2>&1
run_gpu 6 "$PY_SIM" -m recovery_handback.hb2.runtime_probe \
  --config "$CONFIG" --pilot-roots "$HB2_ROOT/roots/hb2_pilot" \
  --policy-pair "$HB2_ROOT/assets/policy_pair_square.json" \
  --execute-selections "$HB2_ROOT/metrics/runtime_probe/selected_branches.json" \
  --output-dir "$HB2_ROOT/metrics/runtime_probe" --base-device cuda --repair-device cuda \
  > "$LOGS/runtime_probe_execute.log" 2>&1
mark runtime-probe.done "four pilot roots passed frozen one-shot interface"

mark test-collect.running "new test roots after protocol freeze"
run_gpu 6 "$PY_SIM" -m recovery_handback.hb2.collect \
  --config "$CONFIG" --role hb2_test --start-index 0 --count 40 \
  --output-dir "$HB2_ROOT/roots/hb2_test" > "$LOGS/collect_hb2_test.log" 2>&1
mark test-roots.done "test roots and anchors complete"
run_gpu 1 "$PY_SIM" -m recovery_handback.execution.run_branches \
  --config "$CONFIG" --policy-pair "$HB2_ROOT/assets/policy_pair_square.json" \
  --anchors "$HB2_ROOT/roots/hb2_test/anchors.parquet" --role hb2_test \
  --output-dir "$HB2_ROOT/branches/hb2_test/shard_000" --num-shards 2 --shard-index 0 \
  --resume --base-device cuda --repair-device cuda > "$LOGS/branches_hb2_test_000.log" 2>&1 & p0=$!
run_gpu 2 "$PY_SIM" -m recovery_handback.execution.run_branches \
  --config "$CONFIG" --policy-pair "$HB2_ROOT/assets/policy_pair_square.json" \
  --anchors "$HB2_ROOT/roots/hb2_test/anchors.parquet" --role hb2_test \
  --output-dir "$HB2_ROOT/branches/hb2_test/shard_001" --num-shards 2 --shard-index 1 \
  --resume --base-device cuda --repair-device cuda > "$LOGS/branches_hb2_test_001.log" 2>&1 & p1=$!
printf '%s\n' "$p0" "$p1" > "$LOGS/hb2-current-workers.pid"
wait "$p0"; wait "$p1"
run_gpu 6 "$PY_SIM" -m recovery_handback.hb2.aggregate \
  --config "$CONFIG" --role hb2_test --anchors "$HB2_ROOT/roots/hb2_test/anchors.parquet" \
  --branches-root "$HB2_ROOT/branches/hb2_test" --output-dir "$HB2_ROOT/data/hb2_test" \
  > "$LOGS/aggregate_hb2_test.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.attach_test \
  --config "$CONFIG" --frozen-dataset "$HB2_ROOT/data/frozen" \
  --test-data "$HB2_ROOT/data/hb2_test" --output-dir "$HB2_ROOT/data/test_attached" \
  > "$LOGS/attach_test.log" 2>&1
mark test-collect.done "test branches and labels complete"

mark test-features.running "test DINOv2 caches"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.features \
  --config "$CONFIG" --dataset "$HB2_ROOT/data/test_attached" --roles test \
  --normalizer "$HB2_ROOT/features/normalizer.json" --output-dir "$HB2_ROOT/features/test" \
  --batch-size 64 > "$LOGS/features_test.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.features \
  --config "$CONFIG" --dataset "$HB2_ROOT/data/test_attached" --roles test --head handoff \
  --normalizer "$HB2_ROOT/features/handoff/normalizer.json" \
  --output-dir "$HB2_ROOT/features/test/handoff" --batch-size 64 > "$LOGS/features_test_handoff.log" 2>&1
mark test-features.done "test caches complete"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.predict \
  --protocol "$HB2_ROOT/config/frozen_protocol.json" \
  --handoff-protocol "$HB2_ROOT/config/frozen_handoff_protocol.json" \
  --dataset "$HB2_ROOT/data/test_attached" --features "$HB2_ROOT/features/test" --role test \
  --output-dir "$HB2_ROOT/predictions/test" > "$LOGS/predict_test.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.evaluate \
  --config "$CONFIG" --protocol "$HB2_ROOT/config/frozen_protocol.json" \
  --handoff-protocol "$HB2_ROOT/config/frozen_handoff_protocol.json" \
  --dataset "$HB2_ROOT/data/test_attached" --predictions "$HB2_ROOT/predictions/test" \
  --output-dir "$HB2_ROOT/metrics/test" > "$LOGS/evaluate_test.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.report \
  --config "$CONFIG" --experiment-root "$HB2_ROOT" \
  --output "$HB2_ROOT/report/HB2_REPORT.md" > "$LOGS/report.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.package \
  --experiment-root "$HB2_ROOT" --output "$HB2_ROOT/package/HB2_results_lightweight.zip" \
  > "$LOGS/package.log" 2>&1
mark hb2-model.done "runtime/test/evaluation/report/package complete"
