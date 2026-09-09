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
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY_FEAT")")/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$STATUS" "$LOGS"
printf '%s\n' "$$" > "$LOGS/hb2-model-supervisor.pid"
stamp() { date '+%F %T %Z'; }
mark() { printf '%s %s\n' "$(stamp)" "$2" > "$STATUS/$1"; }
fail() { code=$?; printf '%s failed line=%s exit=%s\n' "$(stamp)" "$1" "$code" > "$STATUS/hb2-model.failed"; exit "$code"; }
trap 'fail $LINENO' ERR
run_gpu() { local gpu="$1"; shift; CUDA_VISIBLE_DEVICES="$gpu" MUJOCO_EGL_DEVICE_ID="$gpu" "$@"; }

rm -f "$STATUS/hb2-model.failed"
mark hb2-model.running "pid=$$ waiting for frozen data"

while [[ ! -f "$STATUS/hb2-data.done" ]]; do
  if ! kill -0 "$(cat "$LOGS/hb2-data-supervisor.pid" 2>/dev/null || echo 0)" 2>/dev/null; then
    [[ -f "$STATUS/hb2-data.failed" ]] && sleep 20 || true
  fi
  sleep 30
done
data_status="$($PY_FEAT -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$HB2_ROOT/data/frozen/data_sufficiency.json")"
if [[ "$data_status" == "HOLD_HB2_DATA" ]]; then
  mark hb2-model.done "stopped because data gate status=HOLD_HB2_DATA"
  exit 0
fi

mark anchor-features.running "DINOv2 frozen cache on gpu=6"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.features \
  --config "$CONFIG" --dataset "$HB2_ROOT/data/frozen" --roles train validation \
  --output-dir "$HB2_ROOT/features" --batch-size 64 --resume \
  > "$LOGS/features_anchor.log" 2>&1
mark anchor-features.done "anchor cache and normalizer complete"

mark handoff-features.running "DINOv2 frozen handoff cache on gpu=6"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.features \
  --config "$CONFIG" --dataset "$HB2_ROOT/data/frozen" --roles train validation \
  --head handoff --output-dir "$HB2_ROOT/features/handoff" --batch-size 64 --resume \
  > "$LOGS/features_handoff.log" 2>&1
mark handoff-features.done "handoff cache and normalizer complete"

mark model-smoke.running "M2 two-batch gradient smoke on gpu=6"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.train \
  --config "$CONFIG" --model M2_global --seed 0 --dataset "$HB2_ROOT/data/frozen" \
  --features "$HB2_ROOT/features" --smoke-batches 2 --output-dir "$HB2_ROOT/models/smoke" \
  > "$LOGS/train_smoke.log" 2>&1
mark model-smoke.done "gradient/output/label smoke complete"

mark models-seed0.running "fixed six-model matrix"
run_gpu 1 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model M0_time --seed 0 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/M0_time/seed0" > "$LOGS/train_M0_time.log" 2>&1 & p0=$!
run_gpu 2 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model M1_proprio --seed 0 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/M1_proprio/seed0" > "$LOGS/train_M1_proprio.log" 2>&1 & p1=$!
run_gpu 3 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model M2_global --seed 0 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/M2_global/seed0" > "$LOGS/train_M2_global.log" 2>&1 & p2=$!
run_gpu 4 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model M3_local --seed 0 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/M3_local/seed0" > "$LOGS/train_M3_local.log" 2>&1 & p3=$!
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model M4_paired --seed 0 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/M4_paired/seed0" > "$LOGS/train_M4_paired.log" 2>&1 & p4=$!
run_gpu 1 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model M5_single --seed 0 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/M5_single/seed0" > "$LOGS/train_M5_single.log" 2>&1 & p5=$!
printf '%s\n' "$p0" "$p1" "$p2" "$p3" "$p4" "$p5" > "$LOGS/hb2-current-workers.pid"
wait "$p0"; wait "$p1"; wait "$p2"; wait "$p3"; wait "$p4"; wait "$p5"
mark models-seed0.done "six seed0 models complete"

mark validation.running "root-equal probability metrics and calibration"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.validate \
  --config "$CONFIG" --role validation --models-root "$HB2_ROOT/models" \
  --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" \
  --output-dir "$HB2_ROOT/metrics/validation" > "$LOGS/validate_anchor.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.select_protocol \
  --config "$CONFIG" --validation-dir "$HB2_ROOT/metrics/validation" \
  --output "$HB2_ROOT/config/selected_protocol.pre_freeze.json" > "$LOGS/select_protocol.log" 2>&1
mark validation.done "validation selection written"

selected="$($PY_FEAT -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_visual_model"])' "$HB2_ROOT/config/selected_protocol.pre_freeze.json")"
mark selected-seeds.running "additional seeds 1 and 2 for $selected"
run_gpu 1 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model "$selected" --seed 1 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/$selected/seed1" > "$LOGS/train_${selected}_seed1.log" 2>&1 & ps1=$!
run_gpu 2 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --model "$selected" --seed 2 --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features" --output-dir "$HB2_ROOT/models/$selected/seed2" > "$LOGS/train_${selected}_seed2.log" 2>&1 & ps2=$!
printf '%s\n' "$ps1" "$ps2" > "$LOGS/hb2-current-workers.pid"; wait "$ps1"; wait "$ps2"

mark protocol-freeze.running "freeze F protocol before test"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.freeze_protocol \
  --config "$CONFIG" --selection "$HB2_ROOT/config/selected_protocol.pre_freeze.json" \
  --models-root "$HB2_ROOT/models" --output "$HB2_ROOT/config/frozen_protocol.json" \
  > "$LOGS/freeze_protocol.log" 2>&1
mark protocol-freeze.done "F protocol locked"

mark handoff-models.running "proprio vs selected visual H seed0"
run_gpu 1 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --head handoff --model proprio --seed 0 --protocol "$HB2_ROOT/config/frozen_protocol.json" --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features/handoff" --output-dir "$HB2_ROOT/models/handoff/proprio/seed0" > "$LOGS/train_handoff_proprio.log" 2>&1 & ph0=$!
run_gpu 2 "$PY_FEAT" -m recovery_handback.hb2.train --config "$CONFIG" --head handoff --model selected_visual --seed 0 --protocol "$HB2_ROOT/config/frozen_protocol.json" --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features/handoff" --output-dir "$HB2_ROOT/models/handoff/selected_visual/seed0" > "$LOGS/train_handoff_visual.log" 2>&1 & ph1=$!
printf '%s\n' "$ph0" "$ph1" > "$LOGS/hb2-current-workers.pid"; wait "$ph0"; wait "$ph1"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.validate --config "$CONFIG" --head handoff --role validation --models-root "$HB2_ROOT/models/handoff" --dataset "$HB2_ROOT/data/frozen" --features "$HB2_ROOT/features/handoff" --output-dir "$HB2_ROOT/metrics/handoff_validation" > "$LOGS/validate_handoff.log" 2>&1
"$PY_FEAT" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); s=json.load(open(sys.argv[2])); json.dump({"schema_version":"hb2_frozen_handoff_protocol_v1","models":s.get("models",["proprio","selected_visual"]),"canonical_seed":0,"input_support":"fixed_length_handoff_only","validation_locked":True},open(p,"w"),indent=2); Path(p).write_text(Path(p).read_text()+"\n")' "$HB2_ROOT/config/frozen_handoff_protocol.json" "$HB2_ROOT/metrics/handoff_validation/summary.json"
mark handoff-models.done "H protocol locked"

run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.runtime_probe --config "$CONFIG" --protocol "$HB2_ROOT/config/frozen_protocol.json" --pilot-roots "$HB2_ROOT/roots/hb2_pilot" --output-dir "$HB2_ROOT/metrics/runtime_probe" > "$LOGS/runtime_probe.log" 2>&1

mark test-collect.running "new test roots after protocol freeze"
run_gpu 6 "$PY_SIM" -m recovery_handback.hb2.collect --config "$CONFIG" --role hb2_test --start-index 0 --count 40 --output-dir "$HB2_ROOT/roots/hb2_test" > "$LOGS/collect_hb2_test.log" 2>&1
mark test-roots.done "test roots and anchors complete"
run_gpu 1 "$PY_SIM" -m recovery_handback.execution.run_branches --config "$CONFIG" --policy-pair "$HB2_ROOT/assets/policy_pair_square.json" --anchors "$HB2_ROOT/roots/hb2_test/anchors.parquet" --role hb2_test --output-dir "$HB2_ROOT/branches/hb2_test/shard_000" --num-shards 2 --shard-index 0 --resume --base-device cuda --repair-device cuda > "$LOGS/branches_hb2_test_000.log" 2>&1 & pt0=$!
run_gpu 2 "$PY_SIM" -m recovery_handback.execution.run_branches --config "$CONFIG" --policy-pair "$HB2_ROOT/assets/policy_pair_square.json" --anchors "$HB2_ROOT/roots/hb2_test/anchors.parquet" --role hb2_test --output-dir "$HB2_ROOT/branches/hb2_test/shard_001" --num-shards 2 --shard-index 1 --resume --base-device cuda --repair-device cuda > "$LOGS/branches_hb2_test_001.log" 2>&1 & pt1=$!
printf '%s\n' "$pt0" "$pt1" > "$LOGS/hb2-current-workers.pid"; wait "$pt0"; wait "$pt1"
"$PY_SIM" -m recovery_handback.hb2.aggregate --config "$CONFIG" --role hb2_test --anchors "$HB2_ROOT/roots/hb2_test/anchors.parquet" --branches-root "$HB2_ROOT/branches/hb2_test" --output-dir "$HB2_ROOT/data/hb2_test" > "$LOGS/aggregate_hb2_test.log" 2>&1
"$PY_FEAT" -m recovery_handback.hb2.attach_test --config "$CONFIG" --frozen-dataset "$HB2_ROOT/data/frozen" --test-data "$HB2_ROOT/data/hb2_test" --output-dir "$HB2_ROOT/data/test_attached" > "$LOGS/attach_test.log" 2>&1
mark test-collect.done "test branches and labels complete"

mark test-features.running "test DINOv2 cache"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.features --config "$CONFIG" --dataset "$HB2_ROOT/data/test_attached" --roles test --normalizer "$HB2_ROOT/features/normalizer.json" --output-dir "$HB2_ROOT/features/test" --batch-size 64 > "$LOGS/features_test.log" 2>&1
mark test-features.done "test cache complete"
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.predict --protocol "$HB2_ROOT/config/frozen_protocol.json" --handoff-protocol "$HB2_ROOT/config/frozen_handoff_protocol.json" --dataset "$HB2_ROOT/data/test_attached" --features "$HB2_ROOT/features/test" --role test --output-dir "$HB2_ROOT/predictions/test" > "$LOGS/predict_test.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.evaluate --config "$CONFIG" --protocol "$HB2_ROOT/config/frozen_protocol.json" --handoff-protocol "$HB2_ROOT/config/frozen_handoff_protocol.json" --dataset "$HB2_ROOT/data/test_attached" --predictions "$HB2_ROOT/predictions/test" --output-dir "$HB2_ROOT/metrics/test" > "$LOGS/evaluate_test.log" 2>&1
run_gpu 6 "$PY_FEAT" -m recovery_handback.hb2.report --config "$CONFIG" --experiment-root "$HB2_ROOT" --output "$HB2_ROOT/report/HB2_REPORT.md" > "$LOGS/report.log" 2>&1
"$PY_FEAT" -m recovery_handback.hb2.package --experiment-root "$HB2_ROOT" --output "$HB2_ROOT/package/HB2_results_lightweight.zip" > "$LOGS/package.log" 2>&1
sha256sum "$HB2_ROOT/package/HB2_results_lightweight.zip" > "$HB2_ROOT/package/HB2_results_lightweight.zip.sha256"
mark hb2-model.done "features/models/protocol/test/evaluation/report/package complete"
exit 0
