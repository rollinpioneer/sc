#!/usr/bin/env bash
set -euo pipefail
: "${SCIZOR_ROOT:?}"; : "${TRAIN_DATA_DIR:?}"; : "${SAVE_ROOT:?}"
SEED="${SEED:-0}"; STEPS="${STEPS:-10000}"; SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1000}"; BATCH_SIZE="${BATCH_SIZE:-128}"; PORT="${PORT:-1889}"
NUM_DATASETS="${NUM_DATASETS:-1}"; RUN_NAME="${RUN_NAME:-stage1_seed${SEED}}"
mkdir -p "$SAVE_ROOT"
cd "$SCIZOR_ROOT/curation/suboptimal_classifier"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" accelerate launch --num_processes 1 ./accelerate_train_one_loader.py \
  --name "$RUN_NAME" --debug --config.seed="$SEED" \
  --config.hdf5_dataset_kwargs.data_dir="$TRAIN_DATA_DIR" \
  --config.hdf5_dataset_kwargs.batch_size="$BATCH_SIZE" --config.hdf5_dataset_kwargs.num_workers=4 \
  --config.save_dir="$SAVE_ROOT" --config.discriminator.action_query_length=1 \
  --config.discriminator.num_blocks=6 --config.discriminator.head_token=cls --config.optimizer.lr=1e-4 \
  --config.num_steps="$STEPS" --config.window_size=1 --config.action_horizon=1 --config.future_action=0 \
  --config.discriminator.loss_fn_type=cross_entropy --port "$PORT" --config.save_interval="$SAVE_INTERVAL" \
  --config.eval_interval="$EVAL_INTERVAL" --config.discriminator.no_action_input=True --config.future_image=True \
  --config.discriminator.frozen_encoder=True --config.discriminator.encoder_type=dinov2 \
  --config.discriminator.d_model=768 --config.num_datasets="$NUM_DATASETS" \
  --config.discriminator.fusion_blocks_type=self-attn --config.discriminator.head_type=rank \
  --config.discriminator.no_text_input=True --config.grad_accum_steps=1 \
  --config.discriminator_dataset_kwargs.image_key=agentview_image
