#!/bin/bash
set -x

# ============================================================
# SFT Training Script for SOD
# Supervised fine-tuning on multi-turn agent trajectories.
# ============================================================

# [Optional] Uncomment and set your WandB API key for logging
# export WANDB_API_KEY="<Your_WandB_API_Key>"
# python -c "import wandb; wandb.login()"
export WANDB_MODE=${WANDB_MODE:-offline}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

nnodes=${NNODES:-1}
nproc_per_node=${NPROC_PER_NODE:-8}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}
node_rank=${NODE_RANK:-0}

HDFS_ROOT=${HDFS_ROOT:-$PWD}
DATA_ROOT=${DATA_ROOT:-$PWD}

project_name=${PROJECT_NAME:-sod}
experiment_name=${EXPERIMENT_NAME:-qwen3_4b_sft}

# Path to the SFT training data (parquet format)
TRAIN_DATA=${TRAIN_DATA:-<Your_SFT_Data_Path>}
EVAL_DATA=${EVAL_DATA:-$TRAIN_DATA}

# Path to the base model (HuggingFace format)
MODEL_PATH=${MODEL_PATH:-<Your_Base_Model_Path>}

# Path to save checkpoints
SAVE_PATH=${SAVE_PATH:-./checkpoint/$experiment_name}

torchrun --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --master-addr=$master_addr \
    --master-port=$master_port \
    --node-rank=$node_rank \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=32768 \
    data.truncation=right \
    data.train_batch_size=128 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=tools \
    data.micro_batch_size_per_gpu=16 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=5 \
    trainer.save_freq=50 \
    ulysses_sequence_parallel_size=4 \
    use_remove_padding=true "$@"
