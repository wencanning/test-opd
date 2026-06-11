#!/bin/bash
set -x

export WANDB_MODE=${WANDB_MODE:-offline}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

nnodes=${NNODES:-1}
nproc_per_node=${NPROC_PER_NODE:-8}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}
node_rank=${NODE_RANK:-0}

project_name=${PROJECT_NAME:-search_r1}
experiment_name=${EXPERIMENT_NAME:-search_r1_sft}

TRAIN_DATA=${TRAIN_DATA:-$PWD/data/search_r1_sft/train.parquet}
EVAL_DATA=${EVAL_DATA:-$PWD/data/search_r1_sft/val.parquet}
MODEL_PATH=${MODEL_PATH:-<Your_Base_Model_Path>}
SAVE_PATH=${SAVE_PATH:-./checkpoint/$experiment_name}

MAX_LENGTH=${MAX_LENGTH:-8192}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-100}
ULYSSES_SEQUENCE_PARALLEL_SIZE=${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}

torchrun --nnodes=$nnodes \
    --nproc_per_node=$nproc_per_node \
    --master-addr=$master_addr \
    --master-port=$master_port \
    --node-rank=$node_rank \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=$MAX_LENGTH \
    data.truncation=right \
    +data.search_r1_masked.enable=true \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE \
    use_remove_padding=true "$@"
