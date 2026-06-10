#!/bin/bash
set -euo pipefail
set -x

# ============================================================
# Evaluation Script for SOD
# Evaluates a trained model on LiveCodeBench (v6, 2502-2505).
# ============================================================

# ---- Model ----
MODEL_PATH="${MODEL_PATH:-<Your_Model_Path>}"

# Optional readable tag in output directory name.
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH}")}"

# ---- Runtime Env ----
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ -z "${NUM_GPUS:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
        IFS=',' read -r -a __gpu_list <<< "${CUDA_VISIBLE_DEVICES}"
        NUM_GPUS="${#__gpu_list[@]}"
    else
        NUM_GPUS=1
    fi
fi

# CLI overrides for quick run:
#   bash run_eval_livecodebench.sh <MODEL_PATH> [hydra_overrides...]
if [[ $# -gt 0 && "${1}" != *=* ]]; then
    MODEL_PATH="$1"
    shift
fi

if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required"
    exit 1
fi

HDFS_ROOT="${HDFS_ROOT:-$PWD}"
DATA_ROOT="${DATA_ROOT:-$PWD}"

# ---- Evaluation Data Paths ----
LIVECODEBENCH="${LIVECODEBENCH:-<Your_LiveCodeBench_Eval_Data_Path>}"

EVAL_FILES="['${LIVECODEBENCH}']"

TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-recipe/demystify/sandbox_fusion_tool_config.yaml}"
PROJECT_NAME="${PROJECT_NAME:-sod_eval}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-eval_${MODEL_TAG}_livecodebench}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$DATA_ROOT/checkpoint/${EXPERIMENT_NAME}}"
EVAL_DUMP_DIR="${EVAL_DUMP_DIR:-$DEFAULT_LOCAL_DIR/eval_generations}"

mkdir -p "${DEFAULT_LOCAL_DIR}" "${EVAL_DUMP_DIR}"

# ---- Evaluation Configuration ----
MAX_TURNS="${MAX_TURNS:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-20480}"
N_RESP_PER_PROMPT_VAL="${N_RESP_PER_PROMPT_VAL:-32}"

GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.75}"
INFER_TP=4
TRAIN_SP=4

FSDP_OFFLOAD="${FSDP_OFFLOAD:-True}"

ACTOR_MAX_TOKEN_LEN_PER_GPU="$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 1 ))"

# LiveCodeBench uses longer overlong buffer (4096) to accommodate code generation
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-4096}"

if [[ "${ENABLE_WANDB:-0}" == "1" ]]; then
    LOGGER_CFG="['console','wandb']"
else
    LOGGER_CFG="['console']"
fi

# ---- Launch Evaluation ----
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    +algorithm.filter_groups.enable=False \
    +algorithm.token_kl_reg.enable=False \
    data.train_files="${EVAL_FILES}" \
    data.val_files="${EVAL_FILES}" \
    data.return_raw_chat=True \
    data.train_batch_size=1 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.custom_cls.path=recipe/demystify/reward.py \
    data.custom_cls.name=CustomRLHFDataset \
    custom_reward_function.path=recipe/demystify/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${TRAIN_SP}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${FSDP_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${FSDP_OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${INFER_TP}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTIL}" \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${N_RESP_PER_PROMPT_VAL}" \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=true \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}" \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=false \
    +reward_model.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
    trainer.logger="${LOGGER_CFG}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=0 \
    trainer.test_freq=0 \
    trainer.save_freq=0 \
    trainer.log_val_generations=20 \
    trainer.validation_data_dir="${EVAL_DUMP_DIR}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    "$@"
