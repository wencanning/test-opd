#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_USE_V1=${VLLM_USE_V1:-1}

search_train=${SEARCH_TRAIN:-<Your_SearchR1_Train_Parquet>}
search_eval=${SEARCH_EVAL:-<Your_SearchR1_Eval_Parquet>}

student_model_path=${STUDENT_MODEL_PATH:-<Your_Student_Model_Path>}
teacher_model_path=${TEACHER_MODEL_PATH:-<Your_SearchR1_Teacher_Model_Path>}

tool_config_path=${TOOL_CONFIG_PATH:-examples/sglang_multiturn/config/tool_config/search_tool_config.yaml}

project_name=${PROJECT_NAME:-sod}
experiment_name=${EXPERIMENT_NAME:-search_r1_sod}
default_local_dir=${DEFAULT_LOCAL_DIR:-./checkpoint/$experiment_name}

train_files="['$search_train']"
test_files="['$search_eval']"

max_turns=8
max_prompt_length=4096
max_response_length=8192
actor_lr=1e-6

train_batch_size=64
ppo_mini_batch_size=16
n_resp_per_prompt=8
n_resp_per_prompt_val=8

infer_tp=4
train_sp=4

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    +algorithm.token_kl_reg.enable=True \
    +algorithm.token_kl_reg.stepwise_enable=True \
    +algorithm.token_kl_reg.stepwise_epsilon=1e-6 \
    +algorithm.token_kl_reg.stepwise_delta=0.2 \
    +algorithm.token_kl_reg.stepwise_opd_coef=1.0 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    custom_reward_function.path=recipe/search_r1/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 1)) \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 4)) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward_model.reward_manager=naive \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.default_local_dir=$default_local_dir \
    trainer.total_epochs=1 "$@"
