#!/bin/bash
set -x

ulimit -n 65535

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export HYDRA_FULL_ERROR=1

search_train=/data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD/data/nq_hotpotqa_train/train.parquet
search_eval=/data/home/wencanning/workplace/Agentic-RAG/SearchR1-OPD/data/nq_hotpotqa_train/test.parquet

student_model_path=/data/home/wencanning/models/Qwen2.5-0.5B-Ins
teacher_model_path=/data/home/wencanning/models/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3

tool_config_path=examples/sglang_multiturn/config/tool_config/search_tool_config.yaml

project_name=Search-R1
experiment_name=-search_r1_test_opd
default_local_dir=./checkpoint/$experiment_name

train_files="['$search_train']"
test_files="['$search_eval']"

max_turns=4
max_prompt_length=1024
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
max_obs_length=${MAX_OBS_LENGTH:-500}
effective_response_length=$((max_response_length + max_obs_length * max_turns))
actor_lr=1e-6
lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO:-0.285}

train_batch_size=4
val_batch_size=256
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=1
n_resp_per_prompt=8
n_resp_per_prompt_val=1

token_kl_gamma=1.0
token_kl_beta_min=0.0
token_kl_beta_max=0.05

rollout_log_prob_micro_batch_size_per_gpu=2
ref_log_prob_micro_batch_size_per_gpu=2
rollout_gpu_memory_utilization=0.5

infer_tp=1
total_epochs=1
total_training_steps=2
n_gpus_per_node=2

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    +algorithm.token_kl_reg.enable=True \
    +algorithm.token_kl_reg.stepwise_enable=False \
    +algorithm.token_kl_reg.gamma=$token_kl_gamma \
    +algorithm.token_kl_reg.beta_min=$token_kl_beta_min \
    +algorithm.token_kl_reg.beta_max=$token_kl_beta_max \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=True \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$effective_response_length \
    +data.max_model_response_length=$max_response_length \
    +data.max_obs_length=$max_obs_length \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    custom_reward_function.path=recipe/search_r1/reward.py \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$lr_warmup_steps_ratio \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$rollout_log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ref_log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.gpu_memory_utilization=$rollout_gpu_memory_utilization \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$tool_config_path \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.default_local_dir=$default_local_dir \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps "$@" \
    trainer.rollout_data_dir=./checkpoint/$experiment_name/rollout_generations