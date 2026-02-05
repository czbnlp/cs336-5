#!/bin/bash
# export CUDA_VISIBLE_DEVICES='1,3'
# uv run bash cs336_alignment/run_grpo_prompt.sh
# ================= 配置区 =================
# 建议使用 Qwen 2.5 Math 1.5B Base 模型进行对比
BASE_MODEL="model/Qwen2.5-Math-1.5B" 
TRAIN_DATA="data/gsm8k/train.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"
OUTPUT_BASE="result/ablation_prompt"
WANDB_PROJECT="cs336-grpo-after-base-prompt"

BEST_LR=3e-5

# ================= 实验循环 =================
# 1. r1_zero: 包含 <think> 标签引导，使用 r1_zero_reward_fn
# 2. question_only: 仅包含题目，使用 question_only_reward_fn
for STYLE in "r1_zero" "question_only"; do

    PROMPT_FILE="cs336_alignment/prompts/${STYLE}.prompt"
    RUN_NAME="grpo_style_${STYLE}_lr${BEST_LR}"
    
    echo "======================================================="
    echo "🚀 启动 Prompt 消融实验: $STYLE"
    echo "📂 提示词路径: $PROMPT_FILE"
    echo "======================================================="

    uv run python cs336_alignment/train_grpo.py \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --test_data_path "$VAL_DATA" \
        --prompt_path "$PROMPT_FILE" \
        --prompt_style "$STYLE" \
        --output_dir "${OUTPUT_BASE}/${RUN_NAME}" \
        --n_grpo_steps 200 \
        --lr "$BEST_LR" \
        --rollout_batch_size 256 \
        --group_size 8 \
        --gradient_accumulation_steps 32 \
        --train_batch_size 256 \
        --length_norm_type "mask_normalize" \
        --loss_type "grpo_clip" \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.3 \
        --eval_every_steps 8 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME"

    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 失败！"
        exit 1
    fi
    sleep 10
done