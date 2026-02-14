#!/bin/bash
# export CUDA_VISIBLE_DEVICES='1,2'
# uv run bash cs336_alignment/run_grpo_lr.sh
# ================= 配置区 =================

BASE_MODEL="model/Qwen2.5-Math-1.5B" # base model
# BASE_MODEL="result/checkpoints/sft_subset7395_filteredTrue"  # sft model

# 数据与模版
# TRAIN_DATA="data/gsm8k/train.jsonl"
# TEST_DATA="data/gsm8k/test.jsonl"
TRAIN_DATA="data/math12k/data/train-00000-of-00001.parquet"
TEST_DATA="data/math12k/data/test-00000-of-00001.parquet"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/grpo_lr_sweep"

# WANDB_PROJECT="cs336-grpo-after-base-lr-grpo_clip"
WANDB_PROJECT="cs336-grpo-math12k-after-base-lr-grpo_clip"
# 待测试的学习率列
LR_LIST=(1e-6 5e-6 1e-5 3e-5 5e-5)
# LR_LIST=(6e-5) # 学习率过高
# ================= 循环运行 =================
for LR in "${LR_LIST[@]}"; do
    RUN_NAME="grpo_lr${LR}"
    echo "======================================================="
    echo "🚀 [LR Sweep] 启动实验: $RUN_NAME (LR=$LR)"
    echo "======================================================="
    
    uv run python cs336_alignment/train_grpo.py \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --test_data_path "$TEST_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "${OUTPUT_BASE}/${RUN_NAME}" \
        --n_grpo_steps 200 \
        --lr "$LR" \
        --rollout_batch_size 256 \
        --group_size 8 \
        --train_batch_size 256 \
        --gradient_accumulation_steps 32 \
        --epochs_per_rollout_batch 1 \
        --loss_type "grpo_clip" \
        --eval_every_steps 8 \
        --use_std_normalization \
        --length_norm_type "mask_norm" \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.3 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME" \
        --seed 42

    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 失败，跳过..."
    else
        echo "✅ 实验 $RUN_NAME 完成！"
    fi
    
    # 稍微休息释放显存
    sleep 5
done