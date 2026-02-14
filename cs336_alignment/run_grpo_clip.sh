#!/bin/bash

# ================= 配置区 =================
# 指向 Base 模型
BASE_MODEL="model/Qwen2.5-Math-1.5B" 

TRAIN_DATA="data/gsm8k/train.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/ablation_clipping"

# [关键设定]：请填入你在上一步 Sweep 实验中发现效果最好的参数
BEST_E=3
BEST_TB=256
# 对应之前计算出的 LR (3e-5 * 1/sqrt(3) * 256/256 ≈ 1.73e-5)
BEST_LR="0.0000173" 

# 对齐之前的 WandB 项目
WANDB_PROJECT="cs336-grpo-after-base-offpolicy"
RUN_NAME="E${BEST_E}_TB${BEST_TB}_LR${BEST_LR}_NO_CLIP"

echo "======================================================="
echo "🚨 启动截断消融实验 (No-Clip Mode)"
echo "📈 使用最佳参数: Epochs=$BEST_E | TB=$BEST_TB | LR=$BEST_LR"
echo "🎯 WandB Project: $WANDB_PROJECT"
echo "======================================================="

# 执行训练指令
# 注意：确保你的 Python 代码中已经处理了 "grpo_no_clip" 这个 loss_type
uv run python cs336_alignment/train_grpo.py \
    --model_id "$BASE_MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --test_data_path "$VAL_DATA" \
    --prompt_path "$PROMPT_TEMPLATE" \
    --output_dir "${OUTPUT_BASE}/${RUN_NAME}" \
    --n_grpo_steps 200 \
    --lr "$BEST_LR" \
    --rollout_batch_size 256 \
    --group_size 8 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 128 \
    --epochs_per_rollout_batch "$BEST_E" \
    --loss_type "grpo_no_clip" \
    --length_norm_type "mask_normalize" \
    --device cuda:0 \
    --vllm_device cuda:1 \
    --vllm_gpu_util 0.3 \
    --eval_every_steps 8 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$RUN_NAME" \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ 实验崩溃！(这通常验证了 No-Clip 在异策略下的不稳定性)"
    exit 1
fi

echo "✅ 实验 $RUN_NAME 完成！"