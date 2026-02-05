#!/bin/bash

# ================= 配置区 =================

BASE_MODEL="model/Qwen2.5-Math-1.5B" # base model
# BASE_MODEL="result/checkpoints/sft_subset7395_filteredTrue" # sft model
# WANDB_PROJECT="cs336-grpo-after-sft-std_lr3e-5"
WANDB_PROJECT="cs336-grpo-after-base-std"
BEST_LR=5e-5


TRAIN_DATA="data/gsm8k/train.jsonl"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/grpo_baseline_study"


# 消融实验目标
# 1. reinforce_with_baseline: 标准 GRPO (带均值/标准差归一化)
# 2. no_baseline: 朴素策略梯度 (直接用原始 Reward)
# 3. grpo_clip
LOSS_TYPES=("grpo_clip" "reinforce_with_baseline" "no_baseline")

# ================= 循环运行实验 =================
for TYPE in "${LOSS_TYPES[@]}"; do
    
    # 定义具有辨识度的 Run Name
    RUN_NAME="grpo_lr${BEST_LR}_type_${TYPE}"
    
    # 逻辑判断：如果是 no_baseline，则关闭标准差归一化
    if [ "$TYPE" == "no_baseline" ]; then
        STD_NORM_ARG=""
    else
        # reinforce_with_baseline 模式下开启组内标准差归一化
        STD_NORM_ARG="--use_std_normalization"
    fi

    # 每一组实验创建独立的输出文件夹，防止权重覆盖
    CURRENT_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"

    echo "======================================================="
    echo "🚀 [Ablation] 正在启动实验: $RUN_NAME"
    echo "📈 损失类型: $TYPE | 学习率: $BEST_LR"
    echo "📂 输出目录: $CURRENT_OUTPUT_DIR"
    echo "======================================================="

    # 执行训练指令
    uv run python cs336_alignment/train_grpo.py \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "$CURRENT_OUTPUT_DIR" \
        --n_grpo_steps 200 \
        --lr "$BEST_LR" \
        --rollout_batch_size 256 \
        --group_size 8 \
        --train_batch_size 256 \
        --gradient_accumulation_steps 128 \
        --epochs_per_rollout_batch 1 \
        --loss_type "$TYPE" \
        $STD_NORM_ARG \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.6 \
        --eval_every_steps 8 \
        --save_every_steps 20 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME" \
        --seed 42

    # 错误处理：如果实验失败，停止后续脚本运行
    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 异常终止，请检查日志！"
        exit 1
    fi
    
    echo "✅ 实验 $RUN_NAME 执行完毕！"
    echo "-------------------------------------------------------"
    
    # 每组实验间歇 5 秒，确保显存回收
    sleep 5
done

echo "🎉 所有基准线消融实验已完成！"