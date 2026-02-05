#!/bin/bash

# --- 实验空间定义 (根据作业要求的消融范围) ---
# Rollouts：每次 EI 迭代中模拟的路径数量，每个问题回答结果的数量
ROLLOUTS_LIST=(4 8)
# SFT Epochs
EPOCHS_LIST=(1)
# Db (EI Batch Size): 这是每一轮迭代采样的题目总数
DB_SIZES=(512 1024 2048)

# --- 基础配置 ---
# 注意：EI 最好从一个已经 SFT 过的模型开始，而不是 Base 模型
# 这里假设你已经跑完了 SFT 并存了一个模型
BASE_MODEL="model/Qwen2.5-Math-1.5B"

TRAIN_DATA="data/gsm8k/train_sft_reason_gsm8k_raw.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/ei_checkpoints"
WANDB_PROJECT="cs336-ei-gsm8k_raw"

# 固定参数
N_EI_STEPS=10
LR=2e-5 # EI 需要更细腻的学习率

# --- 循环运行实验 ---
for G in "${ROLLOUTS_LIST[@]}"; do
    for E in "${EPOCHS_LIST[@]}"; do
        for DB in "${DB_SIZES[@]}"; do
            
            # 为了节省时间，我们只跑几个有代表性的组合
            # 组合逻辑：控制变量法
            # 1. Base: G=4, E=1, Db=512
            # 2. Scale G: G=8, E=1, Db=512
            # 3. Scale E: G=4, E=2, Db=512
            # 4. Scale Db: G=4, E=1, Db=1024
            
            # 简单的筛选逻辑，跳过所有非目标组合 (可选)
            # if [[ "$G" == "8" && "$E" == "2" ]]; then continue; fi

            RUN_NAME="ei_G${G}_E${E}_Db${DB}"
            
            echo "========================================================="
            echo "🚀 启动 EI 实验: Rollouts=$G | Epochs=$E | Db=$DB"
            echo "📈 WandB Run Name: $RUN_NAME"
            echo "========================================================="

            uv run python cs336_alignment/train_ei.py \
                --model_id "$BASE_MODEL" \
                --train_data_path "$TRAIN_DATA" \
                --val_data_path "$VAL_DATA" \
                --prompt_path "$PROMPT_TEMPLATE" \
                --output_dir "$OUTPUT_BASE" \
                --n_ei_steps "$N_EI_STEPS" \
                --ei_batch_size "$DB" \
                --rollouts "$G" \
                --sft_epochs "$E" \
                --lr "$LR" \
                --batch_size 16 \
                --micro_batch_size 1 \
                --seed 42 \
                --max_tokens 1024 \
                --device cuda:0 \
                --vllm_device cuda:1 \
                --vllm_gpu_util 0.4 \
                --max_eval_samples 100 \
                --wandb_project "$WANDB_PROJECT" \
                --wandb_run_name "$RUN_NAME"

            if [ $? -ne 0 ]; then
                echo "❌ 实验 $RUN_NAME 失败"
                exit 1
            fi
            
            echo "✅ 实验 $RUN_NAME 完成"
            echo ""
            sleep 5 # 释放显存
        done
    done
done

echo "🎉 所有 EI 消融实验执行完毕！"