#!/bin/bash

# 定义消融实验的参数空间
# DATASET_SIZES=(128 256 512 1024 4096 7395)
DATASET_SIZES=(7395)
EPOCHS_LIST=(1)

# 基础配置
MODEL_ID="model/Qwen2.5-Math-1.5B"
TRAIN_DATA="data/gsm8k/train_sft_reason_gsm8k_raw.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/checkpoints"
WANDB_PROJECT="cs336-sft-gsm8k_raw"

# 遍历实验组合
for EPOCH in "${EPOCHS_LIST[@]}"; do
    for SIZE in "${DATASET_SIZES[@]}"; do
        
        # 自动生成具有辨识度的 WandB 运行名称
        # 格式示例: size1024_epoch3_sft
        RUN_NAME="size${SIZE}_epoch${EPOCH}_sft_correct_maxTokens3000"
        
        echo "========================================================="
        echo "🚀 启动实验: 数据量 $SIZE | Epoch $EPOCH"
        echo "📈 WandB Run Name: $RUN_NAME"
        echo "========================================================="

        # 执行训练指令
        uv run python cs336_alignment/train_sft.py \
            --model_id "$MODEL_ID" \
            --train_data_path "$TRAIN_DATA" \
            --val_data_path "$VAL_DATA" \
            --prompt_path "$PROMPT_TEMPLATE" \
            --output_dir "$OUTPUT_BASE" \
            --dataset_size "$SIZE" \
            --epochs "$EPOCH" \
            --lr 2e-5 \
            --batch_size 8 \
            --micro_batch_size 1 \
            --seed 42 \
            --max_tokens 1024 \
            --device cuda:0 \
            --vllm_device cuda:1 \
            --vllm_gpu_util 0.2 \
            --eval_every_steps 10 \
            --max_eval_samples 2000 \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "$RUN_NAME"

        # 如果某一组实验崩溃，脚本会停止，方便排查问题
        if [ $? -ne 0 ]; then
            echo "❌ 实验 $RUN_NAME 失败，停止运行后续实验。"
            exit 1
        fi

        echo "✅ 实验 $RUN_NAME 完成！"
        echo ""
    done
done

echo "🎉 所有 18 组消融实验执行完毕！"