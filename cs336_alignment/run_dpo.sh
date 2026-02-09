#!/bin/bash

# ================= 配置区域 =================

# 1. 显卡设置
# 假设你有至少两张卡。如果只有一张卡，需要调整下方的 DEVICE 参数，并大幅减小 Batch Size
export CUDA_VISIBLE_DEVICES=0,1

WANDB_PROJECT="cs336-dpo-hh"
wandb_run_name="dpo_after_qwen_3b_sft_checkpoint-400"

# 3. 路径设置
MODEL_PATH="model/Qwen2.5-3B" # base, 应使用SFT model
MODEL_PATH="result/sft_qwen_3b_ultraChat_SafetyLlama_new/checkpoint-400"
TRAIN_DATA="data/hh-rlhf"   
OUTPUT_DIR="result/dpo_qwen_3b_sft_checkpoint-400"

# 4. 评估数据路径
GSM8K_PATH="data/gsm8k/test.jsonl"
MMLU_PATH="data/MMLU-Pro/data/test-00000-of-00001.parquet"

POLICY_DEVICE="cuda:0"
REF_DEVICE="cuda:0"
VLLM_DEVICE="cuda:1"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs"

echo "Starting DPO Training..."
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

# ================= 启动命令 =================

# nohup 后台运行 (可选)，这里直接前台运行方便看日志
# 使用 set -x 可以打印执行的具体命令
set -x

python cs336_alignment/train_dpo.py \
    --seed 42 \
    --train_batch_size 64 \
    --gradient_accumulation_steps 16 \
    --model_id "$MODEL_PATH" \
    --data_dir "$TRAIN_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --device "$POLICY_DEVICE" \
    --ref_device "$REF_DEVICE" \
    --lr 5e-7 \
    --num_epochs 1 \
    --beta 0.1 \
    --max_val_samples 1000 \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$wandb_run_name" \
    --eval_every_steps 50 \
    --save_every_steps 500 \
    --vllm_device "$VLLM_DEVICE" \
    --vllm_gpu_util 0.3 \
    --gsm8k_path "$GSM8K_PATH" \
    --mmlu_path "$MMLU_PATH" \
    --enable_eval \
    --eval_mmlu \
    --eval_gsm8k \
    2>&1 | tee "logs/dpo_train_$(date +%Y%m%d_%H%M%S).log"

# 如果不需要 MMLU 评估，注释掉 --eval_mmlu 和 --mmlu_path 即可
