#!/bin/bash
export CUDA_VISIBLE_DEVICES='2,3'
# 模型与数据
MODEL_ID="model/Qwen2.5-Math-1.5B"

# TRAIN_DATA="data/gsm8k/train_sft_reason_gsm8k_raw.jsonl" # 原始数据
TRAIN_DATA="data/gsm8k/train_sft_reason_gsm8k_r1.jsonl"  # r1
# 验证集 (Python 脚本内会自动处理格式)
VAL_DATA="data/gsm8k/test.jsonl"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"

# 输出与日志
OUTPUT_BASE="result/sft-gsm8k-r1"
WANDB_PROJECT="cs336-sft-gsm8k-raw"

# ================= 2. 硬件与资源配置 =================
DEVICE="cuda:0"
VLLM_DEVICE="cuda:1"

VLLM_GPU_UTIL=0.6

# ================= 3. 训练超参数配置 =================
LR=2e-5
BATCH_SIZE=64
MICRO_BATCH_SIZE=4
MAX_TOKENS=1024
SEED=42
gradient_accumulation_steps=32

# 评估设置
EVAL_FREQ=8       # 每多少 step 评估一次
EVAL_SAMPLES=1319   # test dataset 一共有1319条

# ================= 4. 实验变量循环 =================
# 数据量消融 (填入具体的整数)
# DATASET_SIZES=(128 256 512 1024 7473)
DATASET_SIZES=(7473)

# 步数消融 (Total Steps)
STEPS_LIST=(256)

# 是否过滤正确答案 (开关)
# 如果需要开启过滤，设置为 "--filter_correct"
# 如果不需要，设置为空字符串 ""
# FILTER_ARG="--filter_correct" 
FILTER_ARG="--filter_correct" 

for STEPS in "${STEPS_LIST[@]}"; do
    for SIZE in "${DATASET_SIZES[@]}"; do
        
        # 构造 Run Name
        RUN_NAME="sft_size${SIZE}_steps${STEPS}_gsm8k_r1"
        CURRENT_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"

        echo "========================================================="
        echo "🚀 启动实验: $RUN_NAME"
        echo "📚 数据量: $SIZE | 👣 Max Steps: $STEPS"
        echo "🔧 Filter: ${FILTER_ARG:-'OFF'}"
        echo "========================================================="

        # 执行训练指令 (包含所有参数)
        uv run python cs336_alignment/train_sft_step.py \
            --model_id "$MODEL_ID" \
            --train_data_path "$TRAIN_DATA" \
            --val_data_path "$VAL_DATA" \
            --prompt_path "$PROMPT_TEMPLATE" \
            --output_dir "$CURRENT_OUTPUT_DIR" \
            --dataset_size "$SIZE" \
            --max_steps "$STEPS" \
            --lr "$LR" \
            --gradient_accumulation_steps "$gradient_accumulation_steps" \
            --batch_size "$BATCH_SIZE" \
            --micro_batch_size "$MICRO_BATCH_SIZE" \
            --seed "$SEED" \
            --max_tokens "$MAX_TOKENS" \
            --device "$DEVICE" \
            --vllm_device "$VLLM_DEVICE" \
            --vllm_gpu_util "$VLLM_GPU_UTIL" \
            --eval_every_steps "$EVAL_FREQ" \
            --max_eval_samples "$EVAL_SAMPLES" \
            --wandb_project "$WANDB_PROJECT" \
            --wandb_run_name "$RUN_NAME" \
            $FILTER_ARG

        if [ $? -ne 0 ]; then
            echo "❌ 实验 $RUN_NAME 失败！"
            exit 1
        fi

        echo "✅ 实验 $RUN_NAME 完成！"
        sleep 5
    done
done

echo "🎉 所有实验结束"