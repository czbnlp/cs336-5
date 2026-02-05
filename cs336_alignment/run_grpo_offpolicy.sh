#!/bin/bash

# ================= 1. 基础路径与项目配置 =================
BASE_MODEL="model/Qwen2.5-Math-1.5B" 

TRAIN_DATA="data/gsm8k/train.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"
PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
OUTPUT_BASE="result/grpo_offpolicy_study"
WANDB_PROJECT="cs336-grpo-after-base-offpolicy"

# ================= 2. 学习率锚点逻辑 =================
# 锚点：1个 Epoch, 256 Batch 时的基准学习率
ANCHOR_LR="0.00003"  
ANCHOR_BATCH=256

# ================= 3. 定义具体的实验配置列表 =================
# 格式为 "Epochs:TrainBatchSize"
CONFIGS=(
    # "1:256"   
    # "1:64"    
    # "3:128" 
    # "3:256"    
    "5:64"  
    "5:128"  
    "5:256"  
)

# ================= 4. 硬件与通用超参 =================
MICRO_BS=8        # 物理显存 Batch Size
ROLLOUT_SIZE=256   # 采样总数
GROUP_SIZE=8       # G
N_STEPS=200        # 总迭代步数
SEED=42

# ================= 5. 循环执行实验 =================
TOTAL_EXPS=${#CONFIGS[@]}
CURR_EXP=0

for CFG in "${CONFIGS[@]}"; do
    ((CURR_EXP++))

    # 解析配置：将 "3:128" 分解为 E=3, TB=128
    IFS=":" read -r E TB <<< "$CFG"

    # --- 关键逻辑：动态计算超参数 ---
    # 1. 计算学习率: LR = Anchor_LR * (1/sqrt(E)) * (TB / Anchor_Batch)
    LR=$(awk "BEGIN {print $ANCHOR_LR * (1/sqrt($E)) * ($TB/$ANCHOR_BATCH)}")
    
    # 2. 计算梯度累积步数: AccumSteps = TB / MicroBS
    ACCUM_STEPS=$((TB / MICRO_BS))
    
    # 3. 构造运行名称
    RUN_NAME="E${E}_TB${TB}_LR${LR}"
    CURRENT_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"

    echo "========================================================="
    echo "🚀 [进度 $CURR_EXP/$TOTAL_EXPS] 启动实验: $RUN_NAME"
    echo "📊 配置: Epochs=$E | TrainBatch=$TB | AccumSteps=$ACCUM_STEPS"
    echo "📈 计算得出学习率: $LR"
    echo "📂 保存目录: $CURRENT_OUTPUT_DIR"
    echo "========================================================="

    # 执行训练
    uv run python cs336_alignment/train_grpo.py \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --test_data_path "$VAL_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "$CURRENT_OUTPUT_DIR" \
        --n_grpo_steps "$N_STEPS" \
        --lr "$LR" \
        --rollout_batch_size "$ROLLOUT_SIZE" \
        --group_size "$GROUP_SIZE" \
        --train_batch_size "$TB" \
        --gradient_accumulation_steps "$ACCUM_STEPS" \
        --epochs_per_rollout_batch "$E" \
        --loss_type "grpo_clip" \
        --length_norm_type "mask_normalize" \
        --device cuda:0 \
        --vllm_device cuda:1 \
        --vllm_gpu_util 0.3 \
        --eval_every_steps 8 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME" \
        --seed "$SEED" \
        --use_std_normalization # 默认开启标准差归一化

    # 错误处理
    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 失败，停止后续脚本。"
        exit 1
    fi
    
    echo "✅ 实验 $RUN_NAME 完成！"
    echo "---------------------------------------------------------"
    sleep 10 # 缓冲，确保 vLLM 释放显存
done

echo "🎉 预定义实验列表全部执行完毕！"