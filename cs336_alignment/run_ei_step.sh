#!/bin/bash
export CUDA_VISIBLE_DEVICES=2,3
# --- 基础配置 ---
# 注意：EI 最好从一个已经 SFT 过的模型开始，而不是 Base 模型
# 这里假设你已经跑完了 SFT 并存了一个模型
BASE_MODEL="model/Qwen2.5-Math-1.5B"

TRAIN_DATA="data/gsm8k/train_sft_reason_gsm8k_raw.jsonl"
VAL_DATA="data/gsm8k/test.jsonl"
WANDB_PROJECT="cs336-ei-gsm8k_raw"

# TRAIN_DATA="data/math12k/data/train-00000-of-00001.parquet"
# VAL_DATA="data/math12k/data/test-00000-of-00001.parquet"
# WANDB_PROJECT="cs336-ei-math12k_raw"

PROMPT_TEMPLATE="cs336_alignment/prompts/r1_zero.prompt"
# PROMPT_TEMPLATE="cs336_alignment/prompts/question_only.prompt"
OUTPUT_BASE="result/ei_checkpoints"



# ================= 2. 消融实验参数空间 =================
# 根据 PDF 要求进行消融实验：
# G (Rollouts): 每个问题生成多少个候选答案
# Db (ei_batch_size): 每轮迭代采样的题目总数
# Epochs (epochs_per_ei): 对收集到的专家数据练几遍

# 你可以根据需求在数组中添加更多组合
# 格式: "G:Db:Epoch"
CONFIGS=(
    "4:512:1"
    "8:512:1"
    "4:1024:1"
    "8:1024:1"
    "4:2048:1"
    "8:2048:1"
)

# ================= 3. 硬件与通用超参 =================
LR=1e-5
GLOBAL_BS=64
MICRO_BS=8
EI_STEPS=5         # 迭代总代数
MAX_TOKENS=1024
DEVICE="cuda:0"
VLLM_DEVICE="cuda:1"
VLLM_UTIL=0.7      # 预留显存给训练卡

# ================= 4. 循环运行实验 =================
TOTAL_EXPS=${#CONFIGS[@]}
CURR_EXP=0

for CFG in "${CONFIGS[@]}"; do
    ((CURR_EXP++))
    
    # 解析组合参数
    IFS=":" read -r G DB E <<< "$CFG"
    
    # 构造唯一实验名称
    RUN_NAME="ei_G${G}_Db${DB}_E${E}_lr${LR}_add_success-question-rate"
    EXP_OUTPUT_DIR="${OUTPUT_BASE}/${RUN_NAME}"
    
    echo "========================================================="
    echo "🚀 [进度 $CURR_EXP/$TOTAL_EXPS] 启动专家迭代消融实验"
    echo "🆔 Run Name: $RUN_NAME"
    echo "📊 配置: Rollouts(G)=$G | Batch(Db)=$DB | Epochs=$E"
    echo "📂 保存路径: $EXP_OUTPUT_DIR"
    echo "========================================================="

    # 执行训练指令
    uv run python cs336_alignment/train_ei_step.py \
        --model_id "$BASE_MODEL" \
        --train_data_path "$TRAIN_DATA" \
        --val_data_path "$VAL_DATA" \
        --prompt_path "$PROMPT_TEMPLATE" \
        --output_dir "$EXP_OUTPUT_DIR" \
        --lr "$LR" \
        --batch_size "$GLOBAL_BS" \
        --micro_batch_size "$MICRO_BS" \
        --n_ei_steps "$EI_STEPS" \
        --ei_batch_size "$DB" \
        --rollouts "$G" \
        --epochs_per_ei "$E" \
        --max_tokens "$MAX_TOKENS" \
        --device "$DEVICE" \
        --vllm_device "$VLLM_DEVICE" \
        --vllm_gpu_util "$VLLM_UTIL" \
        --max_eval_samples 2000 \
        --eval_every_steps 4 \
        --wandb_project "$WANDB_PROJECT" \
        --wandb_run_name "$RUN_NAME" \
        --seed 42

    # 错误检查
    if [ $? -ne 0 ]; then
        echo "❌ 实验 $RUN_NAME 失败，停止后续任务。"
        exit 1
    fi

    echo "✅ 实验 $RUN_NAME 完成！"
    echo "---------------------------------------------------------"
    
    # 每组实验间歇 10 秒，确保显存彻底释放且 vLLM 进程关闭
    sleep 10
done

echo "🎉 所有专家迭代消融实验执行完毕！"