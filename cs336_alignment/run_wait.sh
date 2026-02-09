# ================= 显存检查配置 =================
GPU_ID=0
THRESHOLD_MB=10240  # 10G = 10 * 1024 MB
CHECK_INTERVAL=60s   # 每 60 秒检查一次

echo "⏳ 等待 GPU:$GPU_ID 的显存占用降至 ${THRESHOLD_MB}MB 以下..."

# 循环检查显存
while true; do
    # 获取 GPU 1 当前已使用的显存（单位：MB）
    CURRENT_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID)
    
    if [ "$CURRENT_USED" -lt "$THRESHOLD_MB" ]; then
        echo "✅ 显存已释放 (当前: ${CURRENT_USED}MB). 开始执行指令..."
        break
    else
        echo "😴 显存占用仍较高 (当前: ${CURRENT_USED}MB). ${CHECK_INTERVAL}秒后重试..."
        sleep $CHECK_INTERVAL
    fi
done

# ================= 后续执行指令 =================
echo "🚀 启动后续任务..."
export CUDA_VISIBLE_DEVICES='0,1'
uv run bash cs336_alignment/run_dpo.sh