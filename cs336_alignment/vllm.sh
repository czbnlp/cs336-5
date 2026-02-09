export CUDA_VISIBLE_DEVICES='2'
vllm serve model/Qwen2.5-7B \
    --served-model-name Qwen2.5-7B-Base \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.5 \
    --max-model-len 5000 \
    --host 0.0.0.0 \
    --port 8010

# vllm serve model/Qwen2.5-7B-Instruct \
#     --served-model-name Qwen2.5-7B-Instruct \
#     --tensor-parallel-size 1 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.95 \
#     --max-model-len 5000 \
#     --host 0.0.0.0 \
#     --port 8010

# vllm  serve model/Qwen2.5-72B-Instruct \
#     --served-model-name Qwen2.5-72B-Instruct \
#     --tensor-parallel-size 2 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.95 \
#     --max-model-len 5000 \
#     --host 0.0.0.0 \
#     --port 8010