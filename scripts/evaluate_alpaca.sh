export CUDA_VISIBLE_DEVICES='0,1'
export OPENAI_BASE_URL="http://localhost:8010/v1"
export OPENAI_API_KEY="vllm-token"
uv run alpaca_eval --model_outputs 'result/alpaca_Qwen2.5-7B-Base_result.json' \
  --annotators_config 'scripts/alpaca_eval_vllm_qwen2_5_72b_fn' \
  --base-dir '.'