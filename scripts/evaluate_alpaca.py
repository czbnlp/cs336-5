import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置 ---
VLLM_API_URL = "http://localhost:8010/v1"
MODEL_NAME = "Qwen2.5-7B-Base" 
DATA_PATH = "/aul/homes/zyin007/zb/assignment5-alignment/data/alpaca_eval/alpaca_eval.jsonl" 
# 确保目录存在
os.makedirs("result", exist_ok=True)
OUTPUT_FILE = "result/alpaca_Qwen2.5-7B-Base_result.json"
MAX_WORKERS = 100

client = OpenAI(base_url=VLLM_API_URL, api_key="vllm-token")

def load_alpaca_data(path):
    """修复后的 JSONL 读取函数"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}\nException: {e}")
    return data

def call_api(item):
    prompt = item['instruction']
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024
        )
        gen_text = response.choices[0].message.content
        
        # 返回 AlpacaEval 要求的 JSON 格式
        return {
            "instruction": item["instruction"],
            "output": gen_text,
            "generator": MODEL_NAME,
            "dataset": item.get("dataset", "alpaca_eval")
        }
    except Exception as e:
        return {"error": str(e), "instruction": item["instruction"]}

def main():
    # 1. 加载数据 (JSONL 格式)
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return
        
    eval_set = load_alpaca_data(DATA_PATH)
    print(f"Loaded {len(eval_set)} AlpacaEval instructions.")

    # 2. 并发推理
    results = []
    start_time = time.time()

    print(f"Starting evaluation with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_api, item): item for item in eval_set}
        for future in tqdm(as_completed(futures), total=len(eval_set)):
            res = future.result()
            if "error" not in res:
                results.append(res)
            else:
                print(f"Error processing item: {res.get('instruction', 'unknown')[:50]}... -> {res['error']}")

    duration = time.time() - start_time
    # 避免除以 0
    throughput = len(results) / duration if duration > 0 else 0

    # 3. 序列化为 JSON 数组 (AlpacaEval 最终评估需要一个标准的 JSON list 文件)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Complete!")
    print(f"Total time: {duration/60:.2f} minutes")
    print(f"Throughput: {throughput:.2f} samples/s")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()