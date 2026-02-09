import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from cs336_alignment.parse_utils import parse_gsm8k_response

# --- 配置 ---
VLLM_API_URL = "http://localhost:8010/v1"
# MODEL_NAME = "Qwen2.5-7B-Base" # 对应你 serve 时的别名
MODEL_NAME = "Qwen2.5-7B-Instruct"
DATA_PATH = "data/gsm8k/test.jsonl"
OUTPUT_FILE = f"result/gsm8k_{MODEL_NAME}_baseline_results.json"
MAX_WORKERS = 100

client = OpenAI(base_url=VLLM_API_URL, api_key="empty")

SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer some controversial topics."
)

def load_gsm8k_data(file_path):
    items = []
    with open(file_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            # 这里的 question 是原始题目
            user_query = f"{ex['question']}\nAnswer:"
            # 组装作业要求的格式
            full_prompt = f"{SYSTEM_PROMPT}\n\n# Query:\n```\n{user_query}\n```\n\n# Answer:\n```\n"
            
            # GSM8K 答案通常在 'answer' 字段，形式如 "... #### 72"
            gold_str = ex['answer'].split("####")[-1].strip().replace(",", "")
            
            items.append({
                "prompt": full_prompt,
                "gold": float(gold_str),
                "original_question": ex['question']
            })
    return items

def call_api(item):
    try:
        # GSM8K 需要生成推理过程，max_tokens 设为 512
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": item["prompt"]}],
            temperature=0.0,
            max_tokens=512,
        )
        gen_text = response.choices[0].message.content
        pred = parse_gsm8k_response(gen_text)
        return {
            "question": item["original_question"],
            "gold": item["gold"],
            "pred": pred,
            "output": gen_text,
            "is_correct": (pred == item["gold"])
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    # 1. 加载数据
    all_items = load_gsm8k_data(DATA_PATH)
    print(f"Loaded {len(all_items)} GSM8K questions.")

    # 2. 并发评估
    all_results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call_api, item): item for item in all_items}
        for future in tqdm(as_completed(futures), total=len(all_items)):
            res = future.result()
            if "error" not in res:
                all_results.append(res)

    duration = time.time() - start_time

    # 3. 统计指标
    correct = sum(1 for r in all_results if r["is_correct"])
    parse_failed = sum(1 for r in all_results if r["pred"] is None)
    accuracy = correct / len(all_results)
    
    metrics = {
        "accuracy": accuracy,
        "throughput_ex_per_sec": len(all_results) / duration,
        "parsing_failure_count": parse_failed,
        "total_examples": len(all_results)
    }

    # 4. 保存
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"metrics": metrics, "details": all_results}, f, indent=2)

    print(f"\nGSM8K Accuracy: {accuracy:.4f}")
    print(f"Throughput: {metrics['throughput_ex_per_sec']:.2f} samples/s")

if __name__ == "__main__":
    main()