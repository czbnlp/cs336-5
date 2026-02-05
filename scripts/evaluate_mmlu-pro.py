import os
import json
import time
import re
import random
import datasets
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# --- 配置 ---
VLLM_API_URL = "http://localhost:8010/v1"
DATA_PATH = "data/MMLU-Pro/data/test-00000-of-00001.parquet"
MODEL_NAME = "Qwen2.5-7B-Instruct"
# MODEL_NAME = "Qwen2.5-7B-Base"
# 确保 result 目录存在
os.makedirs("result", exist_ok=True)
OUTPUT_FILE = f"result/mmlu-pro_{MODEL_NAME}_baseline_results.json"
MAX_WORKERS = 300

# 初始化 OpenAI 客户端
client = OpenAI(base_url=VLLM_API_URL, api_key="vllm-token")
random.seed(42)

def form_options(options: list):
    """格式化 A-J 选项"""
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str

def get_prediction(output):
    """从模型生成的 CoT 内容中提取答案 (A-J)"""
    # 匹配 "The answer is (A)" 或 "answer is A"
    pattern = r"answer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        # 备选逻辑：寻找文本中出现的最后一个独立的 A-J 字母
        alt_pattern = r"\b([A-J])\b"
        matches = re.findall(alt_pattern, output)
        if matches:
            return matches[-1].upper()
        return random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])

def call_vllm_api(query):
    """调用 API 进行零样本推理"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledge expert. Please reason step by step and then derive your final answer as `The answer is (X)` at the end."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            temperature=0.0,
            max_tokens=2048, 
            top_p=1.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"ERROR: {str(e)}"

def process_entry(entry):
    """处理单个样本"""
    query = 'Q: ' + entry['question'] + '\n' + form_options(entry['options']) + '\n'
    
    answer = call_vllm_api(query)
    
    if "ERROR" in answer:
        return {"error": answer, "entry": entry}
        
    prediction = get_prediction(answer)
    # entry['answer'] 已经是 'A', 'B' 等字符
    is_correct = (entry["answer"] == prediction)
    
    return {
        "category": entry['category'],
        "question": entry['question'],
        "gold": entry["answer"],
        "prediction": prediction,
        "solution": answer,
        "is_correct": is_correct
    }

def main():
    # 1. 加载本地数据 (使用 Pandas 避开 datasets 库的本地加载 Bug)
    print(f"Loading MMLU-Pro test set from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return
        
    df = pd.read_parquet(DATA_PATH)
    test_entries = df.to_dict('records')
    print(f"Successfully loaded {len(test_entries)} examples.")

    categories = ['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                  'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                  'psychology', 'history']

    all_results = []
    per_category_accuracy = {c: [0, 0] for c in categories} # [正确数, 总数]

    # 2. 预检查服务器
    try:
        client.models.list()
        print("Connected to vLLM server.")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    # 3. 多线程并发请求
    print(f"Starting Zero-shot inference with {MAX_WORKERS} workers...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_entry = {executor.submit(process_entry, entry): entry for entry in test_entries}
        
        for future in tqdm(as_completed(future_to_entry), total=len(test_entries)):
            res = future.result()
            
            if "error" in res:
                continue
                
            all_results.append(res)
            cat = res['category']
            if cat in per_category_accuracy:
                per_category_accuracy[cat][1] += 1
                if res['is_correct']:
                    per_category_accuracy[cat][0] += 1

    duration = time.time() - start_time

    # 4. 计算并打印统计结果
    print('\n' + '='*60)
    print(f"{'Category':25s} | {'Accuracy':10s} | {'Correct/Total'}")
    print('-'*60)
    
    total_correct = 0
    total_count = 0
    
    for cat in categories:
        corr, total = per_category_accuracy[cat]
        acc = corr / total if total > 0 else 0
        print(f"{cat:25s} | {acc:10.4f} | {corr}/{total}")
        total_correct += corr
        total_count += total
    
    overall_acc = total_correct / total_count if total_count > 0 else 0
    print('='*60)
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"Time: {duration/60:.2f} mins | Throughput: {total_count/duration:.2f} q/s")

    # 5. 保存结果到磁盘
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()