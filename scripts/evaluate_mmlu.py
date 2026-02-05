import os
import json
import time
import re
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 配置 ---
VLLM_API_URL = "http://localhost:8010/v1"
MMLU_DATA_DIR = "data/mmlu/test"
# 请确保此处 MODEL_NAME 与你 vllm serve 启动时的 --served-model-name 一致
# MODEL_NAME = "Qwen2.5-7B-Base" 
MODEL_NAME = "Qwen2.5-7B-Instruct" 
OUTPUT_FILE = f"result/mmlu_{MODEL_NAME}_baseline_results.json"
MAX_WORKERS = 100

# 初始化客户端
client = OpenAI(base_url=VLLM_API_URL, api_key="vllm-token")

# --- 作业模板 ---
SYSTEM_PROMPT = (
    "Below is a list of conversations between a human and an AI assistant (you).\n"
    "Users place their queries under \"# Query:\", and your responses are under \"# Answer:\".\n"
    "You are a helpful, respectful, and honest assistant.\n"
    "You should always answer as helpfully as possible while ensuring safety.\n"
    "Your answers should be well-structured and provide detailed information. They should also have an engaging tone.\n"
    "Your responses must not contain any fake, harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, even if it may be helpful.\n"
    "Your response must be socially responsible, and thus you can reject to answer some controversial topics."
)

MMLU_TASK_TEMPLATE = (
    "Answer the following multiple choice question about {subject}. Respond with a single "
    "sentence of the form \"The correct answer is _\", filling the blank with the letter "
    "corresponding to the correct answer (i.e., A, B, C or D).\n\n"
    "Question: {question}\n"
    "A. {optA}\n"
    "B. {optB}\n"
    "C. {optC}\n"
    "D. {optD}\n"
    "Answer:"
)

def parse_mmlu_response(model_response: str) -> str | None:
    if not model_response: return None
    # 优先匹配标准格式
    match_std = re.search(r"[Tt]he correct answer is\s*([A-D])", model_response)
    if match_std: return match_std.group(1).upper()
    # 备选匹配独立字母
    match_alt = re.search(r"\b([A-D])\b", model_response)
    if match_alt: return match_alt.group(1).upper()
    return None

def load_all_mmlu_tests(data_dir):
    all_items = []
    print(f"Loading CSV files from {data_dir}...")
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith("_test.csv"):
                subject = file.replace("_test.csv", "").replace("_", " ")
                try:
                    df = pd.read_csv(os.path.join(root, file), header=None)
                    df.columns = ["question", "A", "B", "C", "D", "answer"]
                    for _, row in df.iterrows():
                        user_query = MMLU_TASK_TEMPLATE.format(
                            subject=subject, question=row["question"],
                            optA=row["A"], optB=row["B"], optC=row["C"], optD=row["D"]
                        )
                        full_content = f"{SYSTEM_PROMPT}\n\n# Query:\n```\n{user_query}\n```\n\n# Answer:\n```\n"
                        all_items.append({
                            "content": full_content,
                            "gold": str(row["answer"]).strip().upper(),
                            "subject": subject
                        })
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    return all_items

def call_api(item):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": item["content"]}],
            temperature=0.0,
            max_tokens=32,
            stop=["# Query:", "```"]
        )
        gen_text = response.choices[0].message.content
        pred = parse_mmlu_response(gen_text)
        return {
            "subject": item["subject"],
            "gold": item["gold"],
            "pred": pred,
            "output": gen_text,
            "is_correct": (pred == item["gold"])
        }
    except Exception as e:
        return {"error": str(e)}

def main():
    # 1. 预检查服务器连接
    print(f"Connecting to vLLM server at {VLLM_API_URL}...")
    try:
        client.models.list()
        print("Connected successfully.")
    except Exception as e:
        print(f"Could not connect to server: {e}. Please ensure vLLM serve is running.")
        return

    # 2. 加载数据
    all_items = load_all_mmlu_tests(MMLU_DATA_DIR)
    total_questions = len(all_items)
    
    # 3. 执行并发推理
    all_results = []
    start_time = time.time()

    print(f"Starting evaluation on {total_questions} questions with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {executor.submit(call_api, item): item for item in all_items}
        for future in tqdm(as_completed(future_to_item), total=total_questions):
            result = future.result()
            if "error" not in result:
                all_results.append(result)
            else:
                # 记录发生的 API 错误
                all_results.append({"subject": "N/A", "error": result["error"], "is_correct": False, "pred": None})

    duration = time.time() - start_time

    # 4. 计算指标 (Metrics)
    # 过滤掉请求失败的项以进行准确统计
    valid_results = [r for r in all_results if "error" not in r]
    correct_count = sum(1 for r in valid_results if r["is_correct"])
    parse_failed = sum(1 for r in valid_results if r["pred"] is None)
    accuracy = correct_count / len(valid_results) if valid_results else 0
    
    metrics = {
        "accuracy": accuracy,
        "throughput_ex_per_sec": len(all_results) / duration,
        "parsing_failure_count": parse_failed,
        "total_examples": len(all_results),
        "total_time_sec": duration
    }

    # 5. 保存结果 (包含 Metrics 和 Details)
    output_data = {
        "metrics": metrics,
        "details": all_results
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Throughput: {metrics['throughput_ex_per_sec']:.2f} samples/s")
    print(f"Parsing Failures: {parse_failed}")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()