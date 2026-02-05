import json
import re
import os
import time
import threading
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置区 =================
client = OpenAI(
    base_url='https://qianfan.baidubce.com/v2',
    api_key=''
)

MODEL_NAME = "deepseek-r1-250528"
INPUT_FILE = "data/gsm8k/a.jsonl"
OUTPUT_FILE = "data/gsm8k/train_r1_distill_final111.jsonl"
PROMPT_TEMPLATE_FILE = "cs336_alignment/prompts/r1_zero.prompt"

# 并发设置
MAX_WORKERS = 1000  # 同时进行的请求数，根据你的 RPM 配额调整
file_lock = threading.Lock() # 用于多线程安全写入

# ================= 工具函数 =================

def extract_number(text):
    if '<answer>' not in text and '</answer>' not in text:
        return f'<answer>{text}</answer>'

    return text

def get_existing_questions(output_file):
    if not os.path.exists(output_file): return set()
    questions = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                questions.add(json.loads(line)['question'])
            except: continue
    return questions

# ================= 核心处理单例 =================

def process_item(item, train_template, api_request_template):
    """处理单个 GSM8K 条目的函数，供线程池调用"""
    question = item['question']
    gold = item['answer'].split('####')[-1].strip()
    api_prompt_content = api_request_template.replace("{question}", question)
    
    # 简单的重试机制
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": api_prompt_content}],
                temperature=0.6,
                timeout=120 # 防止长文本响应超时
            )


            res_msg = completion.choices[0].message
            r1_content = res_msg.content or ""
            r1_reasoning = getattr(res_msg, 'reasoning_content', "").strip()
            print(f"处理问题: {question}")
            print(f"模型回复: {r1_content}")
            print(f"推理过程: {r1_reasoning}")
            # 验证 R1 是否算对
            r1_answer = extract_number(r1_content)
            if not r1_answer:
                r1_answer = extract_number(r1_reasoning)
      
            full_train_prompt = train_template.replace("{question}", question)
            final_answer = r1_reasoning + '</think> ' + r1_answer
            result = {
                "question": question,
                "prompt": full_train_prompt+"<think>",
                "content": r1_answer,
                "reasoning_content": r1_reasoning,
                "final_answer": final_answer,
                "gold": gold
            }
            return result
 

        except Exception as e:
            print('请求出错，正在重试...', str(e))
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1)) # 指数退避
                continue
            return {"error": f"API Error after {max_retries} attempts: {str(e)}", "question": question}
    return None

# ================= 主程序 =================

def main():
    # 1. 加载模板
    with open(PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        train_template = f.read().strip()
    api_request_template = train_template

    # 2. 加载原始数据并去重
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]

    processed_questions = get_existing_questions(OUTPUT_FILE)
    tasks = [item for item in raw_data if item['question'] not in processed_questions]
    
    print(f"待处理任务数: {len(tasks)} (并发数: {MAX_WORKERS})")

    # 3. 线程池并发执行
    results_count = 0
    errors_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_item = {executor.submit(process_item, item, train_template, api_request_template): item for item in tasks}
        
        # 使用 tqdm 显示进度
        pbar = tqdm(total=len(tasks), desc="处理进度")
        
        for future in as_completed(future_to_item):
            data = future.result()
            
            if data:
                if "error" in data:
                    errors_count += 1
                else:
                    # 线程安全写入文件
                    with file_lock:
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    results_count += 1
            
            pbar.update(1)
            pbar.set_postfix({"已存": results_count})

    pbar.close()
    print(f"\n任务完成！\n- 成功保存正确样本: {results_count}\n- 忽略(算错或错误): {len(tasks) - results_count}")

if __name__ == "__main__":
    main()