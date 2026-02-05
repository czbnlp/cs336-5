import json
import re
import os

# ================= 配置区 =================
INPUT_FILE = "data/gsm8k/train.jsonl"
OUTPUT_FILE = "data/gsm8k/train_sft_reason_gsm8k_raw.jsonl"
# 你的标准系统提示词模版
SYSTEM_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
    "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. "
    "The final result only requires a direct output of the answer without any explanations.\n"
    "User: {question}\n"
    "Assistant: <think>"
)

def clean_gsm8k_reasoning(text):
    """
    清洗 GSM8k 的推理过程：
    1. 移除 <<48/2=24>> 这种计算标签
    2. 确保结尾没有冗余
    """
    # 移除 <<...>> 标签
    cleaned = re.sub(r"<<.*?>>", "", text)
    return cleaned.strip()

def process_line(line):
    item = json.loads(line)
    question = item['question']
    answer_field = item['answer']

    # 1. 拆分推理过程和最终答案 (GSM8k 格式是：推理 #### 数字)
    if "####" not in answer_field:
        return None
    
    reasoning_part, gold_answer = answer_field.split("####")
    reasoning_part = clean_gsm8k_reasoning(reasoning_part)
    gold_answer = gold_answer.strip()

    # 2. 构造 Prompt (完全匹配你提供的格式)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(question=question)

    # 3. 构造 Response (注意空格，确保 reward 识别)
    # 格式：推理过程 </think> <answer> 答案 </answer>
    response = f"{reasoning_part} </think> <answer>{gold_answer}</answer>"

    # 4. 构造完整条目
    return {
        "question": question,
        "prompt": prompt,
        "response": response,
        "gold": gold_answer,
        "is_correct": True, # 原始数据集默认为真
        "reward_details": {
            "format_reward": 1.0,
            "answer_reward": 1.0,
            "reward": 1.0
        }
    }

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    count = 0
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            if not line.strip():
                continue
            
            new_item = process_line(line)
            if new_item:
                f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                count += 1

    print(f"转换完成！处理了 {count} 条数据。")
    print(f"输出文件：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()