import json
import os
from tqdm import tqdm
# 导入作业提供的评测函数
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ================= 配置区 =================
INPUT_FILE = "data/gsm8k/train_sft_reason_r1.jsonl"
OUTPUT_FILE = "data/gsm8k/train_sft_reason_r1_final.jsonl"

def clean_verify_and_format():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0
    }
    
    print(f"开始清洗与校验数据...")
    print(f"输入: {INPUT_FILE}")
    print(f"输出: {OUTPUT_FILE}")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in):
            try:
                item = json.loads(line)
                
                # 1. 优化 Prompt：修复重复标签并去除首尾空格
                original_prompt = item.get('prompt', "")
                cleaned_prompt = original_prompt.replace("Assistant:\nAssistant:", "Assistant:").strip().replace("A conversation between User and Assistant. The User asks a question, and the Assistant solves it.", "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.")
                
                # 2. 准备判定所需的变量
                # r1_zero_reward_fn 期望输入完整的模型回复字符串
                final_answer = item.get('response', "")
                ground_truth = str(item.get('gold', ""))
             
                # 3. 使用官方 Reward Function 判定
                # 返回值示例: {"format_reward": 1.0, "answer_reward": 1.0, "reward": 1.0}
                reward_results = r1_zero_reward_fn(final_answer, ground_truth)
                
                # 4. 构造包含校验结果的新条目
                new_item = {
                    "question": item.get('question'),
                    "prompt": cleaned_prompt,
                    "response": final_answer, 
                    "gold": ground_truth,
                    "is_correct": item.get('is_correct'),   # 新增布尔字段
                    "reward_details": reward_results
                }
                
                # 5. 写入文件
                f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                
                # 统计
                stats["total"] += 1
                if item.get('is_correct'):
                    stats["correct"] += 1
                else:
                    stats["incorrect"] += 1
                
            except Exception as e:
                print(f"处理数据行时出错: {e}")
                continue

    # 打印最终统计报告
    print("\n" + "="*30)
    print("数据校验报告")
    print("-" * 30)
    print(f"总处理样本数: {stats['total']}")
    print(f"判定正确 (is_correct=True): {stats['correct']} ({(stats['correct']/stats['total']):.2%})")
    print(f"判定错误 (is_correct=False): {stats['incorrect']}")
    print("="*30)
    print(f"洁净数据已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_verify_and_format()