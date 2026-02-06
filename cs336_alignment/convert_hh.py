import random
import gzip
import json
import os
from typing import List, Dict

def load_anthropic_hh_dataset(root_dir: str) -> List[Dict[str, str]]:
    """
    针对实际目录结构加载 Anthropic HH 数据集。
    目录结构示例: root_dir/harmless-base/train.jsonl.gz
    """
    subsets = [
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled"
    ]
    
    combined_data = []
    
    for subset in subsets:
        file_path = os.path.join(root_dir, subset, "test.jsonl.gz")
        
        if not os.path.exists(file_path):
            print(f"警告: 未找到文件 {file_path}，跳过该子集。")
            continue
            
        print(f"正在加载子集: {subset}...")
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                chosen_raw = item['chosen']
                rejected_raw = item['rejected']
                
                # --- 核心清洗逻辑 (PDF 要求) ---
                
                # 1. 只保留单轮对话: 如果字符串中出现两次 "Human:"，说明是多轮，丢弃
                if chosen_raw.count("\n\nHuman:") > 1:
                    continue
                
                try:
                    # 2. 提取指令 (Instruction) 和 回答 (Responses)
                    # Anthropic 格式通常为: "\n\nHuman: [问题]\n\nAssistant: [回答]"
                    
                    # 以 "\n\nAssistant:" 为界切分
                    # parts[0] 是 "\n\nHuman: [问题]"
                    # parts[1] 是 "[回答]"
                    chosen_parts = chosen_raw.split("\n\nAssistant:")
                    rejected_parts = rejected_raw.split("\n\nAssistant:")
                    
                    if len(chosen_parts) < 2 or len(rejected_parts) < 2:
                        continue
                    
                    # 去掉 "\n\nHuman: " 前缀并修剪空格
                    instruction = chosen_parts[0].replace("\n\nHuman:", "").strip()
                    chosen_response = chosen_parts[1].strip()
                    rejected_response = rejected_parts[1].strip()
                    
                    combined_data.append({
                        "instruction": instruction,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                        "subset": subset # 记录来源以便后续分析
                    })
                    
                except Exception as e:
                    # 忽略格式异常的极个别样本
                    continue
                    
    print(f"✅ 加载完成！共获得 {len(combined_data)} 条单轮对话偏好数据。")
    return combined_data

def inspect_samples(data):
    # 分别抽取 harmless 和 helpful 的样本
    harmless_samples = [d for d in data if "harmless" in d['subset']]
    helpful_samples = [d for d in data if "helpful" in d['subset']]
    print(len(harmless_samples), len(helpful_samples))
    print("=== 3个 Harmless 样本分析 ===")
    for s in random.sample(harmless_samples, 3):
        print(f"[指令]: {s['instruction']}")
        print(f"  ✅ [选中的]: {s['chosen'][:100]}...")
        print(f"  ❌ [拒绝的]: {s['rejected'][:100]}...")
        print("-" * 20)

    print("\n=== 3个 Helpful 样本分析 ===")
    for s in random.sample(helpful_samples, 3):
        print(f"[指令]: {s['instruction']}")
        print(f"  ✅ [选中的]: {s['chosen'][:100]}...")
        print(f"  ❌ [拒绝的]: {s['rejected'][:100]}...")
        print("-" * 20)


data = load_anthropic_hh_dataset("data/hh-rlhf")

inspect_samples(data)
