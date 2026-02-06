import torch
import torch.nn.functional as F
import random
import gzip
import json
import os
from typing import List, Dict

def get_log_probs(model, input_ids):
    """
    计算序列的总对数概率。
    """
    # [batch, seq_len, vocab]
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 移位 (Shift)：预测下一个词
    # 输入: t1, t2, t3
    # 标签: t2, t3
    shift_logits = log_probs[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # 提取实际出现的 token 的概率
    per_token_log_probs = torch.gather(
        shift_logits, dim=2, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    return per_token_log_probs.sum()

def compute_dpo_loss(
    model, 
    ref_model, 
    tokenizer, 
    beta, 
    prompt, 
    response_chosen, 
    response_rejected
):
    """
    基于你提供的 compute_dpo_loss 进行扩展，额外返回用于 WandB 监控的指标。
    """
    device = model.device

    def format_prompt(p, r):
        return (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{p}\n\n"
            f"### Response:\n{r}"
        )

    full_chosen = format_prompt(prompt, response_chosen) + tokenizer.eos_token
    full_rejected = format_prompt(prompt, response_rejected) + tokenizer.eos_token

    chosen_ids = tokenizer.encode(full_chosen, return_tensors="pt").to(device)
    rejected_ids = tokenizer.encode(full_rejected, return_tensors="pt").to(device)

    # Policy Log Probs
    lp_theta_chosen = get_log_probs(model, chosen_ids)
    lp_theta_rejected = get_log_probs(model, rejected_ids)

    # Reference Log Probs
    with torch.no_grad():
        lp_ref_chosen = get_log_probs(ref_model, chosen_ids.to(ref_model.device)).to(device)
        lp_ref_rejected = get_log_probs(ref_model, rejected_ids.to(ref_model.device)).to(device)

    # DPO Metrics Calculation
    pi_log_ratio = lp_theta_chosen - lp_theta_rejected
    ref_log_ratio = lp_ref_chosen - lp_ref_rejected
    
    logits = beta * (pi_log_ratio - ref_log_ratio)
    loss = -F.logsigmoid(logits)
    
    # --- 新增：计算监控指标 ---
    chosen_reward = beta * (lp_theta_chosen - lp_ref_chosen).detach()
    rejected_reward = beta * (lp_theta_rejected - lp_ref_rejected).detach()
    accuracy = (chosen_reward > rejected_reward).float()
    margin = chosen_reward - rejected_reward

    metrics = {
        "reward_chosen": chosen_reward.item(),
        "reward_rejected": rejected_reward.item(),
        "reward_margin": margin.item(),
        "accuracy": accuracy.item()
    }
    
    return loss, metrics


def load_anthropic_hh_dataset(root_dir: str, split: str = "train") -> List[Dict[str, str]]:
    """
    加载 Anthropic HH 数据集。
    
    Args:
        root_dir: 数据集根目录 (包含 harmless-base 等子文件夹)
        split: "train" 或 "test"，决定加载的文件名。
    """
    if split not in ["train", "test"]:
        raise ValueError(f"split 参数必须是 'train' 或 'test'，当前为: {split}")

    # 映射文件名
    file_name = "train.jsonl.gz" if split == "train" else "test.jsonl.gz"

    subsets = [
        "harmless-base",
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled"
    ]
    
    combined_data = []
    
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"数据集根目录不存在: {root_dir}")

    print(f"[{split.upper()}] 正在扫描目录: {root_dir} (目标文件: {file_name})")
    
    for subset in subsets:
        file_path = os.path.join(root_dir, subset, file_name)
        
        if not os.path.exists(file_path):
            # 对于 test 集，某些 subset 可能没有对应的文件，这是正常的，跳过即可
            # print(f"提示: 未在 {subset} 中找到 {file_name}，跳过。")
            continue
            
        print(f"[{split.upper()}] 加载子集: {subset}...")
        
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        item = json.loads(line)
                        chosen_raw = item['chosen']
                        rejected_raw = item['rejected']
                        
                        # --- 清洗逻辑 ---
                        
                        # 1. 过滤多轮对话 (保留 User -> Assistant 的单轮)
                        if chosen_raw.count("\n\nHuman:") > 1:
                            continue
                        
                        # 2. 提取指令和回复
                        # 分割点 "\n\nAssistant:"
                        chosen_parts = chosen_raw.split("\n\nAssistant:")
                        rejected_parts = rejected_raw.split("\n\nAssistant:")
                        
                        if len(chosen_parts) < 2 or len(rejected_parts) < 2:
                            continue
                        
                        # 第一部分是 Instruction (去掉 Human 前缀)
                        instruction = chosen_parts[0].replace("\n\nHuman:", "").strip()
                        # 第二部分是 Response
                        chosen_response = chosen_parts[1].strip()
                        rejected_response = rejected_parts[1].strip()
                        
                        combined_data.append({
                            "instruction": instruction,
                            "chosen": chosen_response,
                            "rejected": rejected_response,
                            "subset": subset
                        })
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"读取文件错误 {file_path}: {e}")
                    
    print(f"✅ [{split.upper()}] 加载完成！共获得 {len(combined_data)} 条单轮数据。")
    return combined_data

def inspect_samples(data):
    # 分别抽取 harmless 和 helpful 的样本
    harmless_samples = [d for d in data if "harmless" in d['subset']]
    helpful_samples = [d for d in data if "helpful" in d['subset']]
    
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


# data = load_anthropic_hh_dataset("data/hh-rlhf")

# inspect_samples(data)