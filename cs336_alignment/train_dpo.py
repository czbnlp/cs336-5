import torch
import json
import os
import argparse
import random
import wandb
import numpy as np
import re
import gzip
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch
import torch.nn.functional as F

# 假设 load_anthropic_hh_dataset 已经在 dpo_utils 中正确定义并支持 split 参数
# 如果没有，请将该函数的定义也粘贴到本文件中
from cs336_alignment.dpo_utils import load_anthropic_hh_dataset, compute_dpo_loss

def evaluate_validation_set(policy, ref_model, tokenizer, val_data, beta, device):
    """
    在验证集上计算 Average Loss 和 Accuracy
    """
    policy.eval()
    losses, accs = [], []
    
    # 避免验证集过大
    eval_iter = tqdm(val_data, desc="DPO Validation")
    
    with torch.no_grad():
        for item in eval_iter:
            # 简单的单样本处理 (如果你想加速可以改 batch)
            try:
                loss, metrics = compute_dpo_loss(
                    policy, ref_model, tokenizer, beta,
                    item['instruction'], item['chosen'], item['rejected']
                )
                losses.append(loss.item())
                accs.append(metrics['accuracy'])
            except Exception as e:
                # 忽略超长或其他 tokenize 错误
                continue
    
    avg_loss = np.mean(losses) if losses else 0
    avg_acc = np.mean(accs) if accs else 0
    return avg_loss, avg_acc

# ==========================================
#  vLLM 与 外部评估工具
# ==========================================

def init_vllm(model_id, device, seed, gpu_memory_utilization):
    """初始化 vLLM 实例"""
    with patch("torch.distributed.get_world_size", return_value=1), \
         patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None):
        return LLM(
            model=model_id, 
            device=device, 
            dtype=torch.bfloat16,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            trust_remote_code=True
        )

def update_vllm_weights(policy, vllm_instance):
    """将训练中的 Policy 权重同步给 vLLM"""
    print("\n[Sync] Syncing Policy weights to vLLM for evaluation...")
    state_dict = policy.state_dict()
    llm_model = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    print("[Sync] Done.")

def evaluate_gsm8k(vllm_llm, data_path, num_samples=1000):
    """GSM8K 评估"""
    if not os.path.exists(data_path): return None
    
    prompts, golds = [], []
    with open(data_path, "r") as f:
        lines = f.readlines()
        if num_samples is None:
            num_samples = len(lines)
        sample_indices = random.sample(range(len(lines)), min(num_samples, len(lines)))
        for idx in sample_indices:
            item = json.loads(lines[idx])
            # GSM8K 模板
            p = f"Question: {item['question']}\nLet's think step by step\nAnswer:"
            gold = item['answer'].split("####")[-1].strip()
            prompts.append(p)
            golds.append(gold)
            
    outputs = vllm_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1024))
    
    correct = 0
    for i, out in enumerate(outputs):
        pred_text = out.outputs[0].text
        # 简单提取逻辑
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", pred_text.replace(",", ""))
        if nums:
            if abs(float(nums[-1]) - float(golds[i].replace(",", ""))) < 1e-4:
                correct += 1
    return correct / len(prompts)

def evaluate_mmlu_pro(vllm_llm, data_path, num_samples=1000):
    """MMLU-Pro 评估"""
    if not os.path.exists(data_path): return None
    try:
        df = pd.read_parquet(data_path)
    except:
        return None
    
    if num_samples is None:
        num_samples = len(df) 
    df = df.sample(n=min(num_samples, len(df)))
    prompts, golds = [], []
    
    for _, row in df.iterrows():
        opts_str = "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(row['options'])])
        p = f"Question: {row['question']}\n{opts_str}\nAnswer with the option letter directly."
        prompts.append(p)
        golds.append(row['answer'])

    outputs = vllm_llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=100))
    
    correct = 0
    for i, out in enumerate(outputs):
        pred_text = out.outputs[0].text.strip().upper()
        match = re.search(r"([A-J])", pred_text)
        pred = match.group(1) if match else "Z"
        if pred == golds[i]:
            correct += 1
    return correct / len(prompts)

# ==========================================
#  主训练流程
# ==========================================
def run_dpo_training(args):
    # 计算微批次大小
    assert args.train_batch_size % args.gradient_accumulation_steps == 0, "batch_size 必须能被 grad_acc_steps 整除"
    micro_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    print(f"Update Period: {args.train_batch_size} samples")
    print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"Micro Batch Size: {micro_batch_size}")

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_data = load_anthropic_hh_dataset(args.data_dir, split="train")
    val_data = load_anthropic_hh_dataset(args.data_dir, split="test")
    
    max_val_samples = args.max_val_samples
    if len(val_data) > max_val_samples:
        random.seed(args.seed)
        val_data = random.sample(val_data, max_val_samples)

    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(args.device)
    policy.gradient_checkpointing_enable()

    ref_device = args.ref_device if args.ref_device else args.device
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(ref_device)
    ref_model.eval()
    ref_model.requires_grad_(False)

    optimizer = AdamW(policy.parameters(), lr=args.lr)

    vllm_inst = None
    if args.enable_eval:
        try:
            vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)
        except Exception as e:
            print(f"vLLM Init failed: {e}. Disabling external eval.")
            args.enable_eval = False

    def run_evaluation_suite(step_num):
        print(f"\n[Step {step_num}] Running Full Evaluation Suite...")
        logs = {}
        policy.eval() 
        val_loss, val_acc = evaluate_validation_set(policy, ref_model, tokenizer, val_data, args.beta, args.device)
        logs["val/loss"] = val_loss
        logs["val/accuracy"] = val_acc
        
        if args.enable_eval and vllm_inst is not None:
            update_vllm_weights(policy, vllm_inst)
            if args.eval_gsm8k:
                acc = evaluate_gsm8k(vllm_inst, args.gsm8k_path)
                if acc is not None: logs["eval/gsm8k"] = acc
            if args.eval_mmlu:
                acc = evaluate_mmlu_pro(vllm_inst, args.mmlu_path)
                if acc is not None: logs["eval/mmlu-pro"] = acc

        wandb.log(logs, step=step_num)
        policy.train()

    # Step 0 评估
    run_evaluation_suite(step_num=0)

    # 5. 训练循环
    global_step = 0
    # 总更新步数是以 train_batch_size 为单位计算的
    total_update_steps = (len(train_data) * args.num_epochs) // args.train_batch_size
    progress_bar = tqdm(total=total_update_steps, desc="DPO Update Steps")
    
    policy.train()
    
    for epoch in range(args.num_epochs):
        random.shuffle(train_data)
        
        # 指标累积变量（针对一个完整的 train_batch_size）
        current_batch_loss = 0
        current_batch_metrics = {"acc": [], "margin": []}
        
        # 以 micro_batch_size 为步长遍历数据
        for i in range(0, len(train_data), micro_batch_size):
            micro_batch = train_data[i : i + micro_batch_size]
            if len(micro_batch) < micro_batch_size: continue # 舍弃末尾不够一个 micro_batch 的数据

            # 处理一个 Micro-Batch
            for item in micro_batch:
                try:
                    loss, metrics = compute_dpo_loss(
                        policy, ref_model, tokenizer, args.beta,
                        item['instruction'], item['chosen'], item['rejected']
                    )
                    # 梯度缩放：loss 应该除以整个 train_batch_size (64)
                    # 这样 backward 16 次 micro_batch(每批4个) 后，总梯度正好是 64 个样本的平均
                    scaled_loss = loss / args.train_batch_size
                    scaled_loss.backward()
                    
                    current_batch_loss += loss.item()
                    current_batch_metrics['acc'].append(metrics['accuracy'])
                    current_batch_metrics['margin'].append(metrics['reward_margin'])
                except Exception as e:
                    print(f"Error at micro-batch: {e}")
                    continue

            # 判断是否到了执行 Optimizer Step 的时候
            # 即检查处理过的样本数是否达到了 train_batch_size
            samples_processed = (len(current_batch_metrics['acc']))
            if samples_processed >= args.train_batch_size:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 计算并记录当前更新步的平均指标
                avg_loss = current_batch_loss / args.train_batch_size
                avg_acc = np.mean(current_batch_metrics['acc'])
                avg_margin = np.mean(current_batch_metrics['margin'])
                
                wandb.log({
                    "train/loss": avg_loss,
                    "train/accuracy": avg_acc,
                    "train/margin": avg_margin,
                    "step": global_step
                })
                
                # 重置累积变量
                current_batch_loss = 0
                current_batch_metrics = {"acc": [], "margin": []}
                progress_bar.update(1)
                
                # 定期评估
                if global_step % args.eval_every_steps == 0:
                    run_evaluation_suite(step_num=global_step)

                # 定期保存
                if global_step % args.save_every_steps == 0:
                    path = os.path.join(args.output_dir, f"step_{global_step}")
                    policy.save_pretrained(path)
                    tokenizer.save_pretrained(path)

    print("Training Finished.")
    policy.save_pretrained(os.path.join(args.output_dir, "final"))
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="model/Qwen2.5-7B")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/dpo")
    
    # 调整后的 Batch 相关参数
    parser.add_argument("--train_batch_size", type=int, default=64, help="总更新批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ref_device", type=str, default=None)
    parser.add_argument("--enable_eval", action="store_true")
    parser.add_argument("--eval_every_steps", type=int, default=100)
    parser.add_argument("--save_every_steps", type=int, default=200)
    parser.add_argument("--max_val_samples", type=int, default=1000)
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.4)
    parser.add_argument("--eval_gsm8k", action="store_true")
    parser.add_argument("--gsm8k_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--eval_mmlu", action="store_true")
    parser.add_argument("--mmlu_path", type=str, default="data/MMLU-Pro/test.parquet")
    parser.add_argument("--wandb_project", type=str, default="cs336-dpo")
    parser.add_argument("--wandb_run_name", type=str, default="dpo-exp")

    args = parser.parse_args()
    run_dpo_training(args)
