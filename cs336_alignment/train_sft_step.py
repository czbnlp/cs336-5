import torch
import json
import random
import wandb
import os
import argparse
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch

# --- 导入自定义工具函数 ---
from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output, 
    sft_microbatch_train_step,
    log_generations,
    compute_entropy
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ==========================================
# 辅助函数
# ==========================================

def init_vllm(model_id, device, seed, gpu_memory_utilization):
    """初始化 vLLM 实例"""
    with patch("torch.distributed.get_world_size", return_value=1), \
         patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None):
        return LLM(
            model=model_id, 
            device=device, 
            dtype=torch.bfloat16,
            enable_prefix_caching=True, 
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed
        )

def load_policy_into_vllm_instance(policy, llm):
    """同步权重"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    print("\n[Sync] Policy weights synced to vLLM.")

def get_batch(tokenized_data, batch_size, device):
    """
    从预处理好的数据中随机采样一个 Batch。
    实现 Infinite Dataloader 的逻辑。
    """
    total_len = len(tokenized_data["input_ids"])
    batch_indices = random.sample(range(total_len), batch_size)
    
    return {
        "input_ids": tokenized_data["input_ids"][batch_indices].to(device),
        "labels": tokenized_data["labels"][batch_indices].to(device),
        "response_mask": tokenized_data["response_mask"][batch_indices].to(device)
    }

# ==========================================
# 核心训练逻辑
# ==========================================

def run_sft_experiment(args):
    # 1. 基础配置
    # 计算梯度累积步数：逻辑 Batch / 物理 Micro Batch
    grad_accum_steps = args.gradient_accumulation_steps
    
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # 加载 Prompt 模版 (用于验证集)
    with open(args.prompt_path, "r") as f:
        r1_template = f.read().strip()

    # 2. 模型与分词器初始化
    print(f"Initializing Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to(args.device)
    # 开启梯度检查点 (显存优化关键)
    policy.gradient_checkpointing_enable()
    
    optimizer = AdamW(policy.parameters(), lr=args.lr)
    
    # 混合精度上下文
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # 初始化 vLLM
    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)

    # ----------------------------------------------------------
    # 3. 数据加载与预处理 (一次性 Tokenize)
    # ----------------------------------------------------------
    print(f"Loading training data from {args.train_data_path}...")
    raw_train_data = []
    with open(args.train_data_path, "r") as f:
        for line in f:
            raw_train_data.append(json.loads(line))
    
    # 过滤与采样
    if args.filter_correct:
        print("Filtering correct examples...")
        raw_train_data = [item for item in raw_train_data if item.get('is_correct', True)]
        print(f"Filtered data size: {len(raw_train_data)}")

    if args.dataset_size and args.dataset_size < len(raw_train_data):
        raw_train_data = random.sample(raw_train_data, args.dataset_size)
        print(f"Sampled subset size: {args.dataset_size}")

    print("Pre-tokenizing entire training dataset...")
    # 优化点：一次性处理所有数据，后续训练只需切片，极大提升速度
    tokenized_train_data = tokenize_prompt_and_output(
        prompt_strs=[item['prompt'] for item in raw_train_data],
        output_strs=[item['response'] for item in raw_train_data],
        tokenizer=tokenizer
    )
    print(f"Tokenization complete. Total samples: {len(tokenized_train_data['input_ids'])}")

    # 加载验证集
    print(f"Loading validation data from {args.val_data_path}...")
    val_prompts = []
    val_ground_truths = []
    with open(args.val_data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= args.max_eval_samples: break
            item = json.loads(line)
            raw_a = item['answer']
            gold = raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            formatted_prompt = r1_template.replace("{question}", item['question'])
            val_prompts.append(formatted_prompt)
            val_ground_truths.append(gold)

    eval_sampling_params = SamplingParams(
        temperature=0.0, # 评估通常用 Greedy
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # ----------------------------------------------------------
    # 4. Step-based 训练主循环
    # ----------------------------------------------------------
    
    progress_bar = tqdm(range(args.max_steps), desc="SFT Steps")

    print(f"\n[Step 0] Starting Evaluation...")
    policy.eval()
    load_policy_into_vllm_instance(policy, vllm_inst)
    
    metrics = log_generations(
        vllm_model=vllm_inst,
        sampling_params=eval_sampling_params,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        reward_fn=r1_zero_reward_fn,
        step=0,
        log_prefix="eval"
    )
    print(f"Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
    policy.train()

    for step in range(args.max_steps):
        
        accumulated_loss = 0.0
        accumulated_entropy = 0.0
        accumulated_res_entropy = 0.0
        
        # --- 梯度累积循环 ---
        for _ in range(grad_accum_steps):
            # 随机采样一个 micro-batch
            batch = get_batch(tokenized_train_data, args.micro_batch_size, args.device)
            
            with amp_ctx: # 混合精度前向传播
                logits = policy(batch["input_ids"]).logits
                
                # 计算 Log-probs (显存优化写法)
                lse = torch.logsumexp(logits, dim=-1)
                target_logits = torch.gather(logits, -1, batch["labels"].unsqueeze(-1)).squeeze(-1)
                log_probs = target_logits - lse

                # 计算熵 (监控用，不参与梯度)
                with torch.no_grad():
                    token_entropy = compute_entropy(logits)
                    valid_token_mask = (batch["labels"] != tokenizer.pad_token_id)
                    current_res_mask = batch["response_mask"].bool() & valid_token_mask
                    
                    avg_res_entropy = token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                    avg_global_entropy = token_entropy[valid_token_mask].mean().item()

                # 计算 Loss 并反向传播
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=batch["response_mask"],
                    gradient_accumulation_steps=grad_accum_steps,
                    normalize_constant=batch["response_mask"].sum().item() # Token-level average
                )
            
            # 累加统计量用于日志
            accumulated_loss += loss.item() * grad_accum_steps # 还原原始 Loss 大小
            accumulated_entropy += avg_global_entropy
            accumulated_res_entropy += avg_res_entropy

        # --- 优化器更新 ---
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        
        # 记录日志
        wandb.log({
            "train/loss": accumulated_loss / grad_accum_steps,
            "train/global_entropy": accumulated_entropy / grad_accum_steps,
            "train/response_entropy": accumulated_res_entropy / grad_accum_steps,
            "train_step": step + 1
        })

        # --- 定期评估 ---
        if (step + 1) % args.eval_every_steps == 0:
            print(f"\n[Step {step + 1}] Starting Evaluation...")
            policy.eval()
            load_policy_into_vllm_instance(policy, vllm_inst)
            
            metrics = log_generations(
                vllm_model=vllm_inst,
                sampling_params=eval_sampling_params,
                prompts=val_prompts,
                ground_truths=val_ground_truths,
                reward_fn=r1_zero_reward_fn,
                step=step + 1,
                log_prefix="eval"
            )
            print(f"Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
            policy.train()

    # 5. 保存模型
    print("Training finished. Saving model...")
    save_name = f"sft_steps{args.max_steps}_subset{args.dataset_size}_filtered{args.filter_correct}"
    output_dir = os.path.join(args.output_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 SFT Step-based Training")
    # 路径
    parser.add_argument("--model_id", type=str, default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train_sft_reason_gsm8k_raw.jsonl")
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k/test.jsonl") 
    parser.add_argument("--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="result/checkpoints")
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16, help="Logical Total Batch Size")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Physical GPU Batch Size")
    parser.add_argument("--max_steps", type=int, default=200, help="Total training steps") # 替换 epochs
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="显式指定累积步数")
    
    # 实验设置
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--filter_correct", action="store_true")
    
    # 硬件与评估
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.5)
    parser.add_argument("--eval_every_steps", type=int, default=20)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    run_sft_experiment(args)