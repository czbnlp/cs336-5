import torch
import json
import random
import wandb
import os
import argparse
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
    compute_entropy # 确保导入了计算熵的函数
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

# ==========================================
# VLLM 初始化与权重同步辅助函数
# ==========================================

def init_vllm(model_id, device, seed, gpu_memory_utilization):
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
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    print("\n[Sync] 已将最新的训练权重同步至 vLLM 实例。")

# ==========================================
# 核心训练逻辑
# ==========================================

def run_sft_experiment(args):
    # 1. 实验配置
    grad_accum_steps = args.batch_size // args.micro_batch_size
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # 定义 WandB 指标坐标轴
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # 加载 Prompt 模版
    with open(args.prompt_path, "r") as f:
        r1_template = f.read().strip()

    # 2. 数据加载
    print(f"Loading training data from {args.train_data_path}...")
    with open(args.train_data_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    if args.filter_correct:
        print("Filtering correct examples using is_correct flag...")
        data = [item for item in data if item.get('is_correct', True)]
        print(f"Filtered data size: {len(data)}")

    if args.dataset_size and args.dataset_size < len(data):
        data = random.sample(data, args.dataset_size)
        print(f"Sampled subset size: {args.dataset_size}")

    # 动态处理验证集
    print(f"Loading and formatting validation data from {args.val_data_path}...")
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
        temperature=1.0, 
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 3. 模型初始化
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to(args.device)
    policy.gradient_checkpointing_enable()
    
    optimizer = AdamW(policy.parameters(), lr=args.lr)

    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)

    # 4. 训练主循环
    policy.train()
    global_step = 0 
    total_steps = (len(data) // args.batch_size) * args.epochs
    progress_bar = tqdm(total=total_steps, desc="SFT Training")

    for epoch in range(args.epochs):
        random.shuffle(data)
        
        for i in range(0, len(data), args.micro_batch_size):
            if i + args.micro_batch_size > len(data): break
            
            batch = data[i : i + args.micro_batch_size]
            
            # --- A. 数据编码 ---
            inputs = tokenize_prompt_and_output(
                [item['prompt'] for item in batch],
                [item['response'] for item in batch],
                tokenizer
            )
            input_ids = inputs['input_ids'].to(args.device)
            labels = inputs['labels'].to(args.device)
            mask = inputs['response_mask'].to(args.device)

            # --- B. 前向传播 ---
            logits = policy(input_ids).logits
            
            # 显存优化计算 Log-Prob
            lse = torch.logsumexp(logits, dim=-1)
            target_logits = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)
            log_probs = target_logits - lse

            # --- 计算熵相关信息 ---
            with torch.no_grad():
                # 计算每个位置的熵
                token_entropy = compute_entropy(logits) # (B, L)
                
                # 排除 Padding 位置的 Mask (labels 为 pad_id 的位置不计入统计)
                valid_token_mask = (labels != tokenizer.pad_token_id)
                
                # 构造 Prompt 和 Response 的子掩码
                current_res_mask = mask.bool() & valid_token_mask
                current_prompt_mask = (~mask.bool()) & valid_token_mask

                # 计算均值
                avg_global_entropy = token_entropy[valid_token_mask].mean().item()
                # 使用 clamp 防止空 mask 导致 NaN
                avg_res_entropy = token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                avg_prompt_entropy = token_entropy[current_prompt_mask].mean().item() if current_prompt_mask.any() else 0.0

            # --- C. 计算 Loss 并反向传播 ---
            loss, meta = sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=mask,
                gradient_accumulation_steps=grad_accum_steps,
                normalize_constant=mask.sum().item()
            )

            # --- D. 优化器更新 ---
            if (i // args.micro_batch_size + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                progress_bar.update(1)
                
                # 记录详细日志
                wandb.log({
                    "train/loss": loss.item() * grad_accum_steps, 
                    "train/global_entropy": avg_global_entropy,
                    "train/response_entropy": avg_res_entropy,
                    "train/prompt_entropy": avg_prompt_entropy,
                    "train_step": global_step
                })

                # --- E. 定期评估 ---
                if global_step % args.eval_every_steps == 0:
                    print(f"\n[Step {global_step}] 执行权重同步与评估...")
                    policy.eval()
                    load_policy_into_vllm_instance(policy, vllm_inst)
                    
                    metrics = log_generations(
                        vllm_model=vllm_inst,
                        sampling_params=eval_sampling_params,
                        prompts=val_prompts,
                        ground_truths=val_ground_truths,
                        reward_fn=r1_zero_reward_fn,
                        step=global_step,
                        log_prefix="eval"
                    )
                    
                    print(f"[Step {global_step}] Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
                    policy.train()

    # 5. 保存模型
    print("Training finished. Saving model...")
    save_name = f"sft_subset{args.dataset_size}_filtered{args.filter_correct}"
    output_dir = os.path.join(args.output_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 SFT Experiment Script")
    # 路径配置
    parser.add_argument("--model_id", type=str, default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train_sft_reason_r1_final.jsonl")
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k/test.jsonl") 
    parser.add_argument("--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="result/checkpoints")
    # 训练参数
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)
    # 实验设置
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--filter_correct", action="store_true")
    # 硬件与评估
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.5)
    parser.add_argument("--eval_every_steps", type=int, default=20)
    parser.add_argument("--max_eval_samples", type=int, default=2000)
    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    run_sft_experiment(args)