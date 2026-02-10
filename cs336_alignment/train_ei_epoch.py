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
# VLLM 初始化与权重同步辅助函数
# ==========================================

def init_vllm(model_id, device, seed, gpu_memory_utilization):
    """初始化 vLLM 实例用于推理评估和数据生成。"""
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
    """将正在训练的 PyTorch 模型权重同步到 vLLM 推理引擎中。"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    print("\n[Sync] Policy weights synced to vLLM.")

# ==========================================
# Expert Iteration 核心逻辑
# ==========================================

def run_expert_iteration(args):
    # 1. 实验配置
    grad_accum_steps = args.batch_size // args.micro_batch_size
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    # 定义 WandB 坐标轴
    wandb.define_metric("train_step")
    wandb.define_metric("ei_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("ei/*", step_metric="ei_step")

    # 加载 Prompt 模版
    print(f"Loading prompt template from {args.prompt_path}...")
    with open(args.prompt_path, "r") as f:
        r1_template = f.read().strip()

    # 2. 数据加载 (作为题目池)
    # EI 不需要预先存在的 response，只需要 question 和 gold
    print(f"Loading question pool from {args.train_data_path}...")
    question_pool = []
    with open(args.train_data_path, "r") as f:
        for line in f:
            item = json.loads(line)
            # 兼容处理：确保我们有 prompt (以 <think> 结尾) 和 gold
            if 'prompt' in item:
                prompt = item['prompt']
            else:
                prompt = r1_template.replace("{question}", item['question'])
            
            # 提取 gold (兼容 R1 格式和原始格式)
            if 'gold' in item:
                gold = item['gold']
            elif 'answer' in item:
                raw_a = item['answer']
                gold = raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            else:
                continue

            question_pool.append({"prompt": prompt, "gold": gold})
    
    print(f"Total questions loaded: {len(question_pool)}")

    # 准备验证集
    print(f"Loading validation data from {args.val_data_path}...")
    val_samples = []
    with open(args.val_data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= args.max_eval_samples: break
            item = json.loads(line)
            raw_a = item['answer']
            gold = raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            val_samples.append({
                "prompt": r1_template.replace("{question}", item['question']),
                "gold": gold
            })

    # 3. 模型初始化
    print(f"Initializing Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to(args.device)
    # 开启梯度检查点支持长文本
    policy.gradient_checkpointing_enable()
    
    optimizer = AdamW(policy.parameters(), lr=args.lr)

    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)

    # 4. Expert Iteration Loop
    global_train_step = 0
    
    # 采样参数 (用于生成rollouts)
    rollout_params = SamplingParams(
        n=args.rollouts,  # 核心：每个问题生成 G 个回答
        temperature=1.0,  # 必须有温度以保证多样性
        max_tokens=args.max_tokens,
        min_tokens=4,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 评估参数 (Greedy)
    eval_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    for ei_step in range(args.n_ei_steps):
        print(f"\n{'='*20} EI Step {ei_step + 1}/{args.n_ei_steps} {'='*20}")
        
        # --- A. 采样与过滤 (Rollout & Filter) ---
        print(">> Generating rollouts...")
        # 1. 同步权重
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)
        
        # 2. 抽取 Batch (Db)
        batch_db = random.sample(question_pool, min(args.ei_batch_size, len(question_pool)))
        prompts = [q['prompt'] for q in batch_db]
        golds = [q['gold'] for q in batch_db]
        
        # 3. vLLM 生成
        outputs = vllm_inst.generate(prompts, rollout_params)
        
        # 4. 过滤 (Verify)
        expert_dataset = []
        total_generated = len(batch_db) * args.rollouts
        
        for i, output in enumerate(outputs):
            current_gold = golds[i]
            # 遍历该问题的 G 个生成结果
            for candidate in output.outputs:
                resp_text = candidate.text
                # 使用 reward function 验证
                # 注意：r1_zero_reward_fn 检查格式和答案
                score = r1_zero_reward_fn(resp_text, current_gold)
                
                if score['reward'] == 1.0:
                    expert_dataset.append({
                        "prompt": prompts[i],
                        "response": resp_text
                    })
        
        success_rate = len(expert_dataset) / total_generated
        print(f">> Filtering complete.")
        print(f"   Generated: {total_generated} | Kept (Expert): {len(expert_dataset)}")
        print(f"   Success Rate: {success_rate:.2%}")
        
        wandb.log({
            "ei/step": ei_step + 1,
            "ei/dataset_size": len(expert_dataset),
            "ei/success_rate": success_rate
        }, step=global_train_step) # 这里的 step 对齐到总训练步

        # --- B. 训练 (Inner Loop SFT) ---
        print(f">> Training on expert data ({len(expert_dataset)} samples, {args.sft_epochs} epochs)...")
        policy.train()
        
        # 构造训练所需的 indices
        train_indices = list(range(len(expert_dataset)))
        
        for epoch in range(args.sft_epochs):
            random.shuffle(train_indices)
            
            # Micro-batch 循环
            for i in range(0, len(train_indices), args.micro_batch_size):
                batch_idxs = train_indices[i : i + args.micro_batch_size]
                batch = [expert_dataset[idx] for idx in batch_idxs]
                
                # Tokenize
                inputs = tokenize_prompt_and_output(
                    [item['prompt'] for item in batch],
                    [item['response'] for item in batch],
                    tokenizer
                )
                input_ids = inputs['input_ids'].to(args.device)
                labels = inputs['labels'].to(args.device)
                mask = inputs['response_mask'].to(args.device)

                # Forward
                logits = policy(input_ids).logits
                
                # 显存优化 Log-Prob
                lse = torch.logsumexp(logits, dim=-1)
                target_logits = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)
                log_probs = target_logits - lse

                # 计算 Entropy (监控用)
                with torch.no_grad():
                    token_entropy = compute_entropy(logits)
                    valid_token_mask = (labels != tokenizer.pad_token_id)
                    current_res_mask = mask.bool() & valid_token_mask
                    current_prompt_mask = (~mask.bool()) & valid_token_mask
                    
                    avg_res_ent = token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                    avg_prm_ent = token_entropy[current_prompt_mask].mean().item() if current_prompt_mask.any() else 0.0

                # Loss & Backward
                loss, _ = sft_microbatch_train_step(
                    policy_log_probs=log_probs,
                    response_mask=mask,
                    gradient_accumulation_steps=grad_accum_steps,
                    normalize_constant=1.0
                )

                # Optimizer Step
                if (i // args.micro_batch_size + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_train_step += 1
                    
                    wandb.log({
                        "train/loss": loss.item() * grad_accum_steps,
                        "train/response_entropy": avg_res_ent,
                        "train/prompt_entropy": avg_prm_ent,
                        "train_step": global_train_step
                    })

        # --- C. 每轮迭代结束后的评估 ---
        print(f">> Evaluating after EI Step {ei_step + 1}...")
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)
        
        metrics = log_generations(
            vllm_model=vllm_inst,
            sampling_params=eval_params,
            prompts=[ex['prompt'] for ex in val_samples],
            ground_truths=[ex['gold'] for ex in val_samples],
            reward_fn=r1_zero_reward_fn,
            step=global_train_step,
            log_prefix="eval"
        )
        print(f"   Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")

    # 5. 保存最终模型
    print("EI Finished. Saving model...")
    save_name = f"ei_G{args.rollouts}_Db{args.ei_batch_size}_E{args.sft_epochs}"
    output_dir = os.path.join(args.output_dir, save_name)
    os.makedirs(output_dir, exist_ok=True)
    
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS336 Expert Iteration Script")

    # 路径配置
    parser.add_argument("--model_id", type=str, default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train_sft_reason_gsm8k_r1.jsonl")
    parser.add_argument("--val_data_path", type=str, default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="result/ei_checkpoints")

    # 基础训练参数
    parser.add_argument("--lr", type=float, default=1e-5) # EI通常用较小学习率
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024)

    # EI 专属参数 (消融实验核心)
    parser.add_argument("--n_ei_steps", type=int, default=5, help="EI迭代总次数")
    parser.add_argument("--ei_batch_size", type=int, default=1024, help="每次迭代采样的问题数量 (Db)")
    parser.add_argument("--rollouts", type=int, default=8, help="每个问题生成的采样数 (G)")
    parser.add_argument("--sft_epochs", type=int, default=1, help="每次迭代内部SFT的轮数")

    # 硬件与评估
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.2)
    parser.add_argument("--max_eval_samples", type=int, default=100)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-ei")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    run_expert_iteration(args)