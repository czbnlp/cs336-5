import torch
import json
import random
import wandb
import os
import argparse
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from unittest.mock import patch

# --- 导入自定义工具函数 ---
from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output, 
    get_response_log_probs,
)
from cs336_alignment.grpo_utils import (
    compute_group_normalized_rewards,
    grpo_microbatch_train_step,
    log_generations
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn

# ==========================================
# 辅助函数
# ==========================================


def load_math12k_dataset(path, prompt_template=None):
    df = pd.read_parquet(path)
    processed_items = []
    for _, row in df.iterrows():
        q_text = row['problem']
        gold_answer = row['answer']
        
        if gold_answer:
            processed_items.append({
                "prompt": prompt_template.replace("{question}", q_text),
                "gold": gold_answer
            })
    return processed_items

def load_gsm8k_dataset(path, prompt_template=None):
    
    processed_items = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            q_text = item['question']
            full_sol = item['answer']
            gold_answer = full_sol.split("####")[-1].strip() if "####" in full_sol else full_sol.strip()
            processed_items.append({
                "prompt": prompt_template.replace("{question}", q_text),
                "gold": gold_answer
            })
    return processed_items


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


# ==========================================
# GRPO 核心训练逻辑
# ==========================================

def run_grpo_training(args):
    # 1. 实验配置与初始化
    assert args.train_batch_size % args.gradient_accumulation_steps == 0
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    
    assert args.rollout_batch_size % args.group_size == 0
    
    n_prompts_per_rollout = args.rollout_batch_size // args.group_size
    
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    
    if args.prompt_style == "question_only":
        active_reward_fn = question_only_reward_fn
        print("Using [Question-Only] reward function.")
    else:
        active_reward_fn = r1_zero_reward_fn
        print("Using [R1-Zero] reward function.")

    print(f"Loading prompt template from {args.prompt_path}...")
    with open(args.prompt_path, "r") as f:
        prompt_template = f.read().strip()

    if 'math12k' in args.train_data_path.lower():
        questions_pool = load_math12k_dataset(args.train_data_path, prompt_template)
        val_samples = load_math12k_dataset(args.test_data_path, prompt_template)[:args.max_eval_samples]
    elif 'gsm8k' in args.train_data_path.lower():
        questions_pool = load_gsm8k_dataset(args.train_data_path, prompt_template)
        val_samples = load_gsm8k_dataset(args.test_data_path, prompt_template)[:args.max_eval_samples]
    else:
        raise ValueError("Unsupported dataset. Please use Math12K or GSM8K.")
    print(f"Total training questions: {len(questions_pool)}")

    # 定义评估用的采样参数 
    eval_sampling_params = SamplingParams(
        temperature=1.0, # 作业要求 1.0
        max_tokens=args.sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # 3. 初始化模型
    print(f"Initializing Policy Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    ).to(args.device)
    # 开启梯度检查点 (关键)
    policy.gradient_checkpointing_enable()
    
    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)

    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst = init_vllm(args.model_id, args.vllm_device, args.seed, args.vllm_gpu_util)


    # ==========================================
    # Step 0 初始评估 (训练前的基准)
    # ==========================================
    print(f"\n[GRPO Step 0] Starting Initial Evaluation (Baseline)...")
    policy.eval() 
    # 第一次同步初始权重
    load_policy_into_vllm_instance(policy, vllm_inst)
    
    metrics = log_generations(
        vllm_model=vllm_inst,
        sampling_params=eval_sampling_params,
        prompts=[s['prompt'] for s in val_samples],
        ground_truths=[s['gold'] for s in val_samples],
        reward_fn=active_reward_fn ,
        step=0, # 显式记为第 0 步
        log_prefix="eval"
    )
    print(f"[GRPO Step 0] Initial Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
    policy.train() 

    # ==========================================
    # 开始 GRPO 主循环
    # ==========================================
    # 4. GRPO 主循环
    global_step = 0
    # 训练采样参数
    rollout_sampling_params = SamplingParams(
        n=args.group_size,
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    progress_bar = tqdm(range(args.n_grpo_steps), desc="GRPO Steps")

    for step in range(args.n_grpo_steps):
        # ==========================================
        # Phase 1: 采样 (Rollout)
        # ==========================================
        policy.eval()
        load_policy_into_vllm_instance(policy, vllm_inst)
        
        current_batch_questions = random.sample(questions_pool, n_prompts_per_rollout)
        prompts = [q['prompt'] for q in current_batch_questions]
        golds = [q['gold'] for q in current_batch_questions]
        
        # vLLM 生成
        outputs = vllm_inst.generate(prompts, rollout_sampling_params)
        
        flat_prompts = []
        flat_responses = []
        flat_golds = []
        
        for i, output in enumerate(outputs):
            for candidate in output.outputs:
                flat_prompts.append(prompts[i])
                flat_responses.append(candidate.text)
                flat_golds.append(golds[i])
        
        # 计算 Advantage
        advantages, raw_rewards, reward_meta = compute_group_normalized_rewards(
            reward_fn=active_reward_fn ,
            rollout_responses=flat_responses,
            repeated_ground_truths=flat_golds,
            group_size=args.group_size,
            advantage_eps=args.advantage_eps,
            normalize_by_std=args.use_std_normalization
        )
        
        wandb.log({
            "rollout/mean_reward": reward_meta['mean_reward'],
            "rollout/max_reward": reward_meta['max_reward'],
            "rollout_step": step + 1 # 显式 X 轴
        }, step=step + 1)

        # ==========================================
        # Phase 2: 准备训练数据
        # ==========================================
        all_inputs = tokenize_prompt_and_output(
            flat_prompts, flat_responses, tokenizer
        )
        
        # 截断处理
        max_len = args.max_len
        input_ids_tensor = all_inputs['input_ids']
        if input_ids_tensor.size(1) > max_len:
            all_input_ids = input_ids_tensor[:, :max_len].to(args.device)
            all_labels = all_inputs['labels'][:, :max_len].to(args.device)
            all_masks = all_inputs['response_mask'][:, :max_len].to(args.device)
            
        else:
            all_input_ids = input_ids_tensor.to(args.device)
            all_labels = all_inputs['labels'].to(args.device)
            all_masks = all_inputs['response_mask'].to(args.device)
            
        advantages = advantages.to(args.device)
        raw_rewards = raw_rewards.to(args.device)
        policy.eval()
        # 计算 Old Log Probs (用于 Clip)
        with torch.no_grad():
            old_log_probs_list = []
            for i in range(0, len(all_input_ids), micro_train_batch_size):
                batch_ids = all_input_ids[i : i + micro_train_batch_size]
                batch_lbls = all_labels[i : i + micro_train_batch_size]
                res = get_response_log_probs(policy, batch_ids, batch_lbls)
                old_log_probs_list.append(res['log_probs'])
            old_log_probs = torch.cat(old_log_probs_list, dim=0)

        # ==========================================
        # Phase 3: 训练 (Training Loop)
        # ==========================================
        dataset_indices = list(range(len(flat_responses)))
        policy.train()
        for epoch in range(args.epochs_per_rollout_batch):
            random.shuffle(dataset_indices)
            
            for i in range(0, len(dataset_indices), micro_train_batch_size):
                batch_idxs = dataset_indices[i : i + micro_train_batch_size]
                
                batch_input_ids = all_input_ids[batch_idxs]
                batch_labels = all_labels[batch_idxs]
                batch_masks = all_masks[batch_idxs]
                batch_advantages = advantages[batch_idxs]
                batch_old_log_probs = old_log_probs[batch_idxs]
                batch_raw_rewards = raw_rewards[batch_idxs]
                
                # Forward
                current_res = get_response_log_probs(
                    policy, batch_input_ids, batch_labels, return_token_entropy=True
                )
                
                # Loss & Backward
                loss, loss_meta = grpo_microbatch_train_step(
                    policy_log_probs=current_res['log_probs'],
                    response_mask=batch_masks,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    loss_type=args.loss_type,
                    raw_rewards=batch_raw_rewards.unsqueeze(1),
                    advantages=batch_advantages.unsqueeze(1),
                    old_log_probs=batch_old_log_probs,
                    cliprange=0.2,
                    length_norm_type=args.length_norm_type
                )
                
                # Logging
                if (global_step + 1) % 10 == 0:
                    wandb.log({
                        "train/loss": loss_meta['loss'],
                        "train/clip_fraction": loss_meta.get('clip_fraction', 0),
                        "train/ratio_mean": loss_meta.get('ratio_mean', 0),
                        "train/entropy": current_res['token_entropy'].mean().item(),
                        "train_step": global_step # 用于内部监控的 X 轴
                    }, step=step + 1) # 强制与当前 GRPO 步数对齐

            # 更新参数 (在一个 epoch 结束后，或者积累了足够步数后)
            # 这里的逻辑是：一次 Rollout 的数据正好填满 gradient_accumulation_steps
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        progress_bar.update(1)

        # ==========================================
        # Phase 4: 评估与保存 (Moved Here!)
        # ==========================================
        
        # 1. 评估逻辑 (每隔 N 个 GRPO Step)
        if (step + 1) % args.eval_every_steps == 0:
            print(f"\n[GRPO Step {step + 1}] Starting Evaluation...")
            policy.eval() 
            load_policy_into_vllm_instance(policy, vllm_inst)
            
            # 调用更新后的 log_generations
            # 传入 step+1 作为统一的时间戳
            metrics = log_generations(
                vllm_model=vllm_inst,
                sampling_params=eval_sampling_params,
                prompts=[s['prompt'] for s in val_samples],
                ground_truths=[s['gold'] for s in val_samples],
                reward_fn=active_reward_fn ,
                step=step + 1,
                log_prefix="eval"
            )
            print(f"[GRPO Step {step + 1}] Eval Accuracy: {metrics.get('eval/accuracy', 0):.2%}")
            policy.train()

        # 2. 保存逻辑
        if (step + 1) % args.save_every_steps == 0:
            print(f"Saving checkpoint at step {step + 1}...")
            save_path = os.path.join(args.output_dir, f"grpo_step{step+1}")
            os.makedirs(save_path, exist_ok=True)
            policy.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    print("GRPO Training Finished.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 路径
    parser.add_argument("--model_id", type=str, default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/gsm8k/train.jsonl")
    parser.add_argument("--test_data_path", type=str, default="data/gsm8k/test.jsonl")
    
    parser.add_argument("--prompt_path", type=str, default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir", type=str, default="result/grpo_checkpoints")
    
    # 核心超参数
    parser.add_argument("--n_grpo_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    
    
    
    # 采样参数
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=2048)
    
    # GRPO 特定
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--use_std_normalization", action="store_true", help="Enable std normalization")
    parser.add_argument("--loss_type", type=str, default="grpo_clip")
    parser.add_argument("--length_norm_type", type=str, default="mask_mean")
    
    # 硬件与评估
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vllm_device", type=str, default="cuda:1")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_every_steps", type=int, default=8)
    parser.add_argument("--save_every_steps", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=2000)

    # Prompt与奖励函数
    parser.add_argument("--prompt_style", type=str, default="r1_zero", choices=["r1_zero", "question_only"], 
                    help="选择提示词风格和对应的奖励函数")
    
    # WandB
    parser.add_argument("--wandb_project", type=str, default="cs336-grpo")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    run_grpo_training(args)