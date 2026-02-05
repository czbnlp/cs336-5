import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import wandb
import argparse
from tqdm import tqdm
import math

from cs336_alignment.sft_dataset import InstructionDataset

def main():
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument("--train_path", type=str, required=True, help="Path to training jsonl")
    parser.add_argument("--eval_path", type=str, default=None, help="Path to validation jsonl")
    # 模型路径
    parser.add_argument("--model_path", type=str, default="/data/a5-alignment/models/Llama-3.1-8B")
    parser.add_argument("--output_dir", type=str, default="out/sft_model")
    # 超参数
    parser.add_argument("--batch_size", type=int, default=32, help="Effective batch size")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Batch size per forward pass")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=100, help="Eval every X steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    
    args = parser.parse_args()

    # --- 1. 初始化 WandB ---
    run_name = f"lr{args.lr}-bs{args.batch_size}"
    wandb.init(
        project="cs336-sft-ultraChat-SafetyLlama-qwen_7b", 
        name=run_name,
        config=vars(args)
    )

    # --- 2. 加载分词器和模型 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    # 二者不相等，https://github.com/QwenLM/Qwen/issues/419
    print(model.config.vocab_size) # 152064
    print(len(tokenizer)) # 151665
   

    # 开启梯度检查点以节省显存，这在 8B 模型全参数微调时非常有用
    model.gradient_checkpointing_enable()

    # --- 3. 加载数据集 ---
    train_ds = InstructionDataset(
        tokenizer=tokenizer,
        dataset_path=args.train_path,
        seq_length=args.max_seq_len,
        shuffle=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size, shuffle=True)

    eval_loader = None
    if args.eval_path:
        eval_ds = InstructionDataset(
            tokenizer=tokenizer,
            dataset_path=args.eval_path,
            seq_length=args.max_seq_len,
            shuffle=False
        )
        eval_loader = DataLoader(eval_ds, batch_size=args.micro_batch_size, shuffle=False)

    # --- 4. 优化器与调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    grad_accum_steps = args.batch_size // args.micro_batch_size
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    # --- 5. 训练循环 ---
    print(f"Starting training: {total_steps} total update steps.")
    progress_bar = tqdm(total=total_steps, desc="SFT Training")
    
    model.train()
    optimizer.zero_grad()
    
    accumulated_loss = 0
    global_step = 0

    for epoch in range(args.epochs):
        for idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)

            # 前向传播
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            accumulated_loss += loss.item()

            # 梯度累积达到步数，执行更新
            if (idx + 1) % grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 记录指标到 WandB
                metrics = {
                    "train/loss": accumulated_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + (idx / len(train_loader))
                }

                # 周期性验证
                if eval_loader and global_step % args.eval_steps == 0:
                    val_loss = run_evaluation(model, eval_loader)
                    metrics["eval/loss"] = val_loss
                    print(f"\nStep {global_step}: Eval Loss = {val_loss:.4f}")
                    model.train()

                wandb.log(metrics)
                progress_bar.set_postfix({"loss": f"{accumulated_loss:.4f}"})
                progress_bar.update(1)
                
                accumulated_loss = 0

    # --- 6. 保存最终模型 ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Training finished. Model saved to {args.output_dir}")
    wandb.finish()

def run_evaluation(model, eval_loader):
    """简单的验证循环"""
    model.eval()
    total_eval_loss = 0
    # 为了速度，只在验证集上跑 50 个 batch 取平均
    max_eval_batches = 50 
    
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            if i >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(model.device)
            labels = batch["labels"].to(model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            total_eval_loss += outputs.loss.item()
            
    return total_eval_loss / min(len(eval_loader), max_eval_batches)

if __name__ == "__main__":
    main()