import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import wandb # 用于记录日志
import argparse

def train():
    # 1. 设置超参数 (参考 PDF 第 12 页)
    lr = 2e-5
    num_epochs = 1
    context_length = 512
    micro_batch_size = 2      # 单次前向占用的显存
    target_batch_size = 32     # 有效批次大小
    grad_accum_steps = target_batch_size // micro_batch_size
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 加载模型和分词器
    model_name = "/data/a5-alignment/models/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    ).to(device)

    # 3. 准备数据 (使用之前实现的 Dataset)
    from cs336_alignment.sft_dataset import InstructionDataset
    train_dataset = InstructionDataset(tokenizer, "path/to/train.jsonl.gz", context_length, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size)

    # 4. 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * num_epochs // grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.03 * total_steps), 
        num_training_steps=total_steps
    )

    # 初始化 WandB
    wandb.init(project="cs336-sft", config={"lr": lr, "batch_size": target_batch_size})

    # 5. 训练循环
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(num_epochs):
        for idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # 前向传播
            outputs = model(input_ids)
            # 计算 Loss，并根据累积步数进行缩放
            loss = F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            loss = loss / grad_accum_steps
            
            # 反向传播
            loss.backward()

            # 达到累积步数，更新权重
            if (idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 记录日志
                wandb.log({"train_loss": loss.item() * grad_accum_steps, "lr": scheduler.get_last_lr()[0]})
                print(f"Step {idx//grad_accum_steps}: Loss {loss.item()*grad_accum_steps:.4f}")

    # 6. 保存模型 (Deliverable)
    model.save_pretrained("output/sft_model")
    tokenizer.save_pretrained("output/sft_model")
    wandb.finish()

if __name__ == "__main__":
    train()