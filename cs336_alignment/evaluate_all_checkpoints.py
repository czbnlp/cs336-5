import os
import glob
import json
import pandas as pd
import torch
import gc
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

# ==========================================
# 1. 导入你的自定义模块
# ==========================================
sys.path.append(os.getcwd()) 
try:
    # 假设你的 r1_zero_reward_fn 定义在 cs336_alignment.drgrpo_grader
    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
except ImportError:
    print("❌ Error: 找不到 cs336_alignment 模块，请确保在项目根目录运行。")
    exit(1)

# ================= 配置区 =================
CHECKPOINT_ROOT = "result/grpo_lr_sweep" 
TEST_DATA_PATH = "data/gsm8k/test.jsonl"
NUM_EVAL_SAMPLES = 100
OUTPUT_CSV = "grpo_lr_sweep_metrics.csv"
OUTPUT_PLOT = "grpo_lr_sweep_plot.png"


PROMPT_TEMPLATE = """
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. The final result only requires a direct output of the answer without any explanations.
User: {question}
Assistant: <think>
"""
# ================= 数据加载与工具 =================

def load_gsm8k_test_data(path, limit=None):
    print(f"Loading data from {path}...")
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            item = json.loads(line)
            # 兼容处理 answer/gold 字段
            raw_ans = item.get('answer', item.get('gold', ''))
            gold = raw_ans.split("####")[-1].strip()
            prompt = PROMPT_TEMPLATE.replace("{question}", item['question'])
            data.append({"prompt": prompt, "gold": gold})
    return data

def find_checkpoints(root_dir):
    checkpoints = []
    lr_dirs = glob.glob(os.path.join(root_dir, "grpo_lr*"))
    for lr_dir in lr_dirs:
        lr_name = os.path.basename(lr_dir)
        try:
            # 提取 float 类型的 lr
            lr_str = lr_name.replace("grpo_lr", "")
            lr_val = float(lr_str)
        except:
            continue # 跳过无法解析的文件夹

        step_dirs = glob.glob(os.path.join(lr_dir, "grpo_step*"))
        for step_dir in step_dirs:
            step_name = os.path.basename(step_dir)
            try:
                step_val = int(step_name.replace("grpo_step", ""))
            except:
                continue
            
            checkpoints.append({
                "path": step_dir,
                "lr": lr_val,         #用于排序
                "lr_str": lr_str,     #用于显示和分组
                "step": step_val,
                "run_name": lr_name
            })
    
    # 按 LR 大小和 Step 排序
    checkpoints.sort(key=lambda x: (x['lr'], x['step']))
    return checkpoints

# ================= 评测核心逻辑 =================

def evaluate_single_checkpoint(model_path, dataset):
    print(f"Loading model: {model_path} ...")
    torch.cuda.empty_cache()
    gc.collect()

    try:
        # 初始化 vLLM
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.2, 
            trust_remote_code=True,
            dtype="bfloat16"
        )
        
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1024,
            stop=["</answer>"],
            include_stop_str_in_output=True
        )
        
        prompts = [d['prompt'] for d in dataset]
        golds = [d['gold'] for d in dataset]
        
        # 生成
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.outputs[0].text for output in outputs]
        
        # --- 统计指标 ---
        total_reward = 0.0
        total_format_reward = 0.0
        total_answer_reward = 0.0
        count = len(generated_texts)

        # 遍历每一条结果进行打分
        for response, gold in zip(generated_texts, golds):
            # 调用你的 reward function (处理单条数据)
            # 注意：如果你的函数需要 fast=True 参数，请在这里添加
            metrics = r1_zero_reward_fn(response, gold)
            
            total_reward += metrics.get("reward", 0.0)
            total_format_reward += metrics.get("format_reward", 0.0)
            total_answer_reward += metrics.get("answer_reward", 0.0)
            
        # 计算平均值
        avg_metrics = {
            "reward": total_reward / count,               # 最终得分 (通常等同于 Accuracy)
            "format_reward": total_format_reward / count, # 格式遵循率
            "answer_reward": total_answer_reward / count  # 答案正确率 (在格式正确的前提下)
        }
        
        # 销毁模型
        destroy_model_parallel()
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        
        return avg_metrics

    except Exception as e:
        print(f"❌ Error evaluating {model_path}: {e}")
        torch.cuda.empty_cache()
        gc.collect()
        return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}

# ================= 绘图函数 =================

def plot_results(csv_path, output_img_path):
    print(f"Plotting results to {output_img_path}...")
    df = pd.read_csv(csv_path)
    
    # 确保 lr_str 是字符串类型，方便分类绘图
    df['lr_str'] = df['lr_str'].astype(str)
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 创建 1行2列 的图表：左边画总分(Accuracy)，右边画格式遵循率(Format)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 图 1: Average Reward (Accuracy) ---
    sns.lineplot(
        data=df, 
        x="step", 
        y="reward", 
        hue="lr_str", 
        style="lr_str",
        markers=True, 
        dashes=False,
        ax=axes[0],
        palette="tab10"
    )
    axes[0].set_title("Average Reward (Accuracy) vs Step")
    axes[0].set_ylabel("Reward")
    axes[0].set_xlabel("Training Step")
    
    # --- 图 2: Format Reward (Compliance) ---
    sns.lineplot(
        data=df, 
        x="step", 
        y="format_reward", 
        hue="lr_str", 
        style="lr_str",
        markers=True, 
        dashes=False,
        ax=axes[1],
        palette="tab10"
    )
    axes[1].set_title("Format Compliance Rate vs Step")
    axes[1].set_ylabel("Format Reward")
    axes[1].set_xlabel("Training Step")
    
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300)
    print("✅ Plot saved!")

# ================= 主流程 =================

def main():
    # 1. 加载数据
    dataset = load_gsm8k_test_data(TEST_DATA_PATH, limit=NUM_EVAL_SAMPLES)
    if not dataset: return

    # 2. 扫描 Checkpoints
    checkpoints = find_checkpoints(CHECKPOINT_ROOT)
    print(f"Found {len(checkpoints)} checkpoints.")
    
    results = []
    
    # 3. 循环评测
    for ckpt in tqdm(checkpoints, desc="Eval Loop"):
        print(f"\n>>> Evaluating LR={ckpt['lr_str']} Step={ckpt['step']}")
        
        metrics = evaluate_single_checkpoint(ckpt['path'], dataset)
        
        print(f"    Results: Reward={metrics['reward']:.2f}, Format={metrics['format_reward']:.2f}")
        
        record = {
            "lr": ckpt['lr'],
            "lr_str": ckpt['lr_str'],
            "step": ckpt['step'],
            "path": ckpt['path'],
            # 展开字典中的三个指标
            "reward": metrics['reward'],
            "format_reward": metrics['format_reward'],
            "answer_reward": metrics['answer_reward']
        }
        results.append(record)
        
        # 实时保存
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Evaluation finished. Data saved to {OUTPUT_CSV}")
    
    # 4. 画图
    if len(results) > 0:
        plot_results(OUTPUT_CSV, OUTPUT_PLOT)

if __name__ == "__main__":
    main()