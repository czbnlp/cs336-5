import torch
from typing import List, Dict, Callable
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np
import wandb
from vllm import LLM, SamplingParams


def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer: PreTrainedTokenizer
) -> Dict[str, torch.Tensor]:
    """
    对 prompt 和 response 进行分词，拼接，并生成 response_mask。
    核心逻辑：输入与标签的“偏移”（Shift）
    在因果语言模型（Causal LM）训练中，我们预测的是“下一个词”：
    输入序列：[A, B, C, D]
    预测目标：[B, C, D, E]
    为了实现这种预测，代码要求返回长度为 N-1 的张量：
    input_ids：去掉最后一个 Token（因为它没有后继词可以预测）。
    labels：去掉第一个 Token（因为它是作为起始条件，不是被预测出来的）。
    Response Mask 的作用
    在 SFT 阶段，我们只希望模型对 Response（回答） 部分产生的错误负责。Prompt 是用户给的，模型不需要学习生成 Prompt。通过 response_mask，我们在计算 Loss 时可以把 Prompt 对应的梯度设为 0。

    实现细节与注意事项
      Padding 方向：
      在 SFT 训练中，通常使用右填充（Right Padding）。但在某些推理引擎（如 vLLM）中，生成时必须使用左填充（Left Padding）。作业这里是在做训练准备，所以默认使用右填充。
      EOS Token：
      在 SFT 的 output_strs 中，通常末尾应该包含一个 <|endoftext|> 或 </s>。如果你的数据里没有，你可能需要在代码里手动加上 tokenizer.eos_token。
      Labels 的内容：
      虽然 labels 包含了 Prompt 的词，但因为有 response_mask，在后面算 Loss 的时候：
      loss = (F.cross_entropy(logits, labels) * response_mask).sum() / response_mask.sum()
      这样就能确保只有 Response 部分贡献了梯度。
    """
    all_input_ids = []
    all_labels = []
    all_response_masks = []
    all_lengths = []

    # 1. 分别对每一对进行分词
    for p_str, o_str in zip(prompt_strs, output_strs):
        # 注意：add_special_tokens=False，因为我们手动拼接
        # 有些分词器在开头会自动加 BOS，需根据具体模型调整
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        o_ids = tokenizer.encode(o_str, add_special_tokens=False)
        
        # 拼接完整的 ID 序列
        combined_ids = p_ids + o_ids
        all_input_ids.append(combined_ids)
        
        # 计算长度
        all_lengths.append(len(combined_ids))
        
        # 构造初始 mask：Prompt 部分为 0，Response 部分为 1
        # 注意：这里的 mask 是对应原始拼接后的长度
        mask = [0] * len(p_ids) + [1] * len(o_ids)
        all_response_masks.append(mask)

    # 2. 确定 Batch 的最大长度用于 Padding
    max_len = max(all_lengths)
    batch_size = len(prompt_strs)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # 3. 进行填充 (Padding)
    # 我们先分配空间，初始值为 pad_id
    padded_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, m) in enumerate(zip(all_input_ids, all_response_masks)):
        length = len(ids)
        # 这里使用右填充 (Right Padding)
        padded_input_ids[i, :length] = torch.tensor(ids)
        padded_masks[i, :length] = torch.tensor(m)

    # 4. 执行 Shift 操作 (符合题目要求的 max_len - 1)
    # input_ids: 取前 N-1 个
    # labels: 取后 N-1 个 (偏移一位)
    # response_mask: 对应 labels 的位置，所以也取后 N-1 个
    
    final_input_ids = padded_input_ids[:, :-1]
    final_labels = padded_input_ids[:, 1:].clone()
    final_response_mask = padded_masks[:, 1:]

    # 5. 可选：将 labels 中非 response 部分及 padding 部分设为特殊值（如 -100）
    # 这样在 F.cross_entropy 中可以直接使用 ignore_index=-100
    # 但根据题目要求，我们只需要返回 response_mask
    
    return {
        "input_ids": final_input_ids,
        "labels": final_labels,
        "response_mask": final_response_mask
    }



def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算 next-token 预测的熵（基于词表维度）。
    
    Args:
        logits: torch.Tensor, 形状为 (batch_size, sequence_length, vocab_size)
                包含未归一化的 logits。
                
    Returns:
        torch.Tensor, 形状为 (batch_size, sequence_length)
        每个位置的 next-token 预测熵。
    """
    # 1. 计算 Log-Sum-Exp (LSE)
    # 形状: (batch_size, sequence_length)
    lse = torch.logsumexp(logits, dim=-1)
    
    # 2. 计算概率 p = softmax(logits)
    # 形状: (batch_size, sequence_length, vocab_size)
    probs = F.softmax(logits, dim=-1)
    
    # 3. 计算期望值 E[logits] = sum(p_i * z_i)
    # 形状: (batch_size, sequence_length)
    # 我们对最后一个维度（vocab）求和
    exp_logits = torch.sum(probs * logits, dim=-1)
    
    # 4. 熵 H = logsumexp(logits) - sum(p_i * z_i)
    entropy = lse - exp_logits
    
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    获取模型生成的每个 token 的条件对数概率，并可选地返回熵。
    """
    # 1. 获取模型输出的 logits
    # input_ids shape: (batch_size, seq_len)
    # labels shape: (batch_size, seq_len)
    outputs = model(input_ids)
    logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)

    # 2. 计算所有 token 的 log_softmax
    # shape: (batch_size, seq_len, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)

    # 3. 提取对应 labels 的对数概率
    # 使用 gather 函数从 vocab 维度提取标签对应的 log_prob
    # index 需要与 log_probs_all 维度一致，所以用 unsqueeze(-1) 变成 (B, L, 1)
    # 最后用 squeeze(-1) 变回 (B, L)
    log_probs = torch.gather(
        log_probs_all, 
        dim=-1, 
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # 4. 构建返回字典
    results = {"log_probs": log_probs}

    # 5. 如果需要，计算并添加熵
    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)

    return results


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
    在尊重布尔掩码的情况下，对张量元素求和并按常数归一化。
    
    Args:
        tensor: 需要求和的张量（例如每位置的 log_probs）。
        mask: 与 tensor 形状相同的掩码，1 表示包含，0 表示排除。
        normalize_constant: 归一化常数（除数）。
        dim: 沿着哪个维度求和。如果为 None，则对所有元素求和。
        
    Returns:
        归一化后的求和结果。
    """
    # 1. 将 tensor 与 mask 相乘，排除 mask == 0 的元素
    masked_tensor = tensor * mask
    
    # 2. 根据 dim 参数进行求和
    if dim is None:
        # 对张量中的所有元素求和，返回一个标量
        total_sum = torch.sum(masked_tensor)
    else:
        # 沿着指定的维度求和
        total_sum = torch.sum(masked_tensor, dim=dim)
        
    # 3. 除以归一化常数
    return total_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    执行单次微批次的 SFT 更新。
    """
    # 获取 batch 大小
    batch_size = policy_log_probs.shape[0]

    # 1. 计算每个 token 的负对数似然 (NLL)
    nll_per_token = -policy_log_probs

    # 2. 使用之前实现的 masked_normalize 计算掩码后的总 Loss
    # 注意：此时得到的还只是该 microbatch 内所有有效 token 的加权和
    total_masked_loss = masked_normalize(
        tensor=nll_per_token,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )
    # 3. loss取平均 
    # 标准的 Loss 应该是对 Batch 取平均。
    # 同时为了梯度累积，需要再除以 gradient_accumulation_steps。
    # 最终除数 = batch_size * gradient_accumulation_steps
    microbatch_loss_mean = total_masked_loss / batch_size
    scaled_loss = microbatch_loss_mean / gradient_accumulation_steps

    # 4. 执行反向传播
    scaled_loss.backward()

    # 5. 返回结果
    # 第一个元素返回 scaled_loss 用于测试对比
    # 第二个元素记录未缩放之前的 microbatch 平均 loss 用于日志
    metadata = {
        "loss": microbatch_loss_mean.detach(),
    }

    return scaled_loss, metadata




def log_generations(
    vllm_model: LLM,
    sampling_params: SamplingParams,
    prompts: List[str],
    ground_truths: List[str],
    reward_fn: Callable[[str, str], Dict[str, float]],
    step: int,
    log_prefix: str = "eval"
):
    """
    让模型生成回答并记录详细的评估指标。
    """
    # 1. 模型生成回答
    # 注意：在调用此函数前，应确保已将最新的 policy 权重加载到了 vLLM 实例中
    outputs = vllm_model.generate(prompts, sampling_params)
    
    table_data = []
    
    # 用于统计的数据
    all_lengths = []
    correct_lengths = []
    incorrect_lengths = []
    total_reward = 0
    total_format_reward = 0
    total_answer_reward = 0
    
    # 2. 逐条处理生成结果
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gold_answer = ground_truths[i]
        
        # 计算奖励
        scores = reward_fn(generated_text, gold_answer)
        
        r = scores.get("reward", 0.0)
        fr = scores.get("format_reward", 0.0)
        ar = scores.get("answer_reward", 0.0)
        
        # 计算响应长度
        resp_len = len(generated_text)
        all_lengths.append(resp_len)
        
        if r > 0.5: # 认为是正确的
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)
            
        total_reward += r
        total_format_reward += fr
        total_answer_reward += ar

        # 准备存入 wandb Table 的数据（展示前几条即可，防止日志过大）
        if i < 10: 
            table_data.append([
                step, 
                prompts[i][-100:], # 只取 prompt 结尾部分
                generated_text, 
                gold_answer, 
                r, fr, ar
            ])

    # 3. 计算聚合统计量
    metrics = {
        f"{log_prefix}/accuracy": total_reward / len(prompts),
        f"{log_prefix}/format_score": total_format_reward / len(prompts),
        f"{log_prefix}/answer_score": total_answer_reward / len(prompts),
        f"{log_prefix}/avg_length": np.mean(all_lengths),
        f"{log_prefix}/avg_length_correct": np.mean(correct_lengths) if correct_lengths else 0,
        f"{log_prefix}/avg_length_incorrect": np.mean(incorrect_lengths) if incorrect_lengths else 0,
    }

    # 4. 记录到日志系统
    if wandb.run is not None:
        # 记录表格：方便直接在网页看具体的推理逻辑
        columns = ["step", "prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
        wandb.log({f"{log_prefix}/samples": wandb.Table(columns=columns, data=table_data)}, step=step)
        # 记录标量数值
        wandb.log(metrics, step=step)
    
    print(f"Step {step}: Accuracy: {metrics[f'{log_prefix}/accuracy']:.4f}, Avg Len: {metrics[f'{log_prefix}/avg_length']:.1f}")

    return metrics