import torch
from typing import Callable, List, Dict, Tuple, Literal, Optional
import torch
from typing import List, Dict, Callable
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from transformers import PreTrainedModel
import numpy as np
import wandb
from vllm import LLM, SamplingParams


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], Dict[str, float]],
    rollout_responses: List[str],
    repeated_ground_truths: List[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    计算 GRPO (Group Relative Policy Optimization) 的核心组件：组内归一化奖励（即优势 Advantage）。

    该函数执行以下步骤：
    1. 计算每个回答的原始奖励分数。
    2. 将回答按照问题分组（每组 group_size 个）。
    3. 计算组内的平均分（和标准差）。
    4. 将每个回答的分数减去组均值（并可选地除以标准差），得到相对优势。

    Args:
        reward_fn (Callable): 奖励函数。
            - 输入: (模型生成的回答 response, 标准答案 ground_truth)
            - 输出: 一个字典，必须包含 key "reward" (float)。
        
        rollout_responses (List[str]): 模型生成的所有回答列表。
            - 长度 = (问题数量 * group_size)。
            - 顺序隐含了分组逻辑，即前 group_size 个对应第1个问题，接下来的 group_size 个对应第2个问题...
        
        repeated_ground_truths (List[str]): 对应的标准答案列表。
            - 长度必须与 rollout_responses 相同。
            - 对于同一个问题的 group_size 个回答，这里的 ground_truth 是重复的。
        
        group_size (int): 每个问题生成的回答数量 (G)。
            - 用于将扁平的列表 reshape 成 (N, G) 的矩阵。
        
        advantage_eps (float): 防止除零错误的极小值。
            - 当 normalize_by_std=True 时，分母为 (std + eps)。默认 1e-8。
        
        normalize_by_std (bool): 是否除以组内标准差。
            - True: 使用公式 (r - mean) / std (标准 GRPO/DeepSeekMath)。
            - False: 仅使用公式 r - mean (Dr. GRPO 建议的简化版，防止 std 接近 0 时数值不稳定)。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
            1. advantages: 归一化后的优势分数，形状 (Total_Samples,)。
            2. raw_rewards: 原始奖励分数，形状 (Total_Samples,)。
            3. metadata: 包含统计信息（如平均奖励、最大/最小奖励等）的字典，用于日志记录。
    """
    
    # --- 1. 基础校验 ---
    # 确保输入数据长度一致，且能被 group_size 整除
    assert len(rollout_responses) == len(repeated_ground_truths), "Response 和 Ground Truth 数量必须一致"
    assert len(rollout_responses) % group_size == 0, "总样本数必须是 group_size 的整数倍"

    # --- 2. 计算原始奖励 (Raw Rewards) ---
    raw_rewards_list = []
    
    # 遍历每一对 (回答, 标答)
    for response, truth in zip(rollout_responses, repeated_ground_truths):
        # 调用外部传入的奖励函数
        # reward_fn 返回示例: {"reward": 1.0, "format_reward": 1.0, ...}
        score_dict = reward_fn(response, truth)
        
        # 我们只取最终的总分 "reward"
        raw_rewards_list.append(score_dict["reward"])
    
    # 将列表转换为 PyTorch 张量，形状为 (Total_Samples,)，例如 (32,)
    # device 默认为 CPU，后续训练时会自动移到 GPU
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # --- 3. 组内统计 (Group Statistics) ---
    # 计算有多少个独立的问题 (N)
    num_questions = raw_rewards.shape[0] // group_size
    
    # 关键步骤：Reshape
    # 将一维张量 (N*G, ) 变成二维张量 (N, G)
    # 每一行代表一个问题对应的 G 个回答的分数
    grouped_rewards = raw_rewards.view(num_questions, group_size)
    
    # 计算组内均值
    # dim=1 表示沿着“组”的维度（列）求平均
    # keepdim=True 保持形状为 (N, 1)，这是为了后面能利用广播机制让 (N, G) 减去 (N, 1)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    
    # 计算组内标准差 (如果需要)
    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True)

    # --- 4. 计算优势 (Advantage Calculation) ---
    if normalize_by_std:
        # 对应公式 (28)
        # 广播机制：(N, G) - (N, 1) -> (N, G)
        # 每个分数减去它所在组的平均分，再除以标准差
        advantages = (grouped_rewards - group_means) / (group_stds + advantage_eps)
    else:
        # 对应公式 (31) - Dr. GRPO 的建议
        # 仅减去均值
        advantages = grouped_rewards - group_means

    # --- 5. 还原形状 ---
    # 将二维的优势矩阵 (N, G) 拉平回一维 (N*G, )，以便和 log_probs 的形状对齐
    advantages = advantages.view(-1)

    # --- 6. 生成元数据 (Metadata) ---
    # 这些数据不参与计算图，仅用于 WandB/日志 监控训练进度
    metadata = {
        "mean_reward": raw_rewards.mean().item(), # 整体平均分
        "std_reward": raw_rewards.std().item(),   # 整体标准差
        "max_reward": raw_rewards.max().item(),   # 最高分
        "min_reward": raw_rewards.min().item(),   # 最低分
        "mean_advantage": advantages.mean().item(), # 平均优势（理论上应接近0）
    }

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    计算朴素的策略梯度损失（Naive Policy Gradient Loss）。
    本质上是：Loss = -1 * 优势 * 对数概率。

    Args:
        raw_rewards_or_advantages: 形状为 (batch_size, 1) 的张量。
            代表每个生成的回答对应的奖励或优势分。
            对于 GRPO/REINFORCE，通常整句话的所有 token 共享同一个优势值。
        
        policy_log_probs: 形状为 (batch_size, sequence_length) 的张量。
            代表模型生成每个 token 的条件对数概率 log p(token | context)。

    Returns:
        torch.Tensor: 形状为 (batch_size, sequence_length) 的逐 token 损失。
            注意：这里还没有求和或平均，保留了序列维度以便后续应用掩码（Mask）。
    """
    
    # 1. 广播机制 (Broadcasting)
    # raw_rewards_or_advantages 形状是 [B, 1]
    # policy_log_probs 形状是 [B, L]
    # PyTorch 会自动将 [B, 1] 在第二个维度复制 L 次，使其变成 [B, L]
    
    # 2. 计算加权对数概率
    # weighted_log_probs = Advantage * log(p)
    weighted_log_probs = raw_rewards_or_advantages * policy_log_probs
    
    # 3. 取负号
    # 因为我们希望最大化 (Advantage * log_prob)，这等价于最小化 -(Advantage * log_prob)
    loss = -weighted_log_probs
    
    return loss
def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算不带截断的 GRPO 损失（纯重要性采样）。
    """
    # 1. 计算概率比率 ratio = exp(new_log_prob - old_log_prob)
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 2. 按照公式 (34) 计算损失
    # Loss = -(ratio * Advantage)
    # 注意：advantages 形状为 (B, 1)，会广播到 (B, L)
    loss = -(ratio * advantages)
    
    # 3. 记录元数据用于监控比率爆炸情况
    with torch.no_grad():
        metadata = {
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "ratio_min": ratio.min(),
            "clip_fraction": torch.tensor(0.0).to(ratio.device) # 既然没截断，比例始终为 0
        }
        
    return loss, metadata


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    计算 GRPO-Clip 损失函数。
    
    该函数实现了 PPO/GRPO 的核心安全机制：通过限制新旧策略的比率(Ratio)，
    防止模型在单次更新中变化过大，从而保证训练的稳定性。

    Args:
        advantages: 组内归一化后的优势值，形状 (batch_size, 1)。
        policy_log_probs: 当前正在优化的策略的对数概率，形状 (batch_size, sequence_length)。
        old_log_probs: 采样时旧策略的对数概率，形状 (batch_size, sequence_length)。
        cliprange: 截断阈值 epsilon (如 0.2)，定义了 [1-eps, 1+eps] 的“安全区”。

    Returns:
        loss: 逐 token 的损失标量，用于反向传播。
        metadata: 包含 clip_fraction (触发截断的频率) 等监控指标。
    """
    
    # 1. 计算概率比率 ratio = exp(log_prob_new - log_prob_old)
    # 使用对数空间相减再取指数，是为了保证在处理极小概率时的数值稳定性，防止下溢出。
    log_ratio = policy_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    
    # 2. 计算未截断的目标项 (Surrogate 1): 原始策略梯度
    # 如果不做限制，模型会沿着这个方向无限优化，导致策略崩溃。
    surr1 = ratio * advantages
    
    # 3. 计算截断后的目标项 (Surrogate 2): 限制更新幅度
    # 将比率强制约束在 [1-eps, 1+eps] 范围内。
    ratio_clipped = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = ratio_clipped * advantages
    
    # 4. 取最小值 (min) 并取负 (实现梯度上升)
    # 核心逻辑：min(surr1, surr2) 建立了一个“悲观下界”。
    # - 当 A > 0 (奖励): 限制概率增加的上限。若 ratio > 1+eps，min 会选 surr2，此时梯度为 0。
    # - 当 A < 0 (惩罚): 限制概率减少的下限。若 ratio < 1-eps，min 会选 surr2，此时梯度为 0。
    # 只有当模型在安全区内更新，或者模型在“纠正错误方向”的更新时，才会选择 surr1 并保留梯度。
    loss = -torch.min(surr1, surr2)
    
    # 5. 计算元数据 (Metadata) 用于训练监控
    with torch.no_grad():
        # 判定是否触发“梯度截断”的逻辑：
        # 根据 min(surr1, surr2) 的性质，如果最终选了右边的 surr2，说明 surr2 < surr1。
        # 这意味着当前更新已经超出了安全区，且这种超出是被“禁止”的激进更新，
        # 此时该 token 产生的梯度会被锁死为 0。
        
        # 场景分析：
        # - A > 0 且 ratio > 1.2 => surr1=15, surr2=12 => 选 surr2 (触发截断)
        # - A < 0 且 ratio < 0.8 => surr1=-5, surr2=-8 => 选 surr2 (触发截断)
        # - A > 0 且 ratio < 0.8 => surr1=7,  surr2=8  => 选 surr1 (不截断，鼓励模型改错)
        clipped_mask = (surr2 < surr1).float()
        clip_fraction = clipped_mask.mean()
        
        metadata = {
            "clip_fraction": clip_fraction, # 非常关键：若此值接近 1.0，说明学习率过高或模型已停止学习
            "ratio_mean": ratio.mean(),
            "ratio_max": ratio.max(),
            "ratio_min": ratio.min(),
        }
        
    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_no_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    策略梯度损失的统一包装器。根据 loss_type 分发到不同的计算逻辑。

    Args:
        policy_log_probs: 当前策略的对数概率，形状 (batch_size, sequence_length)。
        loss_type: 损失类型字符串。
        raw_rewards: 原始奖励，no_baseline 模式必填。形状 (batch_size, 1)。
        advantages: 归一化优势，reinforce_with_baseline 和 grpo_clip 模式必填。形状 (batch_size, 1)。
        old_log_probs: 旧策略对数概率，grpo_clip 模式必填。形状 (batch_size, sequence_length)。
        cliprange: 截断参数 epsilon，grpo_clip 模式必填。

    Returns:
        Tuple[loss, metadata]: 逐 token 损失张量及元数据字典。
    """

    metadata = {}

    if loss_type == "no_baseline":
        # 断言检查：确保传入了必要的原始奖励
        assert raw_rewards is not None, "no_baseline 模式必须提供 raw_rewards"
        
        # 使用朴素公式：Loss = -raw_reward * log_prob
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "reinforce_with_baseline":
        # 断言检查：确保传入了归一化后的优势值
        assert advantages is not None, "reinforce_with_baseline 模式必须提供 advantages"
        
        # 使用朴素公式：Loss = -advantage * log_prob
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        )

    elif loss_type == "grpo_clip":
        # 断言检查：GRPO 需要比率计算所需的 old_log_probs 和 clip 参数
        assert advantages is not None, "grpo_clip 模式必须提供 advantages"
        assert old_log_probs is not None, "grpo_clip 模式必须提供 old_log_probs"
        assert cliprange is not None, "grpo_clip 模式必须提供 cliprange"
        
        # 调用复杂的 GRPO 截断公式
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
        metadata.update(grpo_metadata)
    elif loss_type == "grpo_no_clip":
        assert advantages is not None and old_log_probs is not None
        loss, grpo_metadata = compute_grpo_no_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs
        )
        metadata.update(grpo_metadata)
    else:
        raise ValueError(f"不支持的 loss_type: {loss_type}")

    return loss, metadata


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

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
) -> torch.Tensor:
    """
    在尊重掩码的情况下计算张量的平均值。
    如果某个维度上的有效元素个数为 0，结果应为 NaN。

    Args:
        tensor: 需要计算平均值的数据张量。
        mask: 掩码张量，1 表示有效，0 表示无效。
        dim: 执行平均操作的维度。
    """
    # 1. 确保掩码与数据张量的类型一致
    mask = mask.to(tensor.dtype)

    # 2. 计算有效元素的总和 (tensor * mask 将无效位置清零)
    masked_sum = torch.sum(tensor * mask, dim=dim)

    # 3. 计算有效元素的数量（分母）
    mask_count = torch.sum(mask, dim=dim)
    
    # 4. 核心修正：直接相除
    # 当 mask_count 为 0 时，0.0 / 0.0 会得到 NaN
    # 这符合测试脚本中 snapshot 的预期
    return masked_sum / mask_count

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    cliprange: Optional[float] = None,
    length_norm_type="mask_mean" # 可选: "mask_mean" (token-level) 或 "mask_normalize" (sentence-level)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    执行 GRPO 的单次 micro-batch 更新。
    包括计算策略梯度损失、掩码平均聚合以及梯度缩放后的反向传播。

    Args:
        policy_log_probs: 当前策略的对数概率 (Batch, Seq).
        response_mask: 掩码 (Batch, Seq)，1 为 Response，0 为 Prompt/Padding。
        gradient_accumulation_steps: 梯度累积步数。
        loss_type: 损失函数类型。
        raw_rewards: 原始奖励 (Batch, 1)。
        advantages: 优势 (Batch, 1)。
        old_log_probs: 旧策略对数概率 (Batch, Seq)。
        cliprange: 截断参数 epsilon。

    Returns:
        loss: 缩放后的标量 Loss (已包含梯度累积因子)。
        metadata: 包含原始 Loss 和其他统计信息的字典。
    """
    
    # 1. 计算 Per-token Loss
    # 调用之前的 Wrapper，得到形状为 (Batch, Seq) 的 Loss 矩阵
    per_token_loss, loss_metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    
    # 2. 聚合 Loss (Aggregation)
    # 第一步：沿序列维度 (dim=1) 计算每个样本的平均 Loss
    # 结果形状: (Batch,)
    if length_norm_type == "mask_mean":
        # [方案 A] Token-level: 每个 token 的权重是平等的 (1/L_actual)
        # 使用之前实现的 masked_mean，每条序列除以它自己的有效长度
        per_example_loss = masked_mean(per_token_loss, response_mask, dim=1)
    else:
        # [方案 B] Sentence-level: 每个 token 的权重被固定常数稀释 (1/C)
        # 对应 PDF 中的 masked_normalize 逻辑
        # 假设 constant 等于序列最大长度（如 1024）
        constant_normalizer = per_token_loss.size(1) 
        per_example_loss = masked_normalize(per_token_loss, response_mask, constant_normalizer, dim=1)

    
    # 第二步：沿批次维度计算整个 microbatch 的平均 Loss
    # 结果形状: Scalar
    microbatch_loss = per_example_loss.mean()
    
    # 3. 梯度缩放 (Gradient Scaling)
    # 为了模拟大 Batch Size，我们需要将 Loss 除以累积步数
    scaled_loss = microbatch_loss / gradient_accumulation_steps
    
    # 4. 反向传播 (Backward)
    # 这一步会将梯度累加到模型参数的 .grad 属性中
    scaled_loss.backward()
    
    # 5. 准备元数据 (Metadata)
    metadata = {
        # 记录未缩放的原始 Loss，方便观察训练曲线
        "loss": microbatch_loss.detach(),
    }
    # 合并来自底层 Loss 函数的元数据 (如 clip_fraction)
    metadata.update(loss_metadata)
    
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
    outputs = vllm_model.generate(prompts, sampling_params)
    
    table_data = []
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
        
        resp_len = len(generated_text)
        all_lengths.append(resp_len)
        
        if r > 0.5:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)
            
        total_reward += r
        total_format_reward += fr
        total_answer_reward += ar

        # 展示前 10 条到 WandB Table
        if i < 10: 
            table_data.append([
                step, 
                prompts[i],
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
        f"{log_prefix}_step": step 
    }

    # 4. 记录到日志系统
    if wandb.run is not None:
        # 记录表格：在 WandB 的 'Media' 或 'Custom Charts' 区域查看
        columns = ["grpo_step", "prompt", "response", "ground_truth", "reward", "format_reward", "answer_reward"]
        wandb.log({f"{log_prefix}/samples": wandb.Table(columns=columns, data=table_data)}, step=step)
        # 记录标量数值
        wandb.log(metrics, step=step)
    
    print(f"Step {step}: Accuracy: {metrics[f'{log_prefix}/accuracy']:.4f}, Avg Len: {metrics[f'{log_prefix}/avg_length']:.1f}")

    return metrics