import torch
import torch.nn.functional as F

def get_log_probs(model, input_ids):
    """
    计算序列的总对数概率。
    """
    # [batch, seq_len, vocab]
    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 移位 (Shift)：预测下一个词
    # 输入: t1, t2, t3
    # 标签: t2, t3
    shift_logits = log_probs[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    
    # 提取实际出现的 token 的概率
    per_token_log_probs = torch.gather(
        shift_logits, dim=2, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    return per_token_log_probs.sum()

def compute_dpo_loss(
    model, 
    ref_model, 
    tokenizer, 
    beta, 
    prompt, 
    response_chosen, 
    response_rejected
):
    """
    计算单样本 DPO Loss。
    接收 7 个参数以适配测试套件。
    """
    device = model.device

    # 1. 按照 Alpaca 模板格式化 (PDF 第 9 页和 18 页要求)
    def format_prompt(p, r):
        # 注意换行符的准确性
        return (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{p}\n\n"
            f"### Response:\n{r}"
        )

    # 2. 拼接并添加 EOS Token (PDF 明确要求)
    full_chosen = format_prompt(prompt, response_chosen) + tokenizer.eos_token
    full_rejected = format_prompt(prompt, response_rejected) + tokenizer.eos_token

    # 3. Tokenize
    # 注意：设置 add_special_tokens=False 因为我们在模板中手动处理或默认已有 BOS
    chosen_ids = tokenizer.encode(full_chosen, return_tensors="pt").to(device)
    rejected_ids = tokenizer.encode(full_rejected, return_tensors="pt").to(device)

    # 4. 计算训练模型 (Policy) 的对数概率
    lp_theta_chosen = get_log_probs(model, chosen_ids)
    lp_theta_rejected = get_log_probs(model, rejected_ids)

    # 5. 计算参考模型 (Reference) 的对数概率
    # 参考模型需要放在它所在的设备上计算，且不需要梯度
    with torch.no_grad():
        lp_ref_chosen = get_log_probs(ref_model, chosen_ids.to(ref_model.device)).to(device)
        lp_ref_rejected = get_log_probs(ref_model, rejected_ids.to(ref_model.device)).to(device)

    # 6. 计算 DPO 核心公式 (Equation 3)
    # 计算隐式奖励差
    # beta * [ (log pi_theta(y_w|x) - log pi_ref(y_w|x)) - (log pi_theta(y_l|x) - log pi_ref(y_l|x)) ]
    pi_log_ratio = lp_theta_chosen - lp_theta_rejected
    ref_log_ratio = lp_ref_chosen - lp_ref_rejected
    
    logits = beta * (pi_log_ratio - ref_log_ratio)
    
    # 7. 计算 Loss
    loss = -F.logsigmoid(logits)
    
    return loss