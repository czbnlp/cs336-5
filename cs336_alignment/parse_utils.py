import re

def parse_mmlu_response(model_response: str) -> str | None:
    """
    根据作业要求解析 MMLU 响应。
    匹配格式："The correct answer is [A-D]"
    """
    if not model_response:
        return None
    
    # 使用正则表达式匹配提示词要求的特定句子格式
    # [A-D] 捕获选项，忽略大小写和末尾可能的标点
    pattern = r"[Tt]he correct answer is\s*([A-D])"
    match = re.search(pattern, model_response)
    
    if match:
        return match.group(1).upper()
    
    return None

def parse_gsm8k_response(model_response: str) -> float | None:
    """
    从模型输出中提取最后一个数字。
    1. 移除数字中的逗号（如 1,234 -> 1234）。
    2. 使用正则提取所有整数和浮点数。
    3. 返回最后一个数字。
    """
    if not model_response:
        return None
    
    # 移除数字中间的逗号 (1,000 -> 1000)
    # 注意：这里只移除数字间的逗号，不破坏句子结构
    text = re.sub(r"(\d),(\d)", r"\1\2", model_response)
    
    # 正则：匹配整数或浮点数
    # [-+]? 符号可选，\d+ 数字，(\.\d+)? 小数部分可选
    pattern = r"[-+]?\d+(?:\.\d+)?"
    numbers = re.findall(pattern, text)
    
    if numbers:
        try:
            # 返回提取到的最后一个数字
            return str(numbers[-1])
        except ValueError:
            return None
    return None