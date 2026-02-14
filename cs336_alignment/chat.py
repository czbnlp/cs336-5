import os
from openai import OpenAI

# 1. 配置 vLLM 地址 (请确保端口号与你部署时一致，默认通常是 8000)
VLLM_BASE_URL = "http://localhost:8010/v1"

client = OpenAI(
    base_url=VLLM_BASE_URL,
    api_key="empty",
)

def get_model_name():
    """获取当前 vLLM 正在服务的模型名称"""
    try:
        models = client.models.list()
        return models.data[0].id
    except Exception as e:
        print(f"无法连接到 vLLM 服务: {e}")
        exit()

def interactive_chat():
    model_name = get_model_name()
    print(f"已连接到服务！当前模型: {model_name}")
    print("输入 'exit' 或 'quit' 退出程序。")

    while True:
        # 获取终端输入
        user_input = input("\n用户: ").strip()
        
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break

        print("\n助手正在思考...", end="\r")

        try:

            prompt_template_path = 'cs336_alignment/prompts/question_only.prompt'
            # prompt_template_path = 'cs336_alignment/prompts/question_only_fewshot.prompt'
            with open(prompt_template_path, "r") as f:
                prompt_template = f.read()
            full_prompt = prompt_template.replace("{question}", user_input)
            response = client.completions.create(
                model=model_name,
                prompt=full_prompt,
                max_tokens=800,
                temperature=0.01,
                # stop=["</answer>"] # 强制在答案结束后停止
            )

            result = response.choices[0].text
            print(f"助手: {result}")
            
        except Exception as e:
            print(f"\n请求出错: {e}")

if __name__ == "__main__":
    interactive_chat()