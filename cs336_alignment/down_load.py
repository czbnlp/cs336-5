import os
from huggingface_hub import snapshot_download

def download_dataset():
    # 数据集在 HF 上的 ID
    # repo_id = "garg-aayush/sft-cs336-assign5-datasets"
    # repo_id = "openbmb/UltraChat"
    # repo_id = "TIGER-Lab/MMLU-Pro"
    # repo_id = "Anthropic/hh-rlhf"
    repo_id = "hiyouga/math12k"
    
    # 本地保存的目标目录
    local_dir = f"data/{repo_id.split('/')[-1]}"
    
    # 如果 data 目录不存在，则创建
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
        print(f"创建目录: {local_dir}")

    print(f"开始从 Hugging Face 下载数据集 '{repo_id}'...")
    
    try:
        # 下载整个数据集存储库
        # repo_type="dataset" 必须指定，因为默认是下载模型(model)
        # local_dir_use_symlinks=False 确保下载的是真实文件而非软链接
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            repo_type="dataset",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"\n下载完成！文件保存在: {os.path.abspath(local_dir)}")
        
        # 列出下载后的文件
        print("目录内容:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                print(f"- {os.path.join(root, file)}")

    except Exception as e:
        print(f"\n下载出错: {e}")

def download_model():
    # 作业中使用的是 Base 版本
    repo_id = "Qwen/Qwen2.5-3B"
    
    # 本地保存路径
    local_dir = "model/" + repo_id.split('/')[-1]
    
    print(f"开始下载 {repo_id} 到 {local_dir} 文件夹...")
    
    try:
        # snapshot_download 会下载整个仓库
        # local_dir_use_symlinks=False 表示下载实际文件而不是软链接
        # 这对于后续移动文件或在没有缓存环境中使用非常重要
        model_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, 
            resume_download=True
        )
        print(f"下载成功")
        
    except Exception as e:
        print(f"下载过程中出错: {e}")

if __name__ == "__main__":
    # download_model()

    download_dataset()