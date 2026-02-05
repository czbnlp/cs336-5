# #!/bin/bash
git config user.name "Zhibo Chu"
git config user.email "zb.chu@mail.ustc.edu.cn"

# 检查是否输入了 commit 信息
if [ -z "$1" ]; then
    echo "错误: 请提供 commit 提交信息！"
    echo "使用方法: bash sync.sh \"你的提交备注\""
    exit 1
fi

# 1. 执行 git add
echo "正在添加文件..."
git add .

# 2. 执行 git commit，使用脚本收到的第一个参数 $1
echo "正在提交: $1"
git commit -m "$1"

# 3. 执行 git push
echo "正在推送到 GitHub..."
git push

echo "✅ 更新完成！"