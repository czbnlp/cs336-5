# #!/bin/bash
  # git config user.name "Zhibo Chu"

  # # 替换为你的 GitHub 注册邮箱（必须一致，否则 GitHub 不会显示你的头像）
  # git config user.email "zb.chu@mail.ustc.edu.cn"


# # 检查是否输入了 commit 信息
# if [ -z "$1" ]; then
#     echo "错误: 请提供 commit 提交信息！"
#     echo "使用方法: bash sync.sh \"你的提交备注\""
#     exit 1
# fi

# # 1. 执行 git add
# echo "正在添加文件..."
# git add .

# # 2. 执行 git commit，使用脚本收到的第一个参数 $1
# echo "正在提交: $1"
# git commit -m "$1"

# # 3. 执行 git push
# echo "正在推送到 GitHub..."
# git push

# echo "✅ 更新完成！"



git init
git config user.name "Zhibo Chu"
git config user.email "zb.chu@mail.ustc.edu.cn"
git add .
git commit -m "Initial commit with correct author"

# 关联你的仓库（换成你自己的地址）
git remote add origin https://github.com/czbnlp/cs336-5.git

# 强制覆盖 GitHub 上的记录
git push -u origin main --force