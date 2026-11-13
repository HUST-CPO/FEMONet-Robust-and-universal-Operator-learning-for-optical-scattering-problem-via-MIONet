#!/bin/bash
#SBATCH --job-name=run_node06       # 作业名称
#SBATCH --output=pidn_output_%j     # 标准输出文件（%j 会被替换为作业 ID）
#SBATCH --error=pidn_error_%j.txt         # 标准错误文件（%j 会被替换为作业 ID）
#SBATCH --time=100:00:00                  # 运行时间限制
#SBATCH --partition=gpu                  # 请求 GPU 分区
#SBATCH --cpus-per-task=10			            # 节点请求的cpu核心数
#SBATCH --mem=128G                          # 请求内存大小
#SBATCH --gres=gpu:1                     # 请求 4 个 GPU 资源
#SBATCH --nodelist=node06              # 指定使用 node06 节点


# 激活虚拟环境
eval "$(/public/apps/miniconda3/bin/conda shell.bash hook)"
conda activate deepnet

# 切换到存放 Python 脚本的目录
cd /public/home/lld/DeepOnet-modelA-Preprocessing


# 执行 Python 脚本
python cnn_branch_test2.py