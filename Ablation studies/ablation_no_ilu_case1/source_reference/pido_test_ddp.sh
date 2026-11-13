#!/bin/bash
#SBATCH --job-name=run_node03       # 作业名称
#SBATCH --output=pidn_output_%j     # 标准输出文件（%j 会被替换为作业 ID）
#SBATCH --error=pidn_error_%j.txt   # 标准错误文件（%j 会被替换为作业 ID）
#SBATCH --time=100:00:00            # 运行时间限制
#SBATCH --partition=gpu             # 请求 GPU 分区
#SBATCH --cpus-per-task=10          # 节点请求的 CPU 核心数
#SBATCH --mem=128G                  # 请求内存大小
#SBATCH --gres=gpu:2                # DDP 并行：请求 4 张 GPU（与 nproc_per_node 一致）
#SBATCH --nodelist=node03           # 指定使用 node06 节点

# 避免 torchrun 的 OMP_NUM_THREADS 提示：每进程 1 线程，防止多进程时 CPU 过载
export OMP_NUM_THREADS=1

# 激活虚拟环境
eval "$(/public/apps/miniconda3/bin/conda shell.bash hook)"
conda activate deepnet

# 切换到存放 Python 脚本的目录
cd /public/home/lld/DeepOnet-modelA-Preprocessing

# DDP 并行：每张 GPU 一个进程，使用 torchrun 启动
# 若只申请了 2 张 GPU，请改为 --nproc_per_node=2
NUM_GPUS=2
torchrun --nproc_per_node=${NUM_GPUS} cnn_branch_test2_DDP.py