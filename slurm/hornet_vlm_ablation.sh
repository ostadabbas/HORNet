#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --job-name=hornet_vlm_ablation
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=hornet-vlm-ablation.%j.out
#SBATCH --error=hornet-vlm-ablation.%j.err

cd /projects/zura-storage/Workspace/Hornet
source env_hornet/bin/activate

python evaluate_vlm.py \
    --load_from checkpoints_3datasets_v4_short/checkpoint-0.550.pt \
    --qwen_model_name Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset msvd
