#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:59:59
#SBATCH --job-name=red_maiden
#SBATCH --mem=128GB
#SBATCH --ntasks=8
#SBATCH --output=red-maiden-eval.%j.out
#SBATCH --error=red-maiden-eval.%j.err

cd /projects/zura-storage/Workspace/Hornet
source env_hornet/bin/activate
python evaluate.py \
    --num_frames 32 \
    --top_k 4 \
    --load_from checkpoints_3datasets_v4_long/checkpoint-0.1500.pt \
    --dataset nextqa