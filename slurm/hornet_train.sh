#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=red_maiden
#SBATCH --mem=128GB
#SBATCH --ntasks=8
#SBATCH --output=red-maiden-train.%j.out
#SBATCH --error=red-maiden-train.%j.err

cd /projects/zura-storage/Workspace/Hornet
source env_hornet/bin/activate
python train.py --num_frames 256 --batch_size 8 --save_eval_freq 500 --num_epochs 1 --n_samples 8 --top_k 16 --save_loc ./checkpoints_3datasets_v3_k16 --scheme long