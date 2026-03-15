#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=23:59:59
#SBATCH --job-name=hornet_3dataset_train_lr_1e5
#SBATCH --mem=64GB
#SBATCH --ntasks=8
#SBATCH --output=hornet-3dataset-train_lr_1e5.%j.out
#SBATCH --error=hornet-3dataset-train_lr_1e5.%j.err

cd /projects/zura-storage/Workspace/Hornet
source env_hornet/bin/activate

python train.py \
    --num_frames 256 \
    --batch_size 8 \
    --save_eval_freq 500 \
    --num_epochs 1 \
    --n_samples 8 \
    --top_k 8 \
    --save_loc ./checkpoints_3dataset_lr_1e-5_v3_long \
    --load_from ./checkpoints_3dataset_lr_1e-5_v3_short/checkpoint-0.1500.pt \
    --max_iter 1501 \
    --scheme long 