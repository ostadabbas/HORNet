#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:59:59
#SBATCH --job-name=hornet_sft_msvd_ablation
#SBATCH --mem=128GB
#SBATCH --ntasks=8
#SBATCH --output=hornet-sft-msvd-ablation.%j.out
#SBATCH --error=hornet-sft-msvd-ablation.%j.err

cd /projects/zura-storage/Workspace/Hornet
source env_hornet/bin/activate

python train_sft.py \
    --num_frames 32 \
    --batch_size 8 \
    --save_eval_freq 500 \
    --num_epochs 1 \
    --top_k 8 \
    --lr 1e-5 \
    --loss_version v1 \
    --scheme short \
    --use_msvd \
    --save_loc ./checkpoints_sft_msvd_ablation