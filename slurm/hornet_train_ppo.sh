#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=07:59:59
#SBATCH --job-name=hornet_ppo_msvd_ablation
#SBATCH --mem=128GB
#SBATCH --ntasks=8
#SBATCH --output=hornet-ppo-msvd-ablation.%j.out
#SBATCH --error=hornet-ppo-msvd-ablation.%j.err

cd /projects/zura-storage/Workspace/Hornet
source env_hornet/bin/activate

python train_ppo.py \
    --num_frames 32 \
    --batch_size 8 \
    --save_eval_freq 500 \
    --num_epochs 1 \
    --n_samples 8 \
    --top_k 8 \
    --ppo_epochs 3 \
    --clip_eps 0.2 \
    --vf_coef 0.5 \
    --ent_coef 0.01 \
    --lr 1e-5 \
    --scheme short \
    --use_msvd \
    --save_loc ./checkpoints_ppo_msvd_ablation