#!/bin/bash
#SBATCH --partition=multigpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=23:59:59
#SBATCH --job-name=red_maiden
#SBATCH --mem=128GB
#SBATCH --ntasks=8
#SBATCH --output=red-maiden-eval-act.%j.out
#SBATCH --error=red-maiden-eval-act.%j.err

cd /projects/zura-storage/Workspace/dora/lmms-eval
source .venv/bin/activate
python -m lmms_eval --model hornet_q3vl --tasks activitynetqa --batch_size 1 --output_path ./results/q3_2b_hornet_activitynetqa --log_samples --trust_remote_code --limit 1000