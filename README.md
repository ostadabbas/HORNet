# HORnet: Hierarchical Optimized Representation Network

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/bishoygaloaa/HORnet)
[![Google Drive](https://img.shields.io/badge/Google%20Drive-Checkpoints-yellow)](https://drive.google.com) *(Coming Soon)*

**HORnet** is a reinforcement learning-based frame selection system for efficient video understanding. It uses GRPO (Group Relative Policy Optimization) to train a policy network that intelligently selects the most informative frames from videos, achieving up to 99% frame reduction and 93% faster processing with competitive accuracy on video question answering tasks.

## Overview

HORnet learns to select a subset of frames from videos that maximize downstream task performance. The system consists of:

- **Frame Selector**: A policy network trained with GRPO that outputs frame selection probabilities
- **Video Encoder**: TimeSformer encoder for extracting spatio-temporal visual features
- **VLM Integration**: Compatible with Qwen3-VL and other vision-language models

## Installation

### Prerequisites

```bash
# Create virtual environment
python -m venv env_hornet
source env_hornet/bin/activate

# Install dependencies
pip install torch torchvision transformers
pip install qwen-vl-utils
pip install trl tqdm loguru
```

### For Evaluation with lmms-eval

Clone and install the lmms-eval library:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval
pip install -e .
```

## Model Weights

Pre-trained HORnet checkpoints are available on Hugging Face:

🤗 **[bishoygaloaa/HORnet](https://huggingface.co/bishoygaloaa/HORnet)**

### Download Checkpoints

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download all checkpoints
huggingface-cli download bishoygaloaa/HORnet --local-dir ./checkpoints --repo-type model

# Or download specific checkpoints
huggingface-cli download bishoygaloaa/HORnet checkpoints/long/checkpoint-0.1500.pt --local-dir ./checkpoints --repo-type model
```

### Available Checkpoints

| Checkpoint | Training Scheme | Use Case | Path |
|------------|----------------|----------|------|
| `checkpoint-0.1500.pt` | Long videos | **Main model** (used in lmms-eval) | `checkpoints/long/` |
| `checkpoint-0.250.pt` | Long videos | Multiple-choice QA evaluation | `checkpoints/long/` |
| `checkpoint-0.550.pt` | Short videos | Short video QA (MSVD), VLM ablation | `checkpoints/short/` |

**Alternative:** Checkpoints will also be available on Google Drive *(Coming Soon)*

## Training

HORnet supports multiple training schemes and datasets:

### Basic Training

```bash
python train.py \
  --num_frames 256 \
  --batch_size 8 \
  --save_eval_freq 500 \
  --num_epochs 1 \
  --n_samples 8 \
  --top_k 16 \
  --save_loc ./checkpoints \
  --scheme long
```

### Training Schemes

- **Long videos**: `--scheme long` (e.g., ActivityNet-QA)
- **Short videos**: `--scheme short` (e.g., MSVD-QA)
- **SFT**: Supervised fine-tuning with `train_sft.py`
- **PPO**: Alternative RL training with `train_ppo.py`

### SLURM Scripts

Pre-configured SLURM scripts are available in `slurm/`:

```bash
sbatch slurm/hornet_train.sh
sbatch slurm/hornet_train_sft.sh
sbatch slurm/hornet_train_ppo.sh
```

## Evaluation

### Using lmms-eval

1. **Add model files** to your lmms-eval installation:
   - Copy `lmms_eval_utils/hornet_q3vl.py` → `lmms-eval/lmms_eval/models/chat/`
   - Copy `lmms_eval_utils/qwen3_vl.py` → `lmms-eval/lmms_eval/models/chat/`
   - Copy `lmms_eval_utils/hornet.py` → `lmms-eval/lmms_eval/`

2. **Update model registry** in `lmms-eval/lmms_eval/models/chat/__init__.py`:

```python
AVAILABLE_CHAT_TEMPLATE_MODELS = {
    "bagel_lmms_engine": "BagelLmmsEngine",
    "llava_hf": "LlavaHf",
    "qwen3_vl": "Qwen3_VL",
    "hornet_q3vl": "Hornet_Q3VL",  # Add this
    "qwen2_5_vl": "Qwen2_5_VL",
    # ... other models
}
```

3. **Run evaluation**:

```bash
# Evaluate HORnet on ActivityNet-QA
python -m lmms_eval \
  --model hornet_q3vl \
  --tasks activitynetqa \
  --batch_size 1 \
  --output_path ./results/hornet_activitynetqa \
  --log_samples

# Evaluate baseline Qwen3-VL
python -m lmms_eval \
  --model qwen3_vl \
  --tasks activitynetqa \
  --batch_size 1 \
  --output_path ./results/qwen3_activitynetqa \
  --log_samples
```

### Supported Benchmarks

- ActivityNet-QA
- MSVD-QA
- MSRVTT-QA
- NExT-QA
- Video-MME

### SLURM Evaluation Scripts

```bash
sbatch slurm/hornet_eval_vqa_act.sh  # ActivityNet-QA
sbatch slurm/hornet_eval_vqa_og.sh   # Other benchmarks
sbatch slurm/hornet_eval_ms.sh       # MSVD/MSRVTT
```

## Results

### Table 1: Research Gap

| Method | Learned Selection | Reward Optimized | Frozen VLM | Param. Efficient |
|--------|:-----------------:|:----------------:|:----------:|:----------------:|
| Uniform Sampling | ✗ | ✗ | ✓ | ✓ |
| SeViLA | ✓ | ✗ | ✗ | ✗ |
| Frame-Voyager | ✓ | ✗ | ✗ | ✗ |
| F2C | ✗ | ✗ | ✓ | ✓ |
| ReFoCUS | ✓ | ✓ | ∼ | ✗ |
| ViaRL | ✓ | ✓ | ✗ | ✗ |
| **HORNet (Ours)** | ✓ | ✓ | ✓ | ✓ |

### Table 2: Open-Ended QA Results

| Dataset | Model | F1-Lev ↑ | Frame Sel. (s) ↓ | Qwen Proc. (s) ↓ | Avg. Frames ↓ |
|---------|-------|:---------:|:-----------------:|:-----------------:|:-------------:|
| MSVD | Qwen3-VL-2B (Baseline) | 0.3483 | – | 0.28 | 11.65 |
| MSVD | HORNet+Qwen3-VL-2B (Ours) | **0.3543** (+1.7%) | 0.12 | **0.10** (↓64%) | **4.00** (↓66%) |
| MSRVTT | Qwen3-VL-2B (Baseline) | 0.3209 | – | 0.58 | 47.52 |
| MSRVTT | HORNet+Qwen3-VL-2B (Ours) | 0.3029 (-5.6%) | 0.09 | **0.09** (↓84%) | **4.00** (↓92%) |
| NextOE | Qwen3-VL-2B (Baseline) | 0.3045 | – | 1.01 | 1157.88 |
| NextOE | HORNet+Qwen3-VL-2B (Ours) | 0.2738 (-10.1%) | 0.52 | **0.19** (↓81%) | **8.00** (↓99%) |

### Table 3: Multiple-Choice QA Results

| Dataset | Model | Accuracy (%) ↑ | Frame Sel. (s) ↓ | Qwen Proc. (s) ↓ | Avg. Frames ↓ |
|---------|-------|:--------------:|:-----------------:|:-----------------:|:-------------:|
| VideoMME | Qwen3-VL-2B (Baseline) | 68.30 | – | 2.53 | 3066.73 |
| VideoMME | HORNet+Qwen3-VL-2B (Ours) | 52.10 (-16.2%) | 1.51 | **0.18** (↓93%) | **8.00** (↓99%) |
| ActivityNetQA | Qwen3-VL-2B (Baseline) | 75.00 | – | 2.37 | 3152.49 |
| ActivityNetQA | HORNet+Qwen3-VL-2B (Ours) | **68.80** (-6.2%) | 1.64 | **0.17** (↓93%) | **8.00** (↓99%) |
| NextQA | Qwen3-VL-2B (Baseline) | 76.80 | – | 0.98 | 1157.88 |
| NextQA | HORNet+Qwen3-VL-2B (Ours) | **71.50** (-5.3%) | 0.53 | **0.25** (↓74%) | **8.00** (↓99%) |

### Table 4: Training Objective Ablation

| Training | MSVD (F1-Lev ↑) | MSRVTT (F1-Lev ↑) |
|----------|:---------------:|:-----------------:|
| No training (baseline) | 0.3483 | 0.3209 |
| SFT (weighted BCE) | 0.3495 | 0.2882 |
| PPO (clipped surrogate) | **0.3585** | 0.2948 |
| **GRPO (Ours)** | 0.3543 | **0.3029** |

### Table 5: Frame Selection Strategy Ablation

| Strategy | MSVD (F1-Lev ↑) | MSRVTT (F1-Lev ↑) | NExT-QA (Acc. ↑) |
|----------|:---------------:|:-----------------:|:----------------:|
| Random | 0.3527 | 0.3027 | 65.88 |
| Uniform | 0.3493 | **0.3058** | 64.24 |
| **HORNet** | **0.3543** | 0.3029 | **71.50** |

### Table 6: VLM Answerer Ablation on MSVD-QA

| VLM Answerer | Size | F1-Lev ↑ |
|--------------|:----:|:---------:|
| Qwen3-VL-Instruct (baseline) | 2B | 0.3483 |
| Qwen3-VL-Instruct + HORNet | 2B | 0.3543 |
| **Qwen2.5-VL-Instruct + HORNet** | 3B | **0.3846** |

## Project Structure

```
hornet/
├── model.py              # Core model architecture (VisionGRPOPolicy)
├── train.py              # GRPO training script
├── train_sft.py          # Supervised fine-tuning
├── train_ppo.py          # PPO training
├── reward.py             # Reward computation for RL
├── dataset.py            # Dataset loaders
├── evaluate.py           # Evaluation utilities
├── vp.py                 # Video processing utilities
├── util.py               # Helper functions
├── slurm/                # SLURM job scripts
└── lmms_eval_utils/      # lmms-eval integration
    ├── hornet_q3vl.py    # HORnet + Qwen3-VL model
    ├── qwen3_vl.py       # Baseline Qwen3-VL
    └── hornet.py         # HORnet utilities
```

## Key Features

- **Massive Efficiency Gains**: Reduces frames by up to 99% (from 3000+ to 8 frames) with 74-93% faster VLM processing
- **Competitive Accuracy**: Maintains or slightly improves accuracy on short videos (MSVD: +1.7%), with acceptable trade-offs on long videos
- **GRPO Training**: Group-based policy optimization for stable RL training, outperforming SFT and competitive with PPO
- **TimeSformer Encoder**: Lightweight spatio-temporal transformer for video feature extraction
- **Flexible Architecture**: Compatible with various VLMs (Qwen3-VL, Qwen2.5-VL)
- **Benchmark Integration**: Easy evaluation via lmms-eval library

## Citation

If you use HORnet in your research, please cite:

```bibtex
@article{bai2026hornet,
  title={HORNet: Task-Guided Frame Selection for Video Question Answering with Vision-Language Models},
  author={Xiangyu Bai*, Bishoy Galoaa*, and Sarah Ostadabbas},
  booktitle={arxiv preprint},
  year={2026}
}
```

## License

See LICENSE file for details.

## Acknowledgments

- Built on top of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- Uses TimeSformer architecture for efficient video encoding
- Integrates with [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) vision-language model
