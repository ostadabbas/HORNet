"""
Evaluate random and uniform frame selection baselines on NExT-QA (MCQ).
No trained model needed — just frame selection + frozen Qwen.

Usage:
  python evaluate_baselines_mcq.py --strategy random --num_select 4 --max_samples 100
  python evaluate_baselines_mcq.py --strategy uniform --num_select 4 --max_samples 100
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import re

from dataset import get_combined_dataset, collate_fn
from util import load_qwen_model, qwen_answer_question


def extract_one_digit_answer(text):
    match = re.search(r'\b(\d)\b', text)
    if match:
        return int(match.group(1))
    return None


def select_frames(videos, strategy, num_select, rng=None):
    """
    videos: [B, T, H, W, C]
    Returns: list of [num_select, H, W, C] tensors
    """
    B, T = videos.shape[0], videos.shape[1]
    selected = []
    for b in range(B):
        if strategy == "uniform":
            idx = np.linspace(0, T - 1, num_select).astype(int)
        elif strategy == "random":
            idx = np.sort(rng.choice(T, size=num_select, replace=False))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        selected.append(videos[b, idx].cpu())
    return selected


def evaluate(dataloader, qwen, qwen_processor, strategy, num_select,
             max_samples=100, seed=42):
    rng = np.random.RandomState(seed)
    correct, total = 0, 0

    pbar = tqdm(dataloader, desc=f"{strategy}-{num_select}")
    for batch in pbar:
        if total >= max_samples:
            break

        # Skip non-MCQ samples
        if batch["choices"] is None:
            continue

        videos = batch["videos"]
        frames_list = select_frames(videos, strategy, num_select, rng)

        for b in range(videos.shape[0]):
            if total >= max_samples:
                break

            prompt = batch["question"][b] + "\n Your choices are: " + \
                     ", ".join(batch["choices"][b])
            gt = int(batch["gt_choice"][b])

            resp = qwen_answer_question(
                prompt, frames_list[b], qwen, qwen_processor, "choice"
            )
            pred = extract_one_digit_answer(resp)
            correct += 1 if pred == gt else 0
            total += 1

        pbar.set_postfix({
            "acc": f"{correct / total:.4f}" if total > 0 else "N/A",
            "n": total,
        })

    acc = correct / total if total > 0 else 0
    print(f"\n[{strategy}, {num_select} frames] "
          f"Samples={total}  Accuracy={acc:.4f}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="uniform",
                        choices=["random", "uniform"])
    parser.add_argument("--num_select", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--qwen_model_name", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of runs for random (results averaged)")
    args = parser.parse_args()

    # NExT-QA only
    eval_dataset = get_combined_dataset(
        num_frames=args.num_frames, h=288, w=288,
        partition="val",
        use_nextqa=True, use_msrvtt=False, use_msvd=False,
    )

    qwen, qwen_processor = load_qwen_model(args.qwen_model_name)

    if args.strategy == "random" and args.num_runs > 1:
        scores = []
        for run in range(args.num_runs):
            print(f"\n=== Random run {run+1}/{args.num_runs} ===")
            dataloader = DataLoader(
                eval_dataset, batch_size=args.batch_size,
                collate_fn=collate_fn, num_workers=0,
                shuffle=True, generator=torch.Generator().manual_seed(1234),
            )
            s = evaluate(dataloader, qwen, qwen_processor,
                         "random", args.num_select,
                         max_samples=args.max_samples, seed=run * 111)
            scores.append(s)
        print(f"\n=== Random ({args.num_runs} runs) ===")
        print(f"Mean Acc: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    else:
        dataloader = DataLoader(
            eval_dataset, batch_size=args.batch_size,
            collate_fn=collate_fn, num_workers=0,
            shuffle=True, generator=torch.Generator().manual_seed(1234),
        )
        evaluate(dataloader, qwen, qwen_processor,
                 args.strategy, args.num_select,
                 max_samples=args.max_samples)