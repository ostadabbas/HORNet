"""
Evaluate random and uniform frame selection baselines on MSVD-QA.
No trained model needed — just frame selection + frozen Qwen.

Usage:
  python evaluate_baselines.py --strategy random --num_select 4 --num_runs 5
  python evaluate_baselines.py --strategy uniform --num_select 4
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import get_combined_dataset, collate_fn
from reward import string_f1
from util import load_qwen_model, qwen_answer_question


def select_frames(videos, strategy, num_select, rng=None):
    """
    videos: [B, T, H, W, C]
    Returns: list of [num_select, H, W, C] tensors, one per batch item
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


def evaluate(dataloader, qwen, qwen_processor, strategy, num_select, max_samples=100, seed=42):
    rng = np.random.RandomState(seed)
    correct, total = 0, 0
    f1_sum = 0.0

    pbar = tqdm(dataloader, desc=f"{strategy}-{num_select}")
    for batch in pbar:
        if total >= max_samples:
            break

        videos = batch["videos"]
        prompts = batch["question"]
        answers = batch["answer"]

        frames_list = select_frames(videos, strategy, num_select, rng)

        for b in range(len(prompts)):
            if total >= max_samples:
                break

            pred = qwen_answer_question(
                prompts[b], frames_list[b], qwen, qwen_processor, "msqa"
            )

            f1 = string_f1(pred, answers[b], is_simple=True)
            f1_sum += f1
            total += 1

        pbar.set_postfix({
            "f1": f"{f1_sum / total:.4f}",
            "n": total,
        })

    avg_f1 = f1_sum / total if total > 0 else 0
    print(f"\n[{strategy}, {num_select} frames] "
          f"Samples={total}  Avg F1={avg_f1:.4f}")
    return avg_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="uniform",
                        choices=["random", "uniform"])
    parser.add_argument("--num_select", type=int, default=4,
                        help="Number of frames to select")
    parser.add_argument("--num_frames", type=int, default=32,
                        help="Total frames sampled from video")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Max eval samples")
    parser.add_argument("--num_runs", type=int, default=1,
                        help="Number of runs (>1 for random to average)")
    parser.add_argument("--qwen_model_name", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_msvd", action="store_true", default=False)
    parser.add_argument("--use_msrvtt", action="store_true", default=False)
    parser.add_argument("--use_nextqa", action="store_true", default=False)
    args = parser.parse_args()

    if not (args.use_msvd or args.use_msrvtt or args.use_nextqa):
        args.use_msvd = True  # default to MSVD

    # Load dataset
    g = torch.Generator().manual_seed(1234)
    eval_dataset = get_combined_dataset(
        num_frames=args.num_frames, h=288, w=288,
        partition="val",
        use_nextqa=args.use_nextqa,
        use_msrvtt=args.use_msrvtt,
        use_msvd=args.use_msvd,
    )
    dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=0,
        shuffle=True, generator=g,
    )

    # Load Qwen
    qwen, qwen_processor = load_qwen_model(args.qwen_model_name)

    # Run evaluation
    if args.strategy == "random" and args.num_runs > 1:
        scores = []
        for run in range(args.num_runs):
            print(f"\n=== Random run {run+1}/{args.num_runs} ===")
            # Recreate dataloader each run for same shuffle order
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
        print(f"Mean F1: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    else:
        evaluate(dataloader, qwen, qwen_processor,
                 args.strategy, args.num_select,
                 max_samples=args.max_samples)