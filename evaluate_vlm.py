"""
Evaluate HORNet policy with different frozen VLM answerers.
Same trained policy, only the answerer changes.

Usage:
  # Qwen3-VL (default)
  python evaluate_vlm.py --load_from checkpoints_3datasets_v4_short/checkpoint-0.550.pt --qwen_model_name Qwen/Qwen3-VL-2B-Instruct --dataset msvd

  # Qwen2.5-VL
  python evaluate_vlm.py --load_from checkpoints_3datasets_v4_short/checkpoint-0.550.pt --qwen_model_name Qwen/Qwen2.5-VL-2B-Instruct --dataset msvd
"""

import argparse
import torch
import time
import re
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import get_combined_dataset, collate_fn
from model import VisionGRPOPolicy
from reward import string_f1
from util import load_qwen_model, qwen_answer_question, get_action_by_k


def extract_one_digit_answer(text):
    match = re.search(r'\b(\d)\b', text)
    if match:
        return int(match.group(1))
    return None


def eval_short(model, dataloader, top_k, qwen, qwen_processor, max_samples=999999):
    """Evaluate on short-form QA (MSVD / MSRVTT) — reports F1."""
    model.eval()
    device = "cuda"
    f1_sum, total = 0.0, 0

    pbar = tqdm(dataloader, desc="Eval (F1)")
    for batch in pbar:
        if total >= max_samples:
            break

        videos = batch["videos"].to(device)
        prompt = batch["question"][0]
        gt = batch["answer"][0]

        out = model(videos)
        keep_prob = out["keep_prob"]
        actions = get_action_by_k(keep_prob, 1, top_k, random_sample=False)
        idx = torch.nonzero(actions[0][0]).squeeze(-1)
        frames = videos.to("cpu")[0, idx.to("cpu")]

        pred = qwen_answer_question(prompt, frames, qwen, qwen_processor, "msqa")
        f1 = string_f1(pred, gt, is_simple=True)
        f1_sum += f1
        total += 1

        pbar.set_postfix({"f1": f"{f1_sum / total:.4f}", "n": total})

    print(f"\nFinal F1: {f1_sum / total:.4f}  (n={total})")


def eval_mcq(model, dataloader, top_k, qwen, qwen_processor, max_samples=999999):
    """Evaluate on MCQ (NExT-QA) — reports accuracy."""
    model.eval()
    device = "cuda"
    correct, total = 0, 0

    pbar = tqdm(dataloader, desc="Eval (Acc)")
    for batch in pbar:
        if total >= max_samples:
            break
        if batch["choices"] is None:
            continue

        videos = batch["videos"].to(device)
        prompt = batch["question"][0] + "\n Your choices are: " + \
                 ", ".join(batch["choices"][0])
        gt = int(batch["gt_choice"][0])

        out = model(videos)
        keep_prob = out["keep_prob"]
        actions = get_action_by_k(keep_prob, 1, top_k, random_sample=False)
        idx = torch.nonzero(actions[0][0]).squeeze(-1)
        frames = videos.to("cpu")[0, idx.to("cpu")]

        resp = qwen_answer_question(prompt, frames, qwen, qwen_processor, "choice")
        pred = extract_one_digit_answer(resp)
        correct += 1 if pred == gt else 0
        total += 1

        pbar.set_postfix({"acc": f"{correct / total:.4f}", "n": total})

    print(f"\nFinal Accuracy: {correct / total:.4f}  (n={total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HORNet with different VLMs")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--load_from", type=str, required=True)
    parser.add_argument("--qwen_model_name", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset", type=str, default="msvd",
                        choices=["msvd", "msrvtt", "nextqa"])
    parser.add_argument("--max_samples", type=int, default=999999)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VLM
    qwen_model, qwen_processor = load_qwen_model(args.qwen_model_name)

    # Load HORNet policy
    model = VisionGRPOPolicy(
        encoder_name=None,
        feat_dim=768,
        action_dim=1,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
    ).to(device)

    state = torch.load(args.load_from, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"Loaded policy from {args.load_from}")
    print(f"VLM answerer: {args.qwen_model_name}")

    # Load dataset
    g = torch.Generator().manual_seed(1234)
    eval_dataset = get_combined_dataset(
        num_frames=args.num_frames, h=288, w=288,
        partition="val",
        use_msvd=(args.dataset == "msvd"),
        use_msrvtt=(args.dataset == "msrvtt"),
        use_nextqa=(args.dataset == "nextqa"),
    )
    dataloader = DataLoader(
        eval_dataset, batch_size=1,
        collate_fn=collate_fn, num_workers=0,
        shuffle=True, generator=g,
    )

    # Run eval
    if args.dataset == "nextqa":
        eval_mcq(model, dataloader, args.top_k, qwen_model, qwen_processor,
                 max_samples=args.max_samples)
    else:
        eval_short(model, dataloader, args.top_k, qwen_model, qwen_processor,
                   max_samples=args.max_samples)