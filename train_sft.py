"""
SFT training for frame selection.

Instead of GRPO (sample K candidates → reward → policy gradient),
we do:
  1. Forward pass → keep_prob per frame
  2. Select top-k frames (greedy)
  3. Run Qwen on selected frames → get answer
  4. Compute reward (F1 vs ground truth)
  5. Use reward as a supervised target: push keep_prob toward 1 for
     selected frames that led to high reward, toward 0 otherwise.

This is essentially "distilling" the reward signal into the selector.
"""

from dataset import get_combined_dataset, collate_fn
from model import VisionGRPOPolicy
from reward import compute_hornet_rewards, string_f1
from util import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from evaluate_mcq import eval_model, get_eval_dataset
import argparse


def sft_loss(logits, rewards, top_k):
    keep_logits = logits.squeeze(-1)
    B, T = keep_logits.shape

    keep_prob = torch.sigmoid(keep_logits.detach())
    topk_idx = keep_prob.topk(top_k, dim=1).indices
    targets = torch.zeros_like(keep_logits)
    targets.scatter_(1, topk_idx, 1.0)

    rewards_t = torch.tensor(rewards, device=keep_logits.device).float()
    weights = rewards_t.clamp(min=0.0).unsqueeze(1)

    bce = F.binary_cross_entropy_with_logits(
        keep_logits, targets, reduction='none'
    )

    loss = (bce * weights).mean()
    return loss


def sft_loss_v2(logits, rewards, top_k):
    keep_logits = logits.squeeze(-1)
    B, T = keep_logits.shape

    keep_prob = torch.sigmoid(keep_logits.detach())
    topk_idx = keep_prob.topk(top_k, dim=1).indices

    rewards_t = torch.tensor(rewards, device=keep_logits.device).float()

    targets = torch.zeros_like(keep_logits)
    reward_expanded = rewards_t.unsqueeze(1).expand(-1, top_k)
    targets.scatter_(1, topk_idx, reward_expanded.clamp(min=0.1))

    bce = F.binary_cross_entropy_with_logits(
        keep_logits, targets, reduction='mean'
    )
    return bce


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SFT Frame Selection Training")

    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--encoder_name", type=str,
                        default=None,
                        help="Encoder name (None = TimeSformerTiny)")
    parser.add_argument("--qwen_model_name", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_eval_freq", type=int, default=2500)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (typically higher than GRPO)")
    parser.add_argument("--loss_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="v1=weighted BCE, v2=soft-target BCE")
    parser.add_argument("--save_loc", type=str, default="./checkpoints_sft")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--max_iter", type=int, default=99999,
                        help="Max training iterations")
    # ---- Dataset selection flags ----
    parser.add_argument("--scheme", type=str, default="short",
                        choices=["all", "short", "long"],
                        help="all=mixed, short=MSRVTT+MSVD, long=NExT-QA")
    parser.add_argument("--use_msrvtt", action="store_true", default=False,
                        help="Include MSRVTT-QA dataset")
    parser.add_argument("--use_msvd", action="store_true", default=True,
                        help="Include MSVD-QA dataset")
    parser.add_argument("--use_nextqa", action="store_true", default=False,
                        help="Include NExT-QA dataset")

    args = parser.parse_args()

    # ---- Determine is_simple based on scheme ----
    is_simple = (args.scheme == "short")

    # ---- Dataset ----
    train_dataset = get_combined_dataset(
        num_frames=args.num_frames, h=288, w=288,
        partition="train",
        use_nextqa=args.use_nextqa,
        use_msrvtt=args.use_msrvtt,
        use_msvd=args.use_msvd,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_dim = 768
    action_dim = 1
    qwen_model, qwen_processor = load_qwen_model(args.qwen_model_name)

    model = VisionGRPOPolicy(
        encoder_name=args.encoder_name,
        feat_dim=feat_dim,
        action_dim=action_dim,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
    ).to(device)

    if args.load_from is not None:
        trainable_state = torch.load(args.load_from, map_location="cpu")
        model.load_state_dict(trainable_state, strict=False)
        print(f"Successfully loaded from {args.load_from}")

    dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        collate_fn=collate_fn, num_workers=0, shuffle=True,
    )
    eval_loader = get_eval_dataset(args.num_frames)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = sft_loss if args.loss_version == "v1" else sft_loss_v2

    for epoch in range(args.num_epochs):
        running = 0
        count = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for i, batch in enumerate(pbar):
            videos = batch["videos"].to(device)
            prompts = batch["question"]

            # 1. Forward pass
            out = model.forward(videos)
            logits = out["logits"]
            keep_prob = out["keep_prob"]

            # 2. Greedy top-k selection
            topk_idx = keep_prob.topk(args.top_k, dim=1).indices

            # 3. Run Qwen on selected frames
            rewards_batch = []
            for b in range(videos.size(0)):
                idx = topk_idx[b].sort().values
                frames = videos[b, idx].cpu()
                prompt = prompts[b]
                gt = batch["answer"][b]

                with torch.no_grad():
                    pred = qwen_answer_question(
                        prompt, frames, model.qwen, model.qwen_processor,
                        "msqa" if is_simple else "general"
                    )

                r = string_f1(pred, gt, is_simple)
                rewards_batch.append(r)

            # 4. SFT loss
            loss = loss_fn(logits, rewards_batch, args.top_k)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running += loss.item()
            count += 1
            avg_reward = sum(rewards_batch) / len(rewards_batch)

            pbar.set_postfix({
                "loss": f"{running / count:.4f}",
                "reward": f"{avg_reward:.3f}",
            })

            if i % args.save_eval_freq == 0 and i != 0:
                print(f"\n[Step {i}] loss={loss.item():.4f} "
                      f"avg_reward={avg_reward:.3f}")
                print(f"  prompts: {prompts[:2]}")
                print(f"  answers: {batch['answer'][:2]}")
                print(f"  rewards: {rewards_batch[:4]}")

                os.makedirs(args.save_loc, exist_ok=True)
                save_trainable(
                    model,
                    os.path.join(args.save_loc, f"checkpoint-{epoch}.{i}.pt"),
                )
                model.eval()
                eval_model(
                    model, eval_loader, args.top_k,
                    model.qwen, model.qwen_processor,
                )
                model.policy.train()
                model.policy_head.train()

            if i > args.max_iter:
                break