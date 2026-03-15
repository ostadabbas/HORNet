"""
PPO training for frame selection.

Minimal changes from GRPO:
  1. Add a small ValueHead as baseline (instead of mean-reward baseline)
  2. Use clipped surrogate objective (instead of raw REINFORCE)
  3. Reuse rollout data for multiple update passes

Everything else (dataset, model, eval, sampling) is identical to train.py.
"""

from dataset import get_combined_dataset, collate_fn
from model import VisionGRPOPolicy
from reward import compute_hornet_rewards, string_f1
from util import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from evaluate_mcq import eval_model, get_eval_dataset
import argparse


# ---- Only new component vs GRPO ----
class ValueHead(nn.Module):
    """Predicts expected reward from policy MLP features."""
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 1),
        )

    def forward(self, h):
        # h: [B, T, 256] from policy MLP
        return self.net(h.mean(dim=1)).squeeze(-1)  # [B]


def compute_log_probs(keep_logits, actions):
    """Bernoulli log-prob: [B, T] logits + [B, T] binary → [B] scalar."""
    lp_keep = -F.softplus(-keep_logits)
    lp_drop = -keep_logits - F.softplus(-keep_logits)
    return (actions * lp_keep + (1 - actions) * lp_drop).sum(dim=1)


def ppo_loss(curr_logits, actions, advantages, old_log_probs, clip_eps=0.2):
    """Clipped surrogate objective."""
    new_lp = compute_log_probs(curr_logits, actions)
    ratio = torch.exp(new_lp - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def compute_rewards(qwen_outputs, batch, is_simple):
    """Compute rewards with is_simple flag."""
    ground_truths = batch['answer'] if is_simple else batch['gt_choice']
    rewards = []
    for idx, gt in enumerate(ground_truths):
        reward = compute_hornet_rewards(qwen_outputs[idx], gt, is_simple)
        rewards.append(reward)
    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PPO Frame Selection Training")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="Encoder name (None = TimeSformerTiny)")
    parser.add_argument("--qwen_model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_eval_freq", type=int, default=2500)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=3,
                        help="Optimization passes per rollout batch")
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save_loc", type=str, default="./checkpoints_ppo")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--max_iter", type=int, default=99999,
                        help="Max training iterations")
    # ---- Dataset selection flags (match train.py) ----
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

    # ---- Setup ----
    train_dataset = get_combined_dataset(
        num_frames=args.num_frames, h=288, w=288,
        partition="train",
        use_nextqa=args.use_nextqa,
        use_msrvtt=args.use_msrvtt,
        use_msvd=args.use_msvd,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_dim = 768
    qwen_model, qwen_processor = load_qwen_model(args.qwen_model_name)
    model = VisionGRPOPolicy(
        encoder_name=args.encoder_name, feat_dim=feat_dim, action_dim=1,
        qwen_model=qwen_model, qwen_processor=qwen_processor,
    ).to(device)

    value_head = ValueHead(dim=256).to(device)

    if args.load_from is not None:
        state = torch.load(args.load_from, map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"Loaded from {args.load_from}")

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            collate_fn=collate_fn, num_workers=0, shuffle=True)
    eval_loader = get_eval_dataset(args.num_frames)

    optimizer = torch.optim.Adam([
        {"params": list(model.policy.parameters()) + list(model.policy_head.parameters()),
         "lr": args.lr},
        {"params": value_head.parameters(), "lr": args.lr * 3},
    ])

    for epoch in range(args.num_epochs):
        running_loss, running_reward, count = 0, 0, 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for i, batch in enumerate(pbar):
            videos = batch["videos"].to(device)
            prompts = batch["question"]
            choices = batch["choices"]
            B = videos.size(0)

            # ==== Phase 1: Collect rollouts ====
            rollout = model.generate(
                videos, text_prompt=prompts,
                top_k=args.top_k, n_samples=args.n_samples,
                scheme=args.scheme, choices=choices,
            )
            actions = rollout["actions"]
            qwen_outputs = rollout["qwen_outputs"]
            rewards = compute_rewards(qwen_outputs, batch, is_simple)

            # Pre-compute old log-probs (detached)
            with torch.no_grad():
                out = model.forward(videos)
                old_logits = out["logits"].squeeze(-1)

            K = actions.shape[1]
            old_lps = []
            for k in range(K):
                old_lps.append(compute_log_probs(old_logits, actions[:, k]))
            old_lps = torch.stack(old_lps, dim=1)

            rewards_t = torch.tensor(rewards, device=device).float()
            avg_reward = rewards_t.mean().item()

            # ==== Phase 2: PPO updates ====
            for _ in range(args.ppo_epochs):
                out = model.forward(videos)
                curr_logits = out["logits"].squeeze(-1)

                # Value baseline
                feats = model.encoder(videos)
                h = model.policy(feats)
                values = value_head(h)

                total_loss = torch.tensor(0.0, device=device)

                for k in range(K):
                    adv = rewards_t[:, k] - values.detach()
                    if adv.std() > 1e-6:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    pl = ppo_loss(curr_logits, actions[:, k], adv,
                                  old_lps[:, k], clip_eps=args.clip_eps)
                    vl = F.mse_loss(values, rewards_t[:, k])

                    total_loss += pl + args.vf_coef * vl

                total_loss /= K

                # Entropy bonus
                p = torch.sigmoid(curr_logits)
                ent = -(p * (p + 1e-8).log() + (1-p) * (1-p + 1e-8).log()).mean()
                total_loss -= args.ent_coef * ent

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(value_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

            running_loss += total_loss.item()
            running_reward += avg_reward
            count += 1

            pbar.set_postfix({
                "loss": f"{running_loss / count:.4f}",
                "reward": f"{running_reward / count:.3f}",
            })

            if i % args.save_eval_freq == 0 and i != 0:
                print(prompts)
                print(rollout['qwen_outputs'])
                print(batch['answer'])
                print(rewards)
                print(f"loss={total_loss.item():.4f}")

                os.makedirs(args.save_loc, exist_ok=True)
                save_trainable(model, os.path.join(args.save_loc, f"policy-{epoch}.{i}.pt"))
                torch.save(value_head.state_dict(),
                           os.path.join(args.save_loc, f"value-{epoch}.{i}.pt"))

                model.eval()
                eval_model(model, eval_loader, args.top_k, model.qwen, model.qwen_processor)
                model.policy.train()
                model.policy_head.train()
                value_head.train()

            if i > args.max_iter:
                break