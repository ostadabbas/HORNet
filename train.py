from trl import GRPOTrainer, GRPOConfig
from dataset import get_combined_dataset, collate_fn
from model import VisionGRPOPolicy, VisionGRPOConfig
from reward import compute_hornet_rewards, grpo_loss_mcq
from util import *
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
from evaluate import eval_model, get_eval_dataset
import argparse

def compute_rewards(qwen_outputs, batch, is_simple) -> list[float]:
    """
    model_outputs:
        {
            "logits": [B, T, A],
            "selected_videos": list of tensors,
            "selected_indices": list of tensors,
            "qwen_outputs": list[str]
        }
    batch:
        {
            "videos": [B, T, H, W, 3],
            "ground_truth": list[str]
        }
    """
    ground_truths = batch['answer'] if is_simple else batch['gt_choice']
    rewards = []
    for idx, ground_truth in enumerate(ground_truths): #[b]
        # print(qwen_outputs[idx], ground_truth)
        reward = compute_hornet_rewards(qwen_outputs[idx], ground_truth, is_simple)
        rewards.append(reward)
        # print(reward)
    return rewards

def grpo_loss(logits, actions, rewards):
    """
    logits:   [B, T] keep logits (before sigmoid)
    actions:  [B, K, T] sampled 0/1 masks
    rewards:  [B, K] reward per candidate
    """
    B, K, T = actions.shape

    # Expand logits to match actions: [B, 1, T] -> [B, K, T]
    keep_logits = logits[..., 0].unsqueeze(1).expand(-1, K, -1)

    # log(sigmoid(l)) and log(1 - sigmoid(l))
    log_prob_keep = -F.softplus(-keep_logits)                 # log(sigmoid)
    log_prob_drop = -keep_logits - F.softplus(-keep_logits)   # log(1 - sigmoid)

    # log-prob per frame per candidate
    log_probs = actions * log_prob_keep + (1 - actions) * log_prob_drop   # [B, K, T]

    # average over frames
    log_probs = log_probs.mean(dim=2)   # [B, K]
    rewards = torch.tensor(rewards).to(log_probs.device)
    # print(rewards)

    # ---- Advantage normalization ---
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    all_equal = (advantages.abs().sum(dim=1, keepdim=True) < 1e-6).float()
    advantages = advantages + all_equal * (-0.1)

    # GRPO objective
    loss = -(log_probs * advantages).mean()
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GRPO Frame Selection Training")

    parser.add_argument("--num_frames", type=int, default=16,
                        help="Total number of frames per video")

    parser.add_argument("--encoder_name", type=str,
                        # default="videoprism_public_v1_base",
                        default=None,
                        help="VideoPrism encoder model name")

    parser.add_argument("--qwen_model_name", type=str,
                        default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Qwen-VL model name")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")

    parser.add_argument("--save_eval_freq", type=int, default=2500,
                        help="Steps between saving/evaluating checkpoints")

    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")

    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of candidate frame selections per video")

    parser.add_argument("--top_k", type=int, default=8,
                        help="Number of frames to select (default: num_frames // 2)")

    parser.add_argument("--save_loc", type=str, default="./checkpoints_3dataset",
                        help="Directory to save checkpoints")

    parser.add_argument("--load_from", type=str, default=None,
                        help="Path to load checkpoint from")

    parser.add_argument("--max_iter", type=int, default=1500,
                        help="max iteration")

    parser.add_argument("--scheme", type=str, default="all", help="all (mixed), short, long")

    args = parser.parse_args()

    # num_frames = 16
    # encoder_name = 'videoprism_public_v1_base'
    # qwen_model_name = "Qwen/Qwen3-VL-2B-Instruct"
    # batch_size = 8
    # save_eval_freq = 2500
    # num_epochs = 10
    # n_samples = 8
    # top_k = num_frames // 2
    # save_loc = r'./checkpoints_3dataset'
    # load_from = None

    # ---- Dataset ----
    if args.scheme == "all":
        train_dataset = get_combined_dataset(
            num_frames=args.num_frames,
            h=288,
            w=288,
            partition="train",
            use_nextqa=True,
            use_ms=True,
        )
    elif args.scheme == "short":
        train_dataset = get_combined_dataset(
            num_frames=args.num_frames,
            h=288,
            w=288,
            partition="train",
            use_nextqa=False,
            use_ms=True,
        )
    else:
        train_dataset = get_combined_dataset(
            num_frames=args.num_frames,
            h=288,
            w=288,
            partition="train",
            use_nextqa=True,
            use_ms=False,
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat_dim = 768
    action_dim = 1      # your discrete action space
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

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            collate_fn=collate_fn, num_workers=0, shuffle=True)
    eval_loader = get_eval_dataset(args.num_frames)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Lower learning rate — change optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # was 1e-4

    for epoch in range(args.num_epochs):
        running = 0
        count = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(pbar):
            videos = batch["videos"].to(device)
            prompts = batch["question"]
            choices = batch["choices"]

            # 1. Sample actions
            rollout = model.generate(videos, text_prompt=prompts, top_k=args.top_k, n_samples=args.n_samples, scheme=args.scheme, choices=choices)
            actions = rollout["actions"]                # [B, T]
            qwen_outputs = rollout["qwen_outputs"]      # list[str]
            selected_indices = rollout["selected_indices"]

            # 3. Compute rewards using Qwen outputs
            rewards = compute_rewards(qwen_outputs, batch, True if args.scheme == "short" else False)

            # 4. Compute GRPO loss (simple REINFORCE)
            out = model.forward(videos)
            logits = out["logits"]

            loss = grpo_loss_mcq(logits, actions, rewards, kl_logits=None, kl_coef=0.0, eps=1e-8)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
            count += 1

            pbar.set_postfix({
                "loss": f"{running / count:.4f}"
            })

            if i % args.save_eval_freq == 0 and i != 0:
                print(prompts)
                print(rollout['qwen_outputs'])
                print(batch['answer'])
                print(rewards)
                print(f"loss={loss.item():.4f}")
                os.makedirs(args.save_loc, exist_ok=True)
                save_trainable(model, os.path.join(args.save_loc, f"checkpoint-{epoch}.{i}.pt"))
                model.eval()
                eval_model(model, eval_loader, args.top_k, model.qwen, model.qwen_processor)
                model.policy.train()
                model.policy_head.train()
                model.encoder.train()
            
            if i > args.max_iter:
                break