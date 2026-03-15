"""Quick smoke test for PPO training — runs 2 steps then exits."""

from dataset import get_combined_dataset, collate_fn
from model import VisionGRPOPolicy
from util import *
from train_ppo import ValueHead, compute_log_probs, ppo_loss
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    num_frames = 16
    top_k = 8
    n_rollouts = 2  # small for test
    ppo_epochs = 2
    batch_size = 2

    # ---- Dataset ----
    print("\n--- Loading dataset ---")
    ds = get_combined_dataset(num_frames=num_frames, h=288, w=288,
                              partition="train", use_nextqa=True)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn,
                        num_workers=0, shuffle=True)

    # ---- Model ----
    print("\n--- Loading model ---")
    qwen_model, qwen_processor = load_qwen_model("Qwen/Qwen3-VL-2B-Instruct")
    model = VisionGRPOPolicy(
        encoder_name="videoprism_public_v1_base",
        feat_dim=768, action_dim=1,
        qwen_model=qwen_model, qwen_processor=qwen_processor,
    ).to(device)

    value_head = ValueHead(dim=256).to(device)

    policy_params = list(model.policy.parameters()) + list(model.policy_head.parameters())
    optimizer = torch.optim.Adam([
        {"params": policy_params, "lr": 3e-5},
        {"params": value_head.parameters(), "lr": 1e-4},
    ])

    # ---- Run 2 steps ----
    print("\n--- Running 2 PPO steps ---")
    for step, batch in enumerate(loader):
        if step >= 2:
            break

        videos = batch["videos"].to(device)
        prompts = batch["question"]
        answers = batch["answer"]
        B = videos.size(0)

        print(f"\nStep {step}")
        print(f"  videos shape: {videos.shape}")
        print(f"  prompts: {prompts}")

        # Phase 1: Rollouts
        with torch.no_grad():
            out = model.forward(videos)
            keep_logits = out["logits"].squeeze(-1)
            keep_prob = out["keep_prob"]

        print(f"  keep_prob range: [{keep_prob.min():.3f}, {keep_prob.max():.3f}]")

        # Use model.generate() just like train_ppo.py does
        rollout = model.generate(videos, text_prompt=prompts,
                                 top_k=top_k, n_samples=n_rollouts)
        actions = rollout["actions"]          # [B, K, T]
        qwen_outputs = rollout["qwen_outputs"]

        # Compute rewards
        from reward import compute_hornet_rewards
        all_rewards = []
        for b, gt in enumerate(answers):
            all_rewards.append(compute_hornet_rewards(qwen_outputs[b], gt))
        rewards_t = torch.tensor(all_rewards, device=device).float()  # [B, K]

        K = actions.shape[1]
        old_lps = []
        for k in range(K):
            old_lps.append(compute_log_probs(keep_logits, actions[:, k]))
        old_lps = torch.stack(old_lps, dim=1)  # [B, K]

        print(f"  actions shape: {actions.shape}")
        print(f"  rewards: {rewards_t}")
        print(f"  avg reward: {rewards_t.mean():.3f}")

        # Phase 2: PPO updates
        for ppo_ep in range(ppo_epochs):
            out = model.forward(videos)
            curr_logits = out["logits"].squeeze(-1)

            feats = model.encoder(videos)
            feats = feats.reshape(B, num_frames, 16, 16, -1).mean(dim=(2, 3))
            h = model.policy(feats)
            values = value_head(h)

            print(f"    PPO epoch {ppo_ep}: values={values.detach().cpu().tolist()}")

            total_loss = torch.tensor(0.0, device=device)
            for k in range(K):
                adv = rewards_t[:, k] - values.detach()
                if adv.std() > 1e-6:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pl = ppo_loss(curr_logits, actions[:, k], adv,
                              old_lps[:, k], clip_eps=0.2)
                vl = F.mse_loss(values, rewards_t[:, k])
                total_loss += pl + 0.5 * vl

            total_loss /= K

            # Entropy
            p = torch.sigmoid(curr_logits)
            ent = -(p * (p + 1e-8).log() + (1-p) * (1-p + 1e-8).log()).mean()
            total_loss -= 0.01 * ent

            print(f"    total_loss={total_loss.item():.4f} entropy={ent.item():.4f}")

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(value_head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            print(f"    ✓ backward + step done")

    print("\n=== PPO smoke test passed! ===")


if __name__ == "__main__":
    main()