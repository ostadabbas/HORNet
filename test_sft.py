"""Quick smoke test for SFT training — runs 2 steps then exits."""

from dataset import get_combined_dataset, collate_fn
from model import VisionGRPOPolicy
from reward import string_f1
from util import *
from train_sft import sft_loss, sft_loss_v2
from torch.utils.data import DataLoader
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    num_frames = 16
    top_k = 8
    batch_size = 2  # small for quick test

    # ---- Dataset (just grab a few samples) ----
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
        feat_dim=768,
        action_dim=1,
        qwen_model=qwen_model,
        qwen_processor=qwen_processor,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- Run 2 steps ----
    print("\n--- Running 2 SFT steps ---")
    for step, batch in enumerate(loader):
        if step >= 2:
            break

        videos = batch["videos"].to(device)
        prompts = batch["question"]
        answers = batch["answer"]

        print(f"\nStep {step}")
        print(f"  videos shape: {videos.shape}")
        print(f"  prompts: {prompts}")
        print(f"  answers: {answers}")

        # Forward
        out = model.forward(videos)
        logits = out["logits"]
        keep_prob = out["keep_prob"]
        print(f"  logits shape: {logits.shape}")
        print(f"  keep_prob: {keep_prob}")

        # Greedy top-k
        topk_idx = keep_prob.topk(top_k, dim=1).indices
        print(f"  selected frames: {topk_idx}")

        # Qwen inference
        rewards = []
        for b in range(videos.size(0)):
            idx = topk_idx[b].sort().values
            frames = videos[b, idx].cpu()
            with torch.no_grad():
                pred = qwen_answer_question(
                    prompts[b], frames, model.qwen, model.qwen_processor
                )
            r = string_f1(pred, answers[b])
            rewards.append(r)
            print(f"  [b={b}] pred='{pred}' gt='{answers[b]}' reward={r:.3f}")

        # Loss v1
        loss_v1 = sft_loss(logits, rewards, top_k)
        print(f"  sft_loss v1: {loss_v1.item():.4f}")

        # Loss v2
        loss_v2 = sft_loss_v2(logits, rewards, top_k)
        print(f"  sft_loss v2: {loss_v2.item():.4f}")

        # Backward + step (use v1)
        optimizer.zero_grad()
        loss_v1.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"  ✓ backward + optimizer step done")

    print("\n=== Smoke test passed! ===")


if __name__ == "__main__":
    main()