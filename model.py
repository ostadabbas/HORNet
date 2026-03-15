import torch
import torch.nn as nn
import torch.nn.functional as F

from vp import TorchVideoPrism, FrozenVideoPrismEncoder
from transformers import PreTrainedModel, PretrainedConfig
from util import *

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.proj(x)  # [B*T, D, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B*T, N, D]
        return x.reshape(B, T, -1, x.shape[-1])  # [B, T, N, D]


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
        )

    def forward(self, x):
        # x: [B, L, D]
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class TimeSformerTiny(nn.Module):
    """
    Input:  [B, T, H, W, 3]
    Output: [B, T, D]
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, patch_size=16):
        super().__init__()
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )

        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

    def forward(self, videos):
        # videos: [B, T, H, W, 3]
        B, T, H, W, C = videos.shape
        x = videos.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]

        # Patchify
        x = self.patch_embed(x)  # [B, T, N, D]
        B, T, N, D = x.shape

        # Spatial attention (per frame)
        x = x.reshape(B*T, N, D)
        for blk in self.spatial_blocks:
            x = blk(x)
        x = x.reshape(B, T, N, D)

        # Temporal attention (per patch index)
        x = x.permute(0, 2, 1, 3)  # [B, N, T, D]
        x = x.reshape(B*N, T, D)
        for blk in self.temporal_blocks:
            x = blk(x)
        x = x.reshape(B, N, T, D).permute(0, 2, 1, 3)  # [B, T, N, D]

        # Pool patches → per-frame feature
        x = x.mean(dim=2)  # [B, T, D]
        return x

class VisionGRPOConfig(PretrainedConfig):
    model_type = "vision_grpo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name_or_path = "Qwen/Qwen3-VL-8B-Instruct"

class MLPPolicy(nn.Module):
    def __init__(self, patch_dim, input_dim=512, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.proj = nn.Linear(patch_dim, input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x = self.proj(x)
        x = x.reshape(B * T, -1)
        h = self.mlp(x)              # [B*T, 256]
        return h.reshape(B, T, -1)   # [B, T, 256]


class VisionGRPOPolicy(nn.Module):
    """
    GRPO-ready model:
      - Frozen VideoPrism encoder
      - Trainable MLP per frame
      - Policy head outputs logits per frame
      - Trainable K via Bernoulli sampling
      - Qwen3-VL evaluates selected frames
    """

    def __init__(self, encoder_name, feat_dim, action_dim, qwen_model, qwen_processor):
        super().__init__()

        # ---- Frozen VideoPrism ----
        if encoder_name:
            jax_encoder = FrozenVideoPrismEncoder(model_name=encoder_name)
            self.encoder = TorchVideoPrism(jax_encoder)
        else:
            self.encoder = TimeSformerTiny(
                embed_dim=feat_dim,
                depth=4,
                num_heads=4,
                patch_size=16
            )

        # ---- Trainable MLP ----
        self.policy = MLPPolicy(feat_dim)
        self.policy_head = nn.Linear(256, action_dim)

        # ---- Frozen Qwen3-VL ----
        self.qwen = qwen_model.eval()
        for p in self.qwen.parameters():
            p.requires_grad = False
        self.qwen_processor = qwen_processor

    def forward(self, videos, **kwargs):
        """
        videos: [B, T, H, W, 3]
        returns:
            logits: [B, T, action_dim]
            selected_videos: [B, K*, H, W, 3]   (K* varies per video)
            selected_indices: list of tensors
            qwen_outputs: list[str]
        """
        B, T, H, W, C = videos.shape

        # ---- 1. Encode video → frame features ----
        feats = self.encoder(videos)        # [B, T, D]
        # feats = feats.reshape(B, T, 16, 16, -1)
        # feats = feats.mean(dim=(2, 3))
        # ---- 2. Trainable MLP per frame ----
        h = self.policy(feats)              # [B, T, 256]

        # ---- 3. Frame-level logits ----
        logits = self.policy_head(h)        # [B, T, action_dim]

        # ---- 4. Convert logits → keep probabilities ----
        # action_dim must be >= 1; we use logits[..., 0]
        keep_logits = logits[..., 0]        # [B, T]
        keep_prob = torch.sigmoid(keep_logits)  # [B, T]

        return {
            "logits": logits,                     # [B, T, action_dim]
            "keep_prob": keep_prob,               # [B, T]
        }

    @torch.no_grad()
    def generate(self, videos, text_prompt, temperature=1.0, top_k=4, n_samples=4, scheme="general", choices=None):
        out = self.forward(videos)
        keep_prob = out["keep_prob"] / temperature
        actions_all = get_action_by_k(keep_prob, n_samples, top_k)
        batch_selected_videos = []
        ind_all = []
        qwen_outputs = []
        assert top_k >= n_samples
        # select frames
        for b in range(videos.size(0)):
            selected_indices = []
            selected_videos = []
            for action_idx in range(n_samples): # or n+1 here
                idx = torch.nonzero(actions_all[b][action_idx]).squeeze(-1)
                selected_indices.append(idx)
                selected_videos.append(videos.to("cpu")[b, idx.to("cpu")])
                # print(videos.to("cpu")[b, idx.to("cpu")].shape)
            batch_selected_videos.append(selected_videos)
            ind_all.append(selected_indices)

        # run Qwen
        if scheme == "long":
            qwen_strat = "choice" 
        elif scheme == "short":
            qwen_strat = "msqa"
        else:
            qwen_strat = "general"
        for b, frames_n_turns in enumerate(batch_selected_videos):
            prompt = text_prompt[b] if text_prompt else "SAY SOMETHING WRONG"
            if qwen_strat == "choice":
                prompt = prompt + "\n Your choices are: " + ", ".join(choices[b])
            qb = []
            for frames in frames_n_turns:
                qwen_resp = qwen_answer_question(prompt, frames, self.qwen, self.qwen_processor, qwen_strat)
                qb.append(qwen_resp.split(".")[0])
            qwen_outputs.append(qb)

        return {
            "actions": actions_all,
            "selected_indices": ind_all,
            "qwen_outputs": qwen_outputs,
        }