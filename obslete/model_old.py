import torch
import torch.nn as nn
import torch.nn.functional as F

from vp import TorchVideoPrism, FrozenVideoPrismEncoder
from transformers import PreTrainedModel, PretrainedConfig

class VisionGRPOConfig(PretrainedConfig):
    model_type = "vision_grpo"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name_or_path = "Qwen/Qwen3-VL-8B-Instruct"

class MLPPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        x = x.reshape(B * T, D)
        h = self.mlp(x)              # [B*T, 256]
        return h.reshape(B, T, -1)   # [B, T, 256]


class VisionGRPOPolicy(PreTrainedModel):
    """
    GRPO-ready model:
      - Frozen VideoPrism encoder
      - Trainable MLP per frame
      - Policy head outputs logits per frame
      - Trainable K via Bernoulli sampling
      - Qwen3-VL evaluates selected frames
    """
    config_class = VisionGRPOConfig

    def __init__(self, config, encoder_name, feat_dim, action_dim, qwen_model, qwen_processor):
        super().__init__(config)

        # ---- Frozen VideoPrism ----
        jax_encoder = FrozenVideoPrismEncoder(model_name=encoder_name)
        self.encoder = TorchVideoPrism(jax_encoder)

        # ---- Trainable MLP ----
        self.policy = MLPPolicy(feat_dim)
        self.policy_head = nn.Linear(256, action_dim)

        # ---- Frozen Qwen3-VL ----
        self.qwen = qwen_model.eval()
        for p in self.qwen.parameters():
            p.requires_grad = False
        self.qwen_processor = qwen_processor

    def forward(self, videos, text_prompt=None, **kwargs):
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

        # ---- 2. Trainable MLP per frame ----
        h = self.policy(feats)              # [B, T, 256]

        # ---- 3. Frame-level logits ----
        logits = self.policy_head(h)        # [B, T, action_dim]

        # ---- 4. Convert logits → keep probabilities ----
        # action_dim must be >= 1; we use logits[..., 0]
        keep_logits = logits[..., 0]        # [B, T]
        keep_prob = torch.sigmoid(keep_logits)  # [B, T]

        # ---- 5. Sample a Bernoulli mask (trainable K) ----
        # During training: stochastic
        mask = torch.bernoulli(keep_prob)   # [B, T]

        # Ensure at least 1 frame is kept
        mask = torch.where(mask.sum(dim=1, keepdim=True) == 0,
                           torch.ones_like(mask), mask)

        # ---- 6. Select frames ----
        selected_videos = []
        selected_indices = []

        for b in range(B):
            idx = torch.nonzero(mask[b]).squeeze(-1)  # variable length
            selected_indices.append(idx)

            frames = videos[b, idx]  # [K*, H, W, 3]
            selected_videos.append(frames)

        # ---- 7. Run Qwen3-VL on each selected subset ----
        qwen_outputs = []
        for b in range(B):
            frames = selected_videos[b]
            prompt = text_prompt[b] if text_prompt else ""

            inputs = self.qwen_processor(images=list(frames),
                                         text=prompt,
                                         return_tensors="pt")
            inputs = {k: v.to(self.qwen.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.qwen.generate(**inputs, max_new_tokens=64)

            decoded = self.qwen_processor.decode(output[0])
            qwen_outputs.append(decoded)

        return {
            "logits": logits,                     # [B, T, action_dim]
            "keep_prob": keep_prob,               # [B, T]
            "selected_videos": selected_videos,   # list of [K*, H, W, 3]
            "selected_indices": selected_indices, # list of [K*]
            "qwen_outputs": qwen_outputs,         # list[str]
        }

    @torch.no_grad()
    def generate(self, videos, temperature=1.0, **kwargs):
        """
        GRPOTrainer calls this to sample actions.
        Returns: [B, T] integer actions
        """
        out = self.forward(videos)
        logits = out["logits"] / temperature   # [B, T, A]
        probs = F.softmax(logits, dim=-1)

        B, T, A = probs.shape
        actions = torch.multinomial(
            probs.reshape(B * T, A),
            num_samples=1
        ).reshape(B, T)

        return actions