import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import random
import decord
import numpy as np
import time

def load_qwen_model(model_id: str, use_lora: bool = False):
    """Load model and processor, with optional LoRA weights."""
    print(f"\n[1/3] Loading processor from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("✓ Processor loaded")
    
    print(f"\n[2/3] Loading model from {model_id}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model.eval()
    print("✓ Model loaded and set to eval mode")
    
    return model, processor

def fit_video_for_qwen(frames):
    cpu_images = []
    for f in frames:  # f: [H, W, 3] float32 tensor
        img = f.detach().cpu()

        # If your frames are in [0,1], convert to [0,255]
        if img.max() <= 1.0:
            img = (img * 255)

        img = img.clamp(0, 255).byte()  # uint8
        cpu_images.append(Image.fromarray(img.numpy()))
    return cpu_images

def tokenize_qwen_images(processor, images):
    """
    Convert a list of images into Qwen-ready pixel values.

    Args:
        processor: Qwen processor (AutoProcessor.from_pretrained(...))
        images: list of images, each can be:
                - torch.Tensor [H, W, 3] uint8 or float
                - numpy array [H, W, 3]
                - PIL.Image

    Returns:
        pixel_values: torch.FloatTensor of shape [B, C, H, W]
    """
    # Qwen requires CPU uint8 or PIL images
    clean_images = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()

            # Ensure shape [H, W, 3]
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.permute(1, 2, 0)  # convert [3,H,W] → [H,W,3]

            if img.max() <= 1.0:
                img = img * 255

            img = img.clamp(0, 255).byte().numpy()

        clean_images.append(img)

    # Use Qwen's image processor
    pixel_values = processor.image_processor(
        images=clean_images,
        return_tensors="pt"
    )["pixel_values"]

    return pixel_values

def qwen_answer_question(prompt, frames, qwen, qwen_processor):
    # print(prompt)
    system_prompt = "Based on the video, please answer the question in only one word and end with a period ('.'). "
    cpu_images = fit_video_for_qwen(frames)
    # 2. Use the processor correctly (removed the redundant tokenize_qwen_images call)
    messages = [
        {
            "role": "system",
            "content":[
                {"type":"text", "text":system_prompt},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "video", "video": cpu_images},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # This 'text' now contains all the necessary <|vision_start|> tags automatically.
    inputs = qwen_processor(text=text, videos=cpu_images, return_tensors="pt")
    inputs = {k: v.to(qwen.device) for k, v in inputs.items()}
    # 3. Pass the unpacked dictionary to generate
    # The ** operator expands the dict into keyword arguments (input_ids=..., pixel_values=...)
    output = qwen.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False  # Greedy decoding for more stable results
    )
    # 4. Decode the result
    # Clean up the output by skipping the prompt and special tokens
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs["input_ids"], output)
    ]
    response = qwen_processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    # 4. Generate with clean output
    # output = self.qwen.generate(**inputs, max_new_tokens=64)
    # 5. Decode while skipping special tokens (crucial for readable text)
    # decoded_text = qwen_processor.decode(output[0], skip_special_tokens=True)
    return response[0].split(".")[0]

def save_trainable(model, path="trainable.pt"):
    trainable_state = {
        name: param
        for name, param in model.state_dict().items()
        if model.get_parameter(name).requires_grad
    }
    torch.save(trainable_state, path)

def get_action_list(keep_prob, n_samples, temps, top_k):
    actions_list = []

    # ---- 1. Pure Bernoulli ----
    bern = torch.bernoulli(keep_prob)
    bern = torch.where(bern.sum(dim=1, keepdim=True) == 0,
                        torch.ones_like(bern), bern)
    actions_list.append(bern)

    # ---- 2. Pure top-k ----
    topk_idx = keep_prob.topk(top_k, dim=1).indices  # [B, K]
    topk_mask = torch.zeros_like(keep_prob)
    topk_mask.scatter_(1, topk_idx, 1.0)
    actions_list.append(topk_mask)

    # Use temperature scaling or Gumbel noise
    for i in range(n_samples - 2):
        temp = temps[i % len(temps)]

        # Temperature-scaled Bernoulli
        scaled_prob = torch.clamp(keep_prob / temp, 0, 1)
        sample = torch.bernoulli(scaled_prob)

        # Ensure at least one frame
        sample = torch.where(
            sample.sum(dim=1, keepdim=True) == 0,
            torch.ones_like(sample),
            sample
        )

        actions_list.append(sample)

    # Stack into [B, N, T]
    actions_all = torch.stack(actions_list, dim=1)
    return actions_all

def get_action_by_k(keep_prob, n_samples, top_k, random_sample=True):
    actions_list = []

    # ---- 1. Pure Bernoulli ----
    bern = torch.bernoulli(keep_prob)
    bern = torch.where(bern.sum(dim=1, keepdim=True) == 0,
                        torch.ones_like(bern), bern)
    actions_list.append(bern)

    # ---- 2. Pure top-k ----
    step = top_k // n_samples
    for keep_k in range(top_k, 0, -step):
        topk_idx = keep_prob.topk(keep_k, dim=1).indices  # [B, K]
        topk_mask = torch.zeros_like(keep_prob)
        topk_mask.scatter_(1, topk_idx, 1.0)
        actions_list.append(topk_mask)

    if random.random() < 0.3 and random_sample:
        actions_list = actions_list[:-1]
    else:
        actions_list = actions_list[1:]
    actions_all = torch.stack(actions_list, dim=1)
    return actions_all

device = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_frames(video_path, sample=True):
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if sample:
        if total_frames >= 32:
            indices = np.linspace(0, total_frames - 1, 32).astype(int)
        else:
            indices = list(range(total_frames)) + [total_frames - 1] * (32 - total_frames)
            indices = np.array(indices)
    else:
        indices = list(range(total_frames))
    frames = vr.get_batch(indices)
    frames = torch.from_numpy(frames.asnumpy()).float()
    frames = torch.nn.functional.interpolate(
        frames.permute(0, 3, 1, 2),
        size=(288, 288),
        mode="bilinear",
        align_corners=False
    ).permute(0, 2, 3, 1)
    return frames.float() / 255.0, total_frames

def select_frames(url, model, top_k=8):
    time_a = time.time()
    videos, total_frames = load_frames(url)
    videos = videos.to(device)
    out = model(videos.unsqueeze(0))
    sel_time = time.time() - time_a
    keep_prob = out['keep_prob']
    actions_all = get_action_by_k(keep_prob, 1, top_k, random_sample=False)
    idx = torch.nonzero(actions_all[0][0]).squeeze(-1)
    if len(videos.shape) == 5:
        frames = videos.to("cpu")[0, idx.to("cpu")]
    else:
        frames = videos.to("cpu")[idx.to("cpu")]
    return fit_video_for_qwen(frames), sel_time, total_frames