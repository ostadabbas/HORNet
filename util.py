from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import random
import re

def extract_one_digit_answer(text):
    '''
    Extracts the first one-digit number (0–9) from messy answer strings.
    Returns the digit as an int, or None if no digit is found.
    '''
    # Look for a single digit anywhere in the string
    match = re.search(r'\b(\d)\b', text)
    if match:
        return int(match.group(1))
    return None

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
        cpu_images.append(img)
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

def qwen_answer_question(prompt, frames, qwen, qwen_processor, task="general"):
    # print(prompt)
    if task == "msqa":
        system_prompt = "Based on the video, please answer the question in only one word and end with a period ('.'). "
    elif task == "general":
        system_prompt = "Based on the video, please give answer in a short sentence. Your first sentence should be the concise answer."
    elif task == "choice":
        system_prompt = "Based on the video, please answer the question by selecting the correct index, output only one digit. "
    cpu_images = fit_video_for_qwen(frames)
    # print(len(cpu_images))
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
    return response[0]

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
