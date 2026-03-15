from dataset import get_combined_dataset, collate_fn
from torch.utils.data import DataLoader, Dataset
from model import VisionGRPOPolicy
from util import *
from tqdm import tqdm
from reward import string_f1
import time
import re

def get_eval_dataset(num_frames):
    g = torch.Generator().manual_seed(1234)
    eval_dataset = get_combined_dataset(
        num_frames=num_frames,
        h=288,
        w=288,
        partition="val",
        use_nextqa=True,
    )
    dataloader = DataLoader(eval_dataset, batch_size=1,
                            collate_fn=collate_fn, num_workers=0, shuffle=True, generator=g)
    return dataloader

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

def eval_model(model, dataloader, top_k, qwen, qwen_processor):
    model.eval()
    device = "cuda"
    pbar = tqdm(dataloader)
    ours_score_all, ref_score_all = 0, 0
    ref_count = 0
    time_ours = 0
    time_ref = 0

    j = 0
    for i, batch in enumerate(pbar):
        if j > 100:
            break
        if batch["choices"] is None:
            continue
        videos = batch["videos"].to(device)
        text_prompt = batch['question']
        out = model(videos)
        keep_prob = out['keep_prob']
        actions_all = get_action_by_k(keep_prob, 1, top_k, random_sample=False)
        # print(actions_all)
        idx = torch.nonzero(actions_all[0][0]).squeeze(-1)
        frames = videos.to("cpu")[0, idx.to("cpu")]
        # print(frames.shape)
        if text_prompt:
            prompt = text_prompt[0] + "\n Your choices are: " + ", ".join(batch["choices"][0])
        else:
            prompt = "SAY SOMETHING WRONG"
        gt = int(batch['gt_choice'][0])
        # print(prompt)

        time_a = time.time()
        qwen_resp_ours = qwen_answer_question(prompt, frames, qwen, qwen_processor, "choice")
        qwen_resp_ours = extract_one_digit_answer(qwen_resp_ours)
        # print(prompt, qwen_resp_ours, gt)
        time_b = time.time()
        time_ours += (time_b - time_a)

        ours_score_all += 1 if gt == qwen_resp_ours else 0

        # ref only available for MSRVTT (videos_full exists)
        if batch["videos_full"] is not None:
        # if False:
            qwen_resp_ref = qwen_answer_question(
                prompt, batch["videos_full"].to(device)[0], qwen, qwen_processor, "choice"
            )
            time_c = time.time()
            time_ref += (time_c - time_b)
            ref_score = 1 if extract_one_digit_answer(qwen_resp_ref) == gt else 0
            ref_score_all += ref_score
            ref_count += 1
        j += 1
        pbar.set_postfix({
            "ours": f"{ours_score_all / (j+1):.4f}",
            "ref":  f"{ref_score_all / ref_count:.4f}" if ref_count > 0 else "N/A",
        })

    print(f"Ours time: {time_ours:.2f}s  Ref time: {time_ref:.2f}s")
    print(f"Final ours: {ours_score_all / j:.4f}  "
          f"Final ref: {ref_score_all / ref_count:.4f}" if ref_count > 0
          else f"Final ours: {ours_score_all / j:.4f}")

        

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_frames = 32
    # encoder_name = 'videoprism_public_v1_base'
    encoder_name = None
    qwen_model_name = "Qwen/Qwen3-VL-2B-Instruct"
    save_eval_freq = 2500
    num_epochs = 10
    n_samples = 4
    top_k = 8
    feat_dim = 768
    action_dim = 1
    save_loc = r'./checkpoints'
    load_from = r'checkpoints_3datasets_v4_long/checkpoint-0.250.pt'
    # load_from = None
    qwen_model, qwen_processor = load_qwen_model(qwen_model_name)
    model = VisionGRPOPolicy(
        encoder_name=encoder_name,
        feat_dim=feat_dim,
        action_dim=action_dim,
        qwen_model=qwen_model, 
        qwen_processor= qwen_processor,
    ).to(device)
    if load_from is not None:
        trainable_state = torch.load(load_from, map_location="cpu")
        model.load_state_dict(trainable_state, strict=False)
        print(f"Successfully loaded from {load_from}")
    # dataloader = get_eval_dataset()
    dataloader = get_eval_dataset(num_frames)  # was get_eval_dataset()
    eval_model(model, dataloader, top_k, qwen_model, qwen_processor)