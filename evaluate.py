from dataset import get_combined_dataset, collate_fn
from torch.utils.data import DataLoader, Dataset
from model import VisionGRPOPolicy
from util import *
from tqdm import tqdm
from reward import string_f1
import time
import argparse

def get_eval_dataset(num_frames, dataset_name):
    g = torch.Generator().manual_seed(1234)
    eval_dataset = get_combined_dataset(
        num_frames=num_frames,
        h=288,
        w=288,
        partition="val",
        use_nextqa=True if dataset_name=="nextqa" else False,
        use_msrvtt=True if dataset_name=="msrvtt" else False,
        use_msvd=True if dataset_name=="msvd" else False,
    )
    dataloader = DataLoader(eval_dataset, batch_size=1,
                            collate_fn=collate_fn, num_workers=0, shuffle=True, generator=g)
    return dataloader

def eval_model(model, dataloader, top_k, qwen, qwen_processor, dataset_name, first=None):
    model.eval()
    device = "cuda"
    pbar = tqdm(dataloader)
    ours_score_all, ref_score_all = 0, 0
    ref_count = 0
    time_ours = 0
    time_ref = 0
    total_frames = 0

    qwen_task_dict = {
        "msvd": "msqa",
        "msrvtt": "msqa",
        "nextqa": "general", 
    }

    for i, batch in enumerate(pbar):
        if first is not None and i > first:
            break
        videos = batch["videos"].to(device)
        text_prompt = batch['question']
        time_a = time.time()
        out = model(videos)
        keep_prob = out['keep_prob']
        actions_all = get_action_by_k(keep_prob, 1, top_k, random_sample=False)
        idx = torch.nonzero(actions_all[0][0]).squeeze(-1)
        frames = videos.to("cpu")[0, idx.to("cpu")]
        time_b = time.time()
        prompt = text_prompt[0] if text_prompt else "SAY SOMETHING WRONG"
        gt = batch['answer'][0]

        qwen_resp_ours = qwen_answer_question(prompt, frames, qwen, qwen_processor, qwen_task_dict[dataset_name],)
        time_ours += (time_b - time_a)

        ours_score = string_f1(qwen_resp_ours, gt, is_simple=True)
        ours_score_all += ours_score

        # ref only available for MSRVTT (videos_full exists)
        if batch["videos_full"] is not None:
            full_vid = batch["videos_full"].to(device)[0]
            qwen_resp_ref = qwen_answer_question(
                prompt, full_vid, qwen, qwen_processor, qwen_task_dict[dataset_name],
            )
            time_c = time.time()
            time_ref += (time_c - time_b)
            ref_score = string_f1(qwen_resp_ref, gt, is_simple=True)
            ref_score_all += ref_score
            total_frames += full_vid.shape[0]
            ref_count += 1

        pbar.set_postfix({
            "ours": f"{ours_score_all / (i+1):.4f}",
            "ref":  f"{ref_score_all / ref_count:.4f}" if ref_count > 0 else "N/A",
        })

    print(f"Frame selection time: {time_ours/i:.2f}s  Ref time: {time_ref/ref_count:.2f}s")
    print(f"Ours frame: {top_k} Ref frame: {total_frames / ref_count:.4f}")
    print(f"Final ours: {ours_score_all / i:.4f}  "
          f"Final ref: {ref_score_all / ref_count:.4f}" if ref_count > 0
          else f"Final ours: {ours_score_all / i:.4f}")



        

if __name__ == "__main__":

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

    parser.add_argument("--top_k", type=int, default=8,
                        help="Number of frames to select (default: num_frames // 2)")

    parser.add_argument("--load_from", type=str, default=None,
                        help="Path to load checkpoint from")

    parser.add_argument("--dataset", type=str, default="msvd", help="msvd, msrvtt, nextqa")

    parser.add_argument("--first", type=int, default=None, help="how many to eval")

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_frames = args.num_frames
    # encoder_name = 'videoprism_public_v1_base'
    encoder_name = args.encoder_name
    qwen_model_name = args.qwen_model_name
    save_eval_freq = 2500
    num_epochs = 10
    n_samples = 4
    top_k = args.top_k
    feat_dim = 768
    action_dim = 1
    save_loc = r'./checkpoints'
    # load_from = r'checkpoints_3datasets_v4_short/checkpoint-0.550.pt'
    load_from = args.load_from
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
    dataloader = get_eval_dataset(num_frames, args.dataset)  # was get_eval_dataset()
    eval_model(model, dataloader, top_k, qwen_model, qwen_processor, args.dataset, args.first)