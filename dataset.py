from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import decord
import json
import numpy as np
import torch


class VideoQADataset(Dataset):
    def __init__(self, msqa_location, num_frames=64, h=288, w=288, partition="train",
                 video_prefix="video", video_ext=".mp4"):
        """
        msqa_location : root folder containing {partition}_qa.json and video/
        video_prefix  : filename prefix before video_id  ("video" for MSRVTT, "" for MSVD)
        video_ext     : file extension
        """
        assert os.path.exists(msqa_location), f"Path not found: {msqa_location}"
        self.dataset_path = msqa_location
        self.json_path = os.path.join(msqa_location, partition + "_qa.json")
        self.video_path = os.path.join(msqa_location, "video")
        self.num_frames = num_frames
        self.h, self.w = h, w
        self.video_prefix = video_prefix
        self.video_ext = video_ext

        with open(self.json_path, 'r') as f:
            self.entry_data = json.load(f)
        self.partition = partition

    def __len__(self):
        return len(self.entry_data)

    def _load_frames(self, video_path, sample=True):
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)

        if sample:
            if total_frames >= self.num_frames:
                indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)
            else:
                indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
                indices = np.array(indices)
        else:
            indices = list(range(total_frames))

        frames = vr.get_batch(indices)
        frames = torch.from_numpy(frames.asnumpy()).float()
        frames = torch.nn.functional.interpolate(
            frames.permute(0, 3, 1, 2),
            size=(self.h, self.w),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)
        return frames.float() / 255.0

    def __getitem__(self, idx):
        info_dict = self.entry_data[idx]
        video_id = info_dict["video_id"]
        one_video_path = os.path.join(
            self.video_path, f"{self.video_prefix}{video_id}{self.video_ext}"
        )
        assert os.path.exists(one_video_path), f"Video not found: {one_video_path}"

        info_dict = dict(info_dict)  # shallow copy to avoid mutating cache
        info_dict["video_frames"] = self._load_frames(one_video_path, sample=True)

        if self.partition != "train":
            info_dict["video_frames_b"] = self._load_frames(one_video_path, sample=False)

        return info_dict


def collate_fn(batch):
    videos = torch.stack([b["video_frames"] for b in batch])
    prompts = [b.get("question", "") for b in batch]
    ground_truth = [b.get("answer", "") for b in batch]
    video_full_exists = "video_frames_b" in batch[0]
    if video_full_exists:
        return {
            "videos": videos,
            "videos_full": torch.stack([b["video_frames_b"] for b in batch]),
            "question": prompts,
            "answer": ground_truth,
            "choices": [b.get("choices", "") for b in batch] if "choices" in batch[0] else None,
            "gt_choice": [b.get("gt_choice", "") for b in batch] if "choices" in batch[0] else None,
        }
    else:
        return {
            "videos": videos,
            "videos_full": None,
            "question": prompts,
            "answer": ground_truth,
            "choices": [b.get("choices", "") for b in batch] if "choices" in batch[0] else None,
            "gt_choice": [b.get("gt_choice", "") for b in batch] if "choices" in batch[0] else None,
        }


def get_combined_dataset(num_frames=16, h=288, w=288, partition="train", use_nextqa=True, use_msrvtt=True, use_msvd=True):
    """
    Returns a ConcatDataset of MSRVTT-QA, MSVD-QA, and optionally NExT-QA.
    Set use_nextqa=False if NExT-QA videos haven't been downloaded yet.
    """
    from msvd_dataset import MsvdQAParquetDataset

    datasets = []

    if use_msrvtt == True:
        msrvtt = VideoQADataset(
            msqa_location="./train_data/MSRVTT-QA/",
            num_frames=num_frames,
            h=h, w=w,
            partition=partition,
            video_prefix="video",
            video_ext=".mp4",
        )
        datasets.append(msrvtt)
    
    if use_msvd == True:
        msvd = MsvdQAParquetDataset(
            parquet_dir="./train_data/MSVD-QA/data/",
            num_frames=num_frames,
            h=h, w=w,
            partition=partition,
        )
        datasets.append(msvd)

    if use_nextqa and os.path.isdir("./train_data/NExT-QA/video"):
        from nextqa_dataset import NextQADataset
        nextqa = NextQADataset(
            root="./train_data/NExT-QA/",
            num_frames=num_frames,
            h=h, w=w,
            partition=partition,
        )
        if len(nextqa) > 0:
            datasets.append(nextqa)

    total = sum(len(d) for d in datasets)
    print(f"Combined {partition}: {total} samples total")
    if len(datasets) > 1:
        return ConcatDataset(datasets)
    elif len(datasets) > 0:
        return datasets[0] #only next qa
    else:
        assert False, "dataset len 0"
        return None