import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import decord
from torch.utils.data import Dataset


class NextQADataset(Dataset):
    """
    NExT-QA dataset.
    CSV columns: video, frame_count, width, height, question, answer (0-4 index), qid, type, a0-a4
    Videos are stored as: {video_dir}/{folder}/{video_id}.mp4
    where folder comes from map_vid_vidorID.json: {"video_id": "folder/video_id"}
    """

    def __init__(self, root, num_frames=16, h=288, w=288, partition="train"):
        """
        root: path to NExT-QA folder, e.g. ./train_data/NExT-QA
              expects:
                {root}/annotations/train.csv  (or val/test)
                {root}/annotations/map_vid_vidorID.json
                {root}/video/{folder}/{video_id}.mp4
        """
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.partition = partition

        csv_path = os.path.join(root, "annotations", f"{partition}.csv")
        map_path = os.path.join(root, "annotations", "map_vid_vidorID.json")
        self.video_dir = os.path.join(root, "video")

        assert os.path.exists(csv_path), f"CSV not found: {csv_path}"
        assert os.path.exists(map_path), f"Map not found: {map_path}"

        with open(map_path) as f:
            self.vid_map = json.load(f)  # {"video_id_str": "folder/video_id"}

        df = pd.read_csv(csv_path)
        answer_cols = ["a0", "a1", "a2", "a3", "a4"]

        self.entries = []
        missing = 0
        for _, row in df.iterrows():
            vid_id = str(row["video"])
            folder_path = self.vid_map.get(vid_id)
            if folder_path is None:
                missing += 1
                continue
            video_path = os.path.join(self.video_dir, f"{folder_path}.mp4")
            if not os.path.exists(video_path):
                missing += 1
                continue

            correct_idx = int(row["answer"])
            answer_text = str(row[answer_cols[correct_idx]])
            choices = row[answer_cols].tolist()
            for i in range(len(choices)):
                choices[i] = f"{i}. {choices[i]}"

            self.entries.append({
                "video_path": video_path,
                "question": str(row["question"]),
                "answer": answer_text,
                "qtype": str(row["type"]),
                "choices": choices,
                "gt_choice": correct_idx,
            })

        print(f"NExT-QA {partition}: {len(self.entries)} QA pairs loaded "
              f"({missing} skipped — video not found)")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        vr = decord.VideoReader(entry["video_path"])
        total = len(vr)

        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames).astype(int)
        else:
            indices = list(range(total)) + [total - 1] * (self.num_frames - total)
            indices = np.array(indices)
    
        frames = vr.get_batch(indices)
        frames = torch.from_numpy(frames.asnumpy()).float()
        frames = F.interpolate(
            frames.permute(0, 3, 1, 2),
            size=(self.h, self.w),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)
        frames = frames / 255.0

        if self.partition != "train":
            frames_b = vr.get_batch(list(range(len(vr))))
            frames_b = torch.from_numpy(frames_b.asnumpy()).float()
            frames_b = F.interpolate(
                frames_b.permute(0, 3, 1, 2),
                size=(self.h, self.w),
                mode="bilinear",
                align_corners=False
            ).permute(0, 2, 3, 1)
            frames_b = frames_b / 255.0

            return {
                "video_frames": frames,
                "question": entry["question"],
                "answer": entry["answer"],
                "choices": entry["choices"],
                "gt_choice": entry["gt_choice"],
                "video_frames_b": frames_b
            }
        else:
            return {
                "video_frames": frames,
                "question": entry["question"],
                "answer": entry["answer"],
                "choices": entry["choices"],
                "gt_choice": entry["gt_choice"],
            }