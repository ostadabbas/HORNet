import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from glob import glob
from torch.utils.data import Dataset


class MsvdQAParquetDataset(Dataset):
    """
    Reads MSVD-QA directly from parquet files.
    Each parquet row has multiple QA pairs → we flatten to one entry per QA pair.
    binary_frames is raw bytes of shape [T, H, W, C] uint8.
    """

    def __init__(self, parquet_dir, num_frames=16, h=288, w=288, partition="train"):
        self.num_frames = num_frames
        self.h = h
        self.w = w
        self.partition = partition

        pattern = os.path.join(parquet_dir, f"{partition}-*.parquet")
        files = sorted(glob(pattern))
        assert len(files) > 0, f"No parquet files found for pattern: {pattern}"

        # Load all parquet files and flatten QA pairs
        self.entries = []  # list of (row_dict, question, answer)
        for f in files:
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                qa_pairs = row['qa']
                for qa in qa_pairs:
                    # qa is [question, answer] or {'question':..., 'answer':...}
                    # qa is a numpy array like ['question text', 'answer text']
                    question, answer = str(qa[0]), str(qa[1])
                    self.entries.append({
                        "binary_frames": row['binary_frames'],
                        "num_frames_orig": int(row['num_frames']),
                        "height": int(row['height']),
                        "width": int(row['width']),
                        "channels": int(row['channels']),
                        "question": question,
                        "answer": answer,
                    })

        print(f"MSVD-QA {partition}: {len(self.entries)} QA pairs loaded")

    def __len__(self):
        return len(self.entries)

    def _decode_frames(self, entry, do_sample):
        T = entry['num_frames_orig']
        H = entry['height']
        W = entry['width']
        C = entry['channels']

        frames = np.frombuffer(entry['binary_frames'], dtype=np.uint8)
        frames = frames.reshape(T, H, W, C)                        # [T, H, W, C]
        frames = torch.from_numpy(frames.copy()).float()            # [T, H, W, C]

        # Sample/pad to num_frames
        if T >= self.num_frames:
            indices = np.linspace(0, T - 1, self.num_frames).astype(int)
        else:
            indices = list(range(T)) + [T - 1] * (self.num_frames - T)
            indices = np.array(indices)
        if do_sample:
            frames = frames[indices]                                    # [num_frames, H, W, C]
        else:
            frames = frames[list(range(T))]

        # Resize
        frames = F.interpolate(
            frames.permute(0, 3, 1, 2),                            # [T, C, H, W]
            size=(self.h, self.w),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1)                                      # [T, H, W, C]

        return frames / 255.0

    def __getitem__(self, idx):
        entry = self.entries[idx]
        if self.partition == "train":
            return {
                "video_frames": self._decode_frames(entry, True),
                "question": entry['question'],
                "answer": entry['answer'],
            }
        else:
            return {
                "video_frames": self._decode_frames(entry, True),
                "video_frames_b": self._decode_frames(entry, False),
                "question": entry['question'],
                "answer": entry['answer'],
            }