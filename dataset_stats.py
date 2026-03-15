"""
Combined Dataset Statistics for Research Paper
Covers: MSRVTT-QA and MSVD-QA
Outputs: video counts, QA counts, duration stats, resolution stats, question/answer length stats
"""

import os
import json
import numpy as np
import pandas as pd
import decord
from glob import glob
from tqdm import tqdm
from collections import defaultdict


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_video_meta(video_path):
    """Returns (duration_sec, fps, width, height) or None on failure."""
    try:
        vr = decord.VideoReader(video_path)
        fps = vr.get_avg_fps()
        n   = len(vr)
        dur = n / fps if fps > 0 else 0
        h, w, _ = vr[0].shape
        return dur, fps, w, h
    except Exception:
        return None


def percentile_summary(arr, label):
    arr = np.array(arr)
    print(f"  {label}:")
    print(f"    min={arr.min():.2f}  max={arr.max():.2f}  mean={arr.mean():.2f}  "
          f"median={np.median(arr):.2f}  std={arr.std():.2f}")
    print(f"    p25={np.percentile(arr,25):.2f}  p75={np.percentile(arr,75):.2f}  "
          f"p95={np.percentile(arr,95):.2f}")


def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


# ─────────────────────────────────────────────
# MSRVTT-QA
# ─────────────────────────────────────────────

def stats_msrvtt(root="./train_data/MSRVTT-QA", partitions=("train", "val", "test")):
    print_section("MSRVTT-QA")

    all_qa       = []
    video_ids    = set()
    durations    = []
    widths       = []
    heights      = []
    q_lengths    = []
    a_lengths    = []
    q_types      = defaultdict(int)

    for part in partitions:
        json_path = os.path.join(root, f"{part}_qa.json")
        if not os.path.exists(json_path):
            print(f"  [skip] {json_path} not found")
            continue
        with open(json_path) as f:
            data = json.load(f)
        all_qa.extend(data)
        for item in data:
            video_ids.add(item["video_id"])
            q_lengths.append(len(item["question"].split()))
            a_lengths.append(len(str(item["answer"]).split()))
            # Infer question type from first word
            qw = item["question"].strip().lower().split()[0]
            q_types[qw] += 1

        print(f"  {part}: {len(data)} QA pairs")

    print(f"\n  Total QA pairs : {len(all_qa)}")
    print(f"  Unique videos  : {len(video_ids)}")

    # Video metadata
    video_dir = os.path.join(root, "video")
    if os.path.isdir(video_dir):
        print(f"\n  Scanning videos in {video_dir} ...")
        for vid_id in tqdm(list(video_ids), desc="  MSRVTT videos"):
            path = os.path.join(video_dir, f"video{vid_id}.mp4")
            meta = get_video_meta(path)
            if meta:
                dur, fps, w, h = meta
                durations.append(dur)
                widths.append(w)
                heights.append(h)

        if durations:
            percentile_summary(durations, "Duration (sec)")
        if widths:
            percentile_summary(widths, "Width (px)")
            percentile_summary(heights, "Height (px)")
    else:
        print(f"  [skip] video dir not found: {video_dir}")

    percentile_summary(q_lengths, "Question length (words)")
    percentile_summary(a_lengths, "Answer length (words)")

    print(f"\n  Top-10 question types (first word):")
    for qw, cnt in sorted(q_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {qw:15s}: {cnt}")

    return {
        "name": "MSRVTT-QA",
        "total_qa": len(all_qa),
        "unique_videos": len(video_ids),
        "durations": durations,
        "q_lengths": q_lengths,
        "a_lengths": a_lengths,
    }


# ─────────────────────────────────────────────
# MSVD-QA
# ─────────────────────────────────────────────

def stats_msvd(parquet_dir="./train_data/MSVD-QA/data", partitions=("train", "val", "test")):
    print_section("MSVD-QA")

    total_qa     = 0
    video_ids    = set()
    durations    = []
    widths       = []
    heights      = []
    q_lengths    = []
    a_lengths    = []
    q_types      = defaultdict(int)

    for part in partitions:
        files = sorted(glob(os.path.join(parquet_dir, f"{part}-*.parquet")))
        if not files:
            print(f"  [skip] no parquet files for partition: {part}")
            continue

        part_qa = 0
        part_vids = set()
        for fpath in tqdm(files, desc=f"  MSVD {part}"):
            df = pd.read_parquet(fpath)
            for _, row in df.iterrows():
                vid_path = str(row.get("video_path", ""))
                vid_id   = vid_path  # use path as unique ID
                part_vids.add(vid_id)
                video_ids.add(vid_id)

                H   = int(row["height"])
                W   = int(row["width"])
                widths.append(W)
                heights.append(H)

                # video_path looks like: ./msvd-qa/data//YouTubeClips/YmXCfQm0_CA_7_16.avi
                # Filename encodes clip start/end seconds: NAME_start_end.avi
                vid_path = str(row.get("video_path", ""))
                fname = os.path.basename(vid_path).replace(".avi", "").replace(".mp4", "")
                parts = fname.rsplit("_", 2)
                if len(parts) == 3:
                    try:
                        start, end = float(parts[1]), float(parts[2])
                        durations.append(end - start)
                    except ValueError:
                        pass  # skip if filename doesn't follow convention

                qa_pairs = row["qa"]
                for qa in qa_pairs:
                    q = str(qa[0])
                    a = str(qa[1])
                    q_lengths.append(len(q.split()))
                    a_lengths.append(len(a.split()))
                    qw = q.strip().lower().split()[0]
                    q_types[qw] += 1
                    part_qa += 1

        total_qa += part_qa
        print(f"  {part}: {part_qa} QA pairs, {len(part_vids)} unique videos")

    print(f"\n  Total QA pairs : {total_qa}")
    print(f"  Unique videos  : {len(video_ids)}")

    percentile_summary(durations, "Duration (sec) [estimated at 29.97fps]")
    percentile_summary(widths,    "Width (px)")
    percentile_summary(heights,   "Height (px)")
    percentile_summary(q_lengths, "Question length (words)")
    percentile_summary(a_lengths, "Answer length (words)")

    print(f"\n  Top-10 question types (first word):")
    for qw, cnt in sorted(q_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {qw:15s}: {cnt}")

    return {
        "name": "MSVD-QA",
        "total_qa": total_qa,
        "unique_videos": len(video_ids),
        "durations": durations,
        "q_lengths": q_lengths,
        "a_lengths": a_lengths,
    }


# ─────────────────────────────────────────────
# Combined summary
# ─────────────────────────────────────────────

def combined_summary(stats_list):
    print_section("COMBINED DATASET SUMMARY")

    total_qa     = sum(s["total_qa"]     for s in stats_list)
    total_videos = sum(s["unique_videos"] for s in stats_list)
    all_durations = []
    all_q_lengths = []
    all_a_lengths = []
    for s in stats_list:
        all_durations.extend(s["durations"])
        all_q_lengths.extend(s["q_lengths"])
        all_a_lengths.extend(s["a_lengths"])

    print(f"\n  {'Dataset':<15} {'Videos':>10} {'QA pairs':>12}")
    print(f"  {'-'*40}")
    for s in stats_list:
        print(f"  {s['name']:<15} {s['unique_videos']:>10,} {s['total_qa']:>12,}")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<15} {total_videos:>10,} {total_qa:>12,}")

    if all_durations:
        print()
        percentile_summary(all_durations, "Combined video duration (sec)")
        total_hours = sum(all_durations) / 3600
        print(f"    Total video time: {total_hours:.1f} hours")

    percentile_summary(all_q_lengths, "Combined question length (words)")
    percentile_summary(all_a_lengths, "Combined answer length (words)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def stats_nextqa(root="./train_data/NExT-QA", partitions=("train", "val", "test")):
    print_section("NExT-QA")

    total_qa  = 0
    video_ids = set()
    durations = []
    widths    = []
    heights   = []
    q_lengths = []
    a_lengths = []
    q_types   = defaultdict(int)
    qtypes_dist = defaultdict(int)  # causal/temporal/descriptive

    map_path = os.path.join(root, "annotations", "map_vid_vidorID.json")
    video_dir = os.path.join(root, "video")
    with open(map_path) as f:
        vid_map = json.load(f)

    for part in partitions:
        csv_path = os.path.join(root, "annotations", f"{part}.csv")
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        answer_cols = ["a0", "a1", "a2", "a3", "a4"]
        part_qa = 0
        for _, row in df.iterrows():
            vid_id = str(row["video"])
            video_ids.add(vid_id)
            correct_idx = int(row["answer"])
            answer_text = str(row[answer_cols[correct_idx]])
            q_lengths.append(len(str(row["question"]).split()))
            a_lengths.append(len(answer_text.split()))
            qw = str(row["question"]).strip().lower().split()[0]
            q_types[qw] += 1
            qtype = str(row["type"])
            # C=causal, T=temporal, D=descriptive
            qtypes_dist[qtype[0]] += 1
            part_qa += 1
        total_qa += part_qa
        print(f"  {part}: {part_qa} QA pairs")

    print(f"\n  Total QA pairs : {total_qa}")
    print(f"  Unique videos  : {len(video_ids)}")

    # Video metadata
    if os.path.isdir(video_dir):
        print(f"\n  Scanning videos ...")
        for vid_id in tqdm(list(video_ids), desc="  NExT-QA videos"):
            folder = vid_map.get(vid_id)
            if not folder:
                continue
            path = os.path.join(video_dir, f"{folder}.mp4")
            meta = get_video_meta(path)
            if meta:
                dur, fps, w, h = meta
                durations.append(dur)
                widths.append(w)
                heights.append(h)
        if durations:
            percentile_summary(durations, "Duration (sec)")
            percentile_summary(widths,    "Width (px)")
            percentile_summary(heights,   "Height (px)")
    else:
        print(f"  [skip] video dir not found: {video_dir}")

    percentile_summary(q_lengths, "Question length (words)")
    percentile_summary(a_lengths, "Answer length (words)")

    print(f"\n  Question category distribution:")
    cat_map = {"C": "Causal", "T": "Temporal", "D": "Descriptive"}
    for k, v in sorted(qtypes_dist.items()):
        print(f"    {cat_map.get(k, k):15s}: {v:6d} ({100*v/total_qa:.1f}%)")

    print(f"\n  Top-10 question types (first word):")
    for qw, cnt in sorted(q_types.items(), key=lambda x: -x[1])[:10]:
        print(f"    {qw:15s}: {cnt}")

    return {
        "name": "NExT-QA",
        "total_qa": total_qa,
        "unique_videos": len(video_ids),
        "durations": durations,
        "q_lengths": q_lengths,
        "a_lengths": a_lengths,
    }


if __name__ == "__main__":
    results = []
    results.append(stats_msrvtt())
    results.append(stats_msvd())
    if os.path.isdir("./train_data/NExT-QA"):
        results.append(stats_nextqa())
    combined_summary(results)