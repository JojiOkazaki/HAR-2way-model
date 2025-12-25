import os
import cv2
import orjson
import torch
import csv
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import dataset_builder.config as config
from dataset_builder.script_writer import write_video_sample
from config_base import *
try:
    from config_local import *
except ImportError:
    pass

def load_label_sentences(csv_path):
    mapping = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["label_id"])] = row["sentence"]
    return mapping

# 並列処理によるすべてのファイルについてptデータ作成
def process(video_dir, json_dir, out_img_dir, out_skel_dir, num_workers, P, T, J, W, H):
    labels = sorted(
        d for d in os.listdir(video_dir)
        if os.path.isdir(video_dir / d)
    )
    label_ids = {label: i for i, label in enumerate(labels)}
    label_sentences = load_label_sentences(DATASET_ROOT / "ucf101_sentences_jp.csv")
    
    jobs = []
    for label in label_ids:
        label_dir = video_dir / label
        video_filenames = [f for f in os.listdir(label_dir) if f.endswith(".avi")]
        label_id = label_ids[label]
        label_sentence = label_sentences[label_id]
        for video_filename in video_filenames:
            jobs.append((
                video_dir,
                json_dir,
                out_img_dir,
                out_skel_dir,
                label,
                label_id,
                label_sentence,
                video_filename,
                P, T, J, W, H
            ))

    print(f"Total videos: {len(jobs)}")
    print(f"Using {num_workers} workers, P={P}, T={T}, J={J}, W={W}, H={H}")

    with Pool(processes=num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(_worker, jobs), total=len(jobs)):
            pass

    print("Done.")

def _worker(args):
    return write_video_sample(*args)

if __name__ == "__main__":
    process(
        RAW_DATA_ROOT / config.VIDEO_DIR,
        DATASET_ROOT / config.POSE_DIR,
        DATASET_ROOT / config.OUT_IMAGE_DIR,
        DATASET_ROOT / config.OUT_SKELETON_DIR,
        config.NUM_WORKERS,
        config.P,
        config.T,
        config.J,
        config.W,
        config.H
    )
