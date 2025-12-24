import os
import torch
import orjson

from dataset_builder.data_generation import generate_video_sample

def write_video_sample(video_dir, json_dir, out_img_dir, out_skel_dir, label, label_id, video_filename, P, T, J, W, H):
    # 出力先ディレクトリの取得
    out_img_dir = out_img_dir / label
    out_skel_dir = out_skel_dir / label
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_skel_dir, exist_ok=True)

    # 元データパスの取得
    video_path = video_dir / label / video_filename
    json_path = json_dir / label / video_filename.replace(".avi", ".json")

    # 出力先パスの取得
    out_img_path = out_img_dir / video_filename.replace(".avi", ".pt")
    out_skel_path = out_skel_dir / video_filename.replace(".avi", ".pt")

    # 元データパスがない場合は処理しない
    if not (os.path.exists(video_path) and os.path.exists(json_path)):
        return

    # jsonデータの取得
    with open(json_path, "rb") as f:
        json_data = orjson.loads(f.read())

    # 処理
    frames, img_data, skel_data, score_data = generate_video_sample(video_path, json_data, P, T, J, W, H)
    labels = torch.full((P,), label_id, dtype=torch.long)

    # ptファイルを作成
    torch.save(
        {"images": img_data, "frames": frames, "labels": labels},
        out_img_path,
    )
    torch.save(
        {"skeletons": skel_data, "scores": score_data, "frames": frames, "labels": labels},
        out_skel_path,
    )
