import cv2
import torch
import numpy as np

from dataset_builder.video_frame_utils import sample_frame_ids
from dataset_builder.video_frame_utils import read_frames_by_id
from dataset_builder.video_frame_utils import find_closest_past_frame
from dataset_builder.video_frame_utils import crop_person_image

def generate_video_sample(video_path, json_data, T, N, K, W, H):
    # 動画データのフレーム数とjsonデータのフレーム数を基にフレーム数を決定
    json_frames = min(json_data["num_frames"], len(json_data["frames"]))

    cap = cv2.VideoCapture(video_path)
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    frames = sample_frame_ids([video_frames, json_frames], T)

    # 画像群の取得
    imgs = read_frames_by_id(video_path, frames)

    person_img_data = []
    skel_data = []
    score_data = []

    for frame in frames:
        frame = min(frame, len(json_data["frames"]) - 1)
        
        # 画像データを取得
        img = imgs[find_closest_past_frame(frames, frame)]

        # jsonデータを取得
        persons = json_data["frames"][frame]["persons"]
        persons = preprocess_persons(persons, N, K)

        person_img_list = []
        skel_list = []
        score_list = []

        for person in persons:
            # 人物画像の取得
            bbox = person.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
            person_img = crop_person_image(img, bbox, (W, H)) # shape: (C, H, W)
            person_img_list.append(person_img)

            # 骨格、信頼度スコアの取得
            skel = np.array([row[:2] for row in person["keypoints_coco"]]) # shape: (K, 2)
            score = np.array([row[2] for row in person["keypoints_coco"]]) # shape: (K, )
            skel /= np.array([W, H], dtype=np.float32) # # 0~1に正規化
            skel_list.append(skel)
            score_list.append(score)

        person_img_data.append(person_img_list)
        skel_data.append(skel_list)
        score_data.append(score_list)
    
    # torch.Tensor化
    frames = torch.tensor(frames, dtype=torch.long)
    person_img_data = torch.from_numpy(np.array(person_img_data, dtype=np.float32)) # shape: (T, N, C_img, H, W)
    skel_data = torch.from_numpy(np.array(skel_data, dtype=np.float32)) # shape: (T, N, K, C_kp)
    score_data = torch.from_numpy(np.array(score_data, dtype=np.float32)) # shape: (T, N, K)

    return frames, person_img_data, skel_data, score_data

# personsデータをソート、指定人数分のみ、欠損の仮データ化する
def preprocess_persons(persons, person_max, kp_len=17):
    # データのある人物のみを取得する
    usable_persons = []
    for person in persons:
        person_id = person.get("person_id", None)
        bbox = person.get("bbox_xyxy")
        kps = person.get("keypoints_coco")

        if person_id is not None and bbox is not None and kps is not None:
            usable_persons.append((person_id, person))

    # person_idでの昇順ソート(指定人数分に収める)
    usable_persons.sort(key=lambda x: x[0])
    persons_sorted = [p for _, p in usable_persons[:person_max]]

    # 不足している部分をゼロ埋めした仮データで埋める
    while len(persons_sorted) < person_max:
        persons_sorted.append({
            "person_id": -1,
            "bbox_xyxy": [0.0, 0.0, 0.0, 0.0],
            "keypoints_coco": [[0.0, 0.0, 0.0] for _ in range(kp_len)],
            "is_dummy": True,
        })

    return persons_sorted
