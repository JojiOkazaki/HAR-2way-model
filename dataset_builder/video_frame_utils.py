import cv2
import numpy as np

# sizesの中で一番小さいフレーム数をnum個のフレームにサンプリングする
def sample_frame_ids(sizes, num=32):
    size = min(sizes)
    if size <= 0:
        raise ValueError("frame size must be 0 or greater.") 
    
    if num == 1:
        return [0]
    
    return [round((size - 1) * i / (num - 1)) for i in range(num)]

# 動画からフレーム番号の画像を取得する
def read_frames_by_id(video_path, frame_ids):
    img_dict = {}
    img_id = 0

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, image = cap.read()
        if not ret:
            break
        if img_id in frame_ids:
            img_dict[img_id] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_id += 1
    cap.release()

    if not img_dict:
        raise RuntimeError(f"No frames read: {video_path}")
    
    return img_dict

# 画像からbbox部分の人物画像を取得する
def crop_person_image(img, bbox, out_size):
    h, w = img.shape[:2]
    out_w, out_h = out_size
    x1, y1, x2, y2 = bbox

    # bboxが画像外でも対応
    x1 = int(max(0, min(w - 1, x1)))
    x2 = int(max(0, min(w, x2)))
    y1 = int(max(0, min(h - 1, y1)))
    y2 = int(max(0, min(h, y2)))

    if x2 <= x1 or y2 <= y1:
        # 有効でない場合はゼロ埋めしたデータにする
        crop = np.zeros((3, out_h, out_w), dtype=np.float32)
    else:
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (out_w, out_h)) 
        crop = crop.astype(np.float32) / 255.0 # 0~1に正規化
        crop = np.transpose(crop, (2, 0, 1))  # (H, W, C) → (C, H, W)

    return crop # shape: (C, H, W)

# 指定フレームに最も近いフレーム番号群のフレームを取得する
def find_closest_past_frame(frames, now_frame):
    candidates = [i for i in frames if i <= now_frame]
    nearest_frame = candidates[-1] if candidates else frames[0]
    return nearest_frame
