import torch

COCO17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
COCO17_SKELETON_EDGES_1IDX = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11),
    (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
]

def build_coco17_adj(device=None, dtype=torch.float32):
    K = 17
    adj = torch.zeros((K, K), dtype=dtype, device=device)

    for a, b in COCO17_SKELETON_EDGES_1IDX:
        i = a - 1
        j = b - 1
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return adj
