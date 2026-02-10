import torch
import numpy as np

COCO17_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 1-based index from source
COCO17_SKELETON_EDGES_1IDX = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7),
    (6, 8), (7, 9), (8, 10), (9, 11),
    (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),
]

def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def build_coco17_adj(device=None, dtype=torch.float32, layout='coco', strategy='spatial'):
    # COCO 17 keypoints
    num_node = 17
    # 0-based edges
    self_link = [(i, i) for i in range(num_node)]
    neighbor_link = [(i - 1, j - 1) for (i, j) in COCO17_SKELETON_EDGES_1IDX]
    edge = self_link + neighbor_link
    
    # 重心の設定: Left Hip(11) と Right Hip(12) の中間を重心とみなすため、
    # 両方を距離0（中心）として扱います。(indexは10と11)
    center_nodes = [10, 11]

    # ホップ距離の計算
    hop_dis = get_hop_distance(num_node, neighbor_link, max_hop=num_node)
    
    # 各ノードの「重心からの距離」を計算
    # (中心ノードまでの距離の最小値を採用)
    node_dist = np.min(hop_dis[center_nodes, :], axis=0)

    A = []
    # Strategy: Spatial Configuration Partitioning
    # 0: Self loop (自己)
    # 1: Centripetal (求心: 重心に近い方へ向かうエッジ)
    # 2: Centrifugal (遠心: 重心から遠い方へ向かうエッジ)
    for i in range(3):
        A.append(np.zeros((num_node, num_node)))

    for i, j in neighbor_link:
        if node_dist[i] < node_dist[j]:
            # neighbor(i) is closer to center -> j receives from i (Centripetal)
            # A[row, col] = 1 means col -> row information flow?
            # PyTorch implementation usually uses A[v, w] meaning w -> v.
            # Here: neighbor_link definition is symmetric in graph, but we need direction.
            # edge (i, j) means connection.
            
            # j -> i (i is closer) : Centripetal for j? No, flow is usually Neighbor -> Root.
            # Root(target) = i, Neighbor(source) = j.
            # If dist(j) > dist(i): Neighbor is farther. Flow is "Inward" (Centripetal).
            A[1][i, j] = 1
            A[1][j, i] = 1 # ? No, directed.
            
            # 整理:
            # (i, j) エッジについて:
            # node_dist[i] < node_dist[j]: iはjより中心に近い。
            # エッジ i-j において、
            # jにとってiは「求心(内側)」、iにとってjは「遠心(外側)」
            
            # 行列 A[target, source] として定義する場合:
            # A[j, i]: source(i) -> target(j). i is closer. Flow is Centrifugal (Outward). -> Class 2
            # A[i, j]: source(j) -> target(i). j is farther. Flow is Centripetal (Inward). -> Class 1
            
            A[2][j, i] = 1
            A[1][i, j] = 1
            
        elif node_dist[j] < node_dist[i]:
            # j is closer.
            # i -> j (Centripetal) -> A[j, i] is Class 1
            # j -> i (Centrifugal) -> A[i, j] is Class 2
            A[1][j, i] = 1
            A[2][i, j] = 1
            
        else:
            # 同距離 (ごく稀だが、腰回りなどで発生しうる)
            # 便宜上、求心に含めるなどの処理を行うが、今回は無視(0)または求心に入れる
            # ここでは距離が変わらない移動として双方Class 1に入れておく（実装による）
            A[1][j, i] = 1
            A[1][i, j] = 1

    # Self loop
    for i, j in self_link:
        A[0][i, j] = 1

    # Normalize
    for i in range(3):
        A[i] = normalize_digraph(A[i])

    # shape: (3, 17, 17)
    A = np.stack(A).astype(np.float32)
    return torch.from_numpy(A).to(device).to(dtype)