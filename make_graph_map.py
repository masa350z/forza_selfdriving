# %%
from PIL import Image
import pickle
from typing import List, Tuple
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from math import sqrt
from itertools import combinations
import networkx as nx
from skimage.morphology import skeletonize
from scipy.ndimage import label, center_of_mass
import numpy as np
from modules import extract_driving_line_from_d32, extract_radius_from_d32, down_scale_map
import config

ERASE_R = 3
DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, sqrt(2)), (-1, 1, sqrt(2)),
        (1, -1, sqrt(2)),  (1, 1, sqrt(2))]


def geodesic_length(coord_set: set[tuple[int, int]],
                    start: tuple[int, int],
                    goal: tuple[int, int]) -> float:
    """ピクセル集合を無向重み付きグラフとみなし、最短距離を返す"""
    DIR_COST = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, sqrt(2)), (-1, 1, sqrt(2)),
                (1, -1, sqrt(2)), (1, 1, sqrt(2))]
    pq = [(0.0, start)]
    dist = {start: 0.0}
    while pq:
        d, cur = heappop(pq)
        if cur == goal:
            return d
        if dist[cur] < d:          # outdated
            continue
        x, z = cur
        for dx, dz, c in DIR_COST:
            nxt = (x+dx, z+dz)
            if nxt not in coord_set:
                continue
            nd = d + c
            if nd < dist.get(nxt, 1e18):
                dist[nxt] = nd
                heappush(pq, (nd, nxt))
    return 0.0                    # 非連結の場合


def get_degree_dict(coords: np.ndarray) -> dict[tuple[int, int], int]:
    """各画素の次数(同ラベル近傍数)を返す"""
    st = set(map(tuple, coords))
    deg = {}
    for x, z in coords:
        cnt = sum(((x+dx, z+dz) in st) for dx, dz, _ in DIRS)
        deg[(x, z)] = cnt
    return deg


def endpoints(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    """次数=1 を端点とし、0本なら最遠ペアを返す"""
    deg = get_degree_dict(coords)
    ends = [p for p, d in deg.items() if d == 1]
    if len(ends) >= 2:
        return ends[0], ends[1]  # 通常は2本
    # 環状の場合 – 画素集合内でユークリッド距離最大ペア
    max_d = -1
    ep0 = ep1 = None
    for a, b in combinations(coords, 2):
        d = (a[0]-b[0])**2 + (a[1]-b[1])**2
        if d > max_d:
            max_d, ep0, ep1 = d, tuple(a), tuple(b)
    return ep0, ep1


def erase_nodes(skel: np.ndarray,
                nodes: list[tuple[int, int]],
                radius: int = ERASE_R) -> np.ndarray:
    """ノード中心半径 r 内を 0 にする"""
    img = skel.copy()
    W, H = img.shape
    for cx, cz in nodes:
        x0 = max(cx - radius, 0)
        x1 = min(cx + radius + 1, W)
        z0 = max(cz - radius, 0)
        z1 = min(cz + radius + 1, H)
        zz, xx = np.ogrid[z0:z1, x0:x1]  # 正しい軸順
        mask = (zz - cz)**2 + (xx - cx)**2 <= radius**2
        img[x0:x1, z0:z1][mask] = 0
    return img


def choose_longest_endpair(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], float]:
    """次数=1 の端点ペアの中で、スケルトン実長が最大のものを返す"""
    deg = get_degree_dict(coords)
    ends = [p for p, d in deg.items() if d == 1]
    if len(ends) < 2:                      # ループなど → 最遠ユークリッド
        ep0, ep1 = endpoints(coords)
        g_len = geodesic_length(set(map(tuple, coords)), ep0, ep1)
        return ep0, ep1, g_len

    best_len = -1.0
    best_pair = (ends[0], ends[1])
    coord_set = set(map(tuple, coords))
    for a, b in combinations(ends, 2):
        g = geodesic_length(coord_set, a, b)
        if g > best_len:
            best_len = g
            best_pair = (a, b)
    return best_pair[0], best_pair[1], best_len


def extract_edges(skeleton_img: np.ndarray,
                  node_centers: list[tuple[int, int]],
                  erase_r: int = ERASE_R,
                  save_coords=False) -> np.ndarray:
    """
    Args:
        skeleton_img : uint8 0/1 スケルトン画像
        node_centers : [(cx,cz), ...] 交差点ノード中心 (array座標)
    Returns:
        np.ndarray dtype=object : [[edge_id, road_len(px), (xs,zs), (xe,ze)], ...]
    """
    sk = erase_nodes(skeleton_img, node_centers, erase_r)
    labeled, n_cc = label(sk, structure=np.ones((3, 3)))

    edges: list[list] = []

    for lab in range(1, n_cc + 1):
        coords = np.column_stack(np.where(labeled == lab))
        if coords.size == 0:
            continue
        p0, p1, length_px = choose_longest_endpair(coords)

        # 原点(0,0)に近い端点を始点
        if (p1[0]**2 + p1[1]**2) < (p0[0]**2 + p0[1]**2):
            p0, p1 = p1, p0  # 入れ替え

        edge_id = len(edges)
        edges.append([edge_id, length_px, tuple(p0), tuple(p1)])
        if save_coords:
            np.save(f'map/road_edges/edge_{edge_id}', coords.astype(np.uint16))

    return np.array(edges, dtype=object)


def visualize_edges_colored(
    skeleton: np.ndarray,
    edges: np.ndarray,
    COMS,
    figsize=(10, 10),
    xlim: tuple[int, int] = None,
    ylim: tuple[int, int] = None
):
    """
    各エッジを色分けして可視化する（IDと長さ付き）

    Args:
        skeleton (np.ndarray): 背景用のスケルトン画像 (H, W), 0/1
        edges (np.ndarray): extract_edges() で得られた配列
            shape=(N, 4): [edge_id, road_len, (x0,z0), (x1,z1)]
        figsize (tuple[int, int]): matplotlib の figure サイズ
        xlim (tuple[int, int], optional): 表示する x 範囲 (横: col)
        ylim (tuple[int, int], optional): 表示する y 範囲 (縦: row)
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(skeleton.T, cmap='gray', interpolation='none', origin='upper')
    ax.set_title("Road Edges with Colors and Labels")
    ax.axis("on")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)  # 画像y軸は上が0なので逆順

    cmap = cm.get_cmap('hsv', len(edges))
    norm = mcolors.Normalize(vmin=0, vmax=len(edges) - 1)

    for eid, length, p0, p1 in edges:
        y0, x0 = p0[1], p0[0]
        y1, x1 = p1[1], p1[0]
        color = cmap(norm(eid))
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2)

        # 中点にラベル表示
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(xm, ym,
                f"{eid}\n{length:.1f}px",
                fontsize=9,
                color=color,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))

    for nid, (axx, ay) in enumerate(COMS):
        ax.plot(axx, ay, 'ro', markersize=24)
        ax.text(axx, ay - 3, f"{nid}", color='yellow', ha='center', fontsize=16)

    plt.tight_layout()
    plt.show()


def _nearest_node_id(
    coms: List[Tuple[int, int]],
    point: Tuple[int, int]
) -> int:
    """ポイントに最も近いノード ID を返す（単純ユークリッド最小）。"""
    cx_arr = np.array([c[0] for c in coms])
    cz_arr = np.array([c[1] for c in coms])
    dx = cx_arr - point[0]
    dz = cz_arr - point[1]
    dist2 = dx * dx + dz * dz
    return int(dist2.argmin())


def build_road_graph(
    coms: List[Tuple[int, int]],
    edges: np.ndarray,
    scale: float = 1.0
) -> nx.Graph:
    """交差点 COMS とエッジ配列から重み付き Graph を生成する。

    Args:
        coms (list[tuple[int,int]]): 各ノード中心 (array_x, array_z)
        edges (np.ndarray): shape=(M,4) – [edge_id, road_len(px),
                                           start(x,z), end(x,z)]
        scale (float, optional): ピクセル→物理距離変換係数。
                                 1(px) = scale(m)。デフォルト1.

    Returns:
        networkx.Graph: ノード属性
            - 'array': (x,z)
          エッジ属性
            - 'length': 距離[m]
    """
    G = nx.Graph()

    # --- ノード追加 ---
    for nid, (ax, az) in enumerate(coms):
        G.add_node(nid, array=(ax, az))

    # --- エッジ追加 ---
    for edge_id, road_len_px, p0, p1 in edges:
        n0 = _nearest_node_id(coms, p0)
        n1 = _nearest_node_id(coms, p1)
        if n0 == n1:            # 同じノードに帰属 → 無視
            continue
        road_len_m = float(road_len_px) * scale
        # 多重エッジ対策: 既にある場合短い方を残す
        if G.has_edge(n0, n1):
            if road_len_m < G[n0][n1]['length']:
                G[n0][n1]['length'] = road_len_m
                G[n0][n1]['edge_id'] = edge_id
        else:
            G.add_edge(n0, n1, length=road_len_m, edge_id=edge_id)

    return G


def shortest_path_by_distance(
    G: nx.Graph,
    src: int,
    dst: int
):
    """距離重みで最短経路を返す。

    Args:
        G (nx.Graph): build_road_graph で得たグラフ
        src (int): 出発ノード ID
        dst (int): 目的ノード ID

    Returns:
        tuple[list[int], float]:
            - ノード ID 列
            - 総距離[m]
    """
    path = nx.shortest_path(G, src, dst, weight='length')
    total = nx.path_weight(G, path, weight='length')
    return path, total


def route_nodes_to_edges(route: list[int], G: nx.Graph) -> list[int]:
    """
    ノード列 [n1,n2,…] をエッジ ID 列 [e1,e2,…] に変換する。

    Args:
        route (list[int]): 最短経路で得たノード ID 列
        G (nx.Graph)    : build_road_graph() で作ったグラフ

    Returns:
        list[int]: 経由する edge_id の順序付きリスト
    """
    edge_ids = []
    for u, v in zip(route[:-1], route[1:]):
        if not G.has_edge(u, v):
            raise ValueError(f"edge {u}-{v} not found in graph")
        edge_ids.append(G[u][v]['edge_id'])
    return edge_ids


def make_routed_x1_driveline(x4_map, road_edge_indexes, array_shape=(4250, 2500)):
    road_edge_map = np.zeros(array_shape, dtype=np.uint16)

    for i in road_edge_indexes:
        road_edge = np.load(f'map/road_edges/edge_{i}.npy')
        road_edge_map[road_edge[:, 0], road_edge[:, 1]] = 1

    div5_driving_line = down_scale_map(road_edge_map, 5, mode='max')

    x1_driveline = x4_map*div5_driving_line.repeat(20, axis=0).repeat(20, axis=1)

    return x1_driveline


def main(road_map, CURV_THR=30, MIN_AREA=10):
    road_raw = np.memmap(road_map, dtype=np.uint32,
                         shape=(config.MAP_SIZE_X*config.MAP_SCALE, config.MAP_SIZE_Z*config.MAP_SCALE), mode="r")

    drive_line = extract_driving_line_from_d32(road_raw)   # 0–127
    radius_map = extract_radius_from_d32(road_raw)         # 0–255
    radius_map[radius_map == 0] = 255

    drive_line = down_scale_map(drive_line, 16, mode='max')
    radius_map = down_scale_map(radius_map, 16, mode='min')

    # ★道路マスク（0/1）
    road_mask = (drive_line > config.ROAD_CENTER_DISTANCE_THRESHOLD).astype(np.uint8)

    # ★高曲率マスク（＝交差点候補）
    curve_mask = (radius_map < CURV_THR) & road_mask

    labeled, n_cc = label(curve_mask, structure=np.ones((3, 3)))
    COMS = []              # 各ノード: (array_x, array_z)

    for idx in range(1, n_cc+1):
        area = np.sum(labeled == idx)
        if area < MIN_AREA:        # ノイズ除去
            continue
        cx, cz = center_of_mass(curve_mask, labeled, idx)
        COMS.append((int(cx), int(cz)))

    # 画像を [0,1] にしてからスケルトン化
    skeleton = skeletonize(road_mask.astype(bool)).astype(np.uint8)

    edge_array = extract_edges(skeleton.astype(np.uint8), COMS, save_coords=True)

    G = build_road_graph(COMS, edge_array, scale=1)

    return G, skeleton, edge_array, COMS


# %%
G, skeleton, edge_array, COMS = main(road_map="map/road_map_x4.dat", CURV_THR=30)
visualize_edges_colored(skeleton, edge_array, COMS,
                        xlim=(200, 650),
                        ylim=(650, 1000))
# %%
GRAPH_MAP_FILE = 'map/road_graph_.pickle'
with open(GRAPH_MAP_FILE, 'wb') as f:
    pickle.dump(G, f)
# %%
CURV_THR = 40
MIN_AREA = 10
road_map = "map/road_map_x4.dat"
# %%
road_raw = np.memmap(road_map, dtype=np.uint32,
                     shape=(config.MAP_SIZE_X*config.MAP_SCALE, config.MAP_SIZE_Z*config.MAP_SCALE), mode="r")

drive_line = extract_driving_line_from_d32(road_raw)   # 0–127
radius_map = extract_radius_from_d32(road_raw)         # 0–255
radius_map[radius_map == 0] = 255

drive_line = down_scale_map(drive_line, 16, mode='max')
radius_map = down_scale_map(radius_map, 16, mode='min')
# %%
# ★道路マスク（0/1）
road_mask = (drive_line > config.ROAD_CENTER_DISTANCE_THRESHOLD).astype(np.uint8)

# ★高曲率マスク（＝交差点候補）
curve_mask = (radius_map < CURV_THR) & road_mask

labeled, n_cc = label(curve_mask, structure=np.ones((3, 3)))
COMS = []              # 各ノード: (array_x, array_z)

for idx in range(1, n_cc+1):
    area = np.sum(labeled == idx)
    if area < MIN_AREA:        # ノイズ除去
        continue
    cx, cz = center_of_mass(curve_mask, labeled, idx)
    COMS.append((int(cx), int(cz)))

# 画像を [0,1] にしてからスケルトン化
skeleton = skeletonize(road_mask.astype(bool)).astype(np.uint8)
# %%
Image.fromarray(((skeleton*255)[int(800/4):int(2400/4), int(2500/4):int(4000/4)].T)[::-1].astype(np.uint8))
# %%
Image.fromarray(((((radius_map < 20) & road_mask)*255)[int(800/4):int(2400/4), int(2500/4):int(4000/4)].T)[::-1].astype(np.uint8))

# %%
