# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_manual_graph_map.py
========================
赤円付き PNG + RAW (32-bit) から
    • road_graph_manual.pickle
    • map/road_edges_manual/edge_<id>.npy
    • img/manual_edges_labeled.png
    • img/route_x1.png  ← 最短経路(例:0→3)のドライビングライン
を生成する。
"""
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from __future__ import annotations
from typing import List, Tuple
import os
import pickle

import numpy as np
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from heapq import heappush, heappop
from itertools import combinations
from math import sqrt
import make_quadrant_map

import config
from modules import (
    down_scale_map, extract_driving_line_from_d32
)

# ---------- パス設定 ----------
MASK_PNG = "img/map_mask.png"                       # 透明+白+赤 の PNG
RAW_DAT = "map/road_map_x4.dat"            # 32-bit RAW (read-only)
OUT_PKL = "map/road_graph_manual.pickle"   # 出力グラフ
EDGE_DIR = "map/road_edges_manual"          # エッジ座標保存ディレクトリ
LABEL_IMG = "img/manual_edges_labeled.png"   # エッジIDラベル付き画像
ROUTE_IMG = "img/route_x1.png"               # ルートドライビングライン
# ---------- しきい値 ----------
RED_THR = (200, 60, 60)
WHITE_THR = (200, 200, 200)
CURV_THR = 30            # 曲率しきい値 (今回使わないが互換保持)
MIN_AREA = 10
ERASE_MARGIN = 1
PX2M = 1.0                # 1 px ≒ 1 m
MAX_ROAD_LEN_PX = 20000
# -----------------------------
TURN_RULES: dict[int, dict[int, set[int]]] = {
    # 例: 交差点 12 では edge 5 から 7,8 へ直進のみ許可
    # 12: {5: {7, 8}},
}


def load_x1_map(raw_path: str) -> np.ndarray:
    X4 = config.MAP_SIZE_X * config.MAP_SCALE
    Z4 = config.MAP_SIZE_Z * config.MAP_SCALE
    raw = np.memmap(raw_path, dtype=np.uint32, mode='r', shape=(X4, Z4))
    dl_x4 = extract_driving_line_from_d32(raw)
    dl_x1 = down_scale_map(dl_x4, config.MAP_SCALE, mode='max')  # (17000,10000)

    return dl_x1  # (17000,10000)


# ======================================================================
# 0. PNG マスク → white_mask, nodes
# ======================================================================
def load_mask(png_path: str) -> tuple[np.ndarray, List[Tuple[int, int, int]]]:
    img = Image.open(png_path).convert("RGBA")
    img = img.resize((config.MAP_SIZE_X, config.MAP_SIZE_Z), Image.NEAREST)
    rgba = np.asarray(img)
    r, g, b, a = [rgba[..., i] for i in range(4)]
    alpha = a > 0

    white = alpha & (r >= WHITE_THR[0]) & (g >= WHITE_THR[1]) & (b >= WHITE_THR[2])
    red = alpha & (r >= RED_THR[0]) & (g < RED_THR[1]) & (b < RED_THR[2])

    # 上下反転 → 転置して (row=X, col=Z) へ
    white_mask = white[::-1].T

    red_flip = red[::-1]

    np.save("tmp/intersection_mask.npy", red_flip)  # デバッグ用保存

    lbl, n_lbl = label(red_flip, connectivity=2, return_num=True)
    nodes: List[Tuple[int, int, int]] = []
    for reg in regionprops(lbl):
        cy, cx = reg.centroid
        radius = int(np.ceil(np.sqrt(reg.area / np.pi)))
        nodes.append((int(round(cx)), int(round(cy)), radius))

    return white_mask, nodes  # white_mask:(17000,10000)


# ======================================================================
# 1. RAW → road_mask
# ======================================================================
def road_mask_from_raw(raw_path: str, white_mask: np.ndarray) -> np.ndarray:
    X4 = config.MAP_SIZE_X * config.MAP_SCALE
    Z4 = config.MAP_SIZE_Z * config.MAP_SCALE
    raw = np.memmap(raw_path, dtype=np.uint32, mode='r', shape=(X4, Z4))
    dl_x4 = extract_driving_line_from_d32(raw)
    dl_x1 = down_scale_map(dl_x4, config.MAP_SCALE, mode='max')  # (17000,10000)
    road = (dl_x1 > config.ROAD_CENTER_DISTANCE_THRESHOLD).astype(np.uint8)
    road[white_mask] = 0
    return road  # (17000,10000)


# ======================================================================
# 2. エッジ抽出ユーティリティ
# ======================================================================
DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, sqrt(2)), (-1, 1, sqrt(2)),
        (1, -1, sqrt(2)),  (1, 1, sqrt(2))]


def erase_nodes(arr: np.ndarray,
                nodes: List[Tuple[int, int, int]],
                margin: int = ERASE_MARGIN) -> np.ndarray:
    out = arr.copy()
    h, w = out.shape
    for rx, cz, r in nodes:
        R = r + margin
        x0, x1 = max(rx - R, 0), min(rx + R + 1, h)
        z0, z1 = max(cz - R, 0), min(cz + R + 1, w)
        xx, zz = np.ogrid[x0:x1, z0:z1]
        mask = (xx - rx) ** 2 + (zz - cz) ** 2 <= R ** 2
        out[x0:x1, z0:z1][mask] = 0
    return out


def geodesic_length(coord_set: set[tuple[int, int]],
                    start: tuple[int, int],
                    goal: tuple[int, int]) -> float:
    pq = [(0.0, start)]
    dist = {start: 0.0}
    while pq:
        d, cur = heappop(pq)
        if cur == goal:
            return d
        if dist[cur] < d:
            continue
        x, z = cur
        for dx, dz, c in DIRS:
            nxt = (x+dx, z+dz)
            if nxt not in coord_set:
                continue
            nd = d + c
            if nd < dist.get(nxt, 1e18):
                dist[nxt] = nd
                heappush(pq, (nd, nxt))
    return 0.0


def get_degree_dict(coords: np.ndarray) -> dict[tuple[int, int], int]:
    st = set(map(tuple, coords))
    deg = {}
    for x, z in coords:
        cnt = sum(((x+dx, z+dz) in st) for dx, dz, _ in DIRS)
        deg[(x, z)] = cnt
    return deg


def endpoints(coords: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    deg = get_degree_dict(coords)
    ends = [p for p, d in deg.items() if d == 1]
    if len(ends) >= 2:
        return ends[0], ends[1]
    # ループ
    max_d = -1
    ep0 = ep1 = None
    for a, b in combinations(coords, 2):
        d = (a[0]-b[0])**2 + (a[1]-b[1])**2
        if d > max_d:
            max_d, ep0, ep1 = d, tuple(a), tuple(b)
    return ep0, ep1


def choose_longest_endpair(coords: np.ndarray,
                           max_len_px: int = MAX_ROAD_LEN_PX
                           ) -> tuple[tuple[int, int], tuple[int, int], float]:
    """
    端点ペアのうち最長ジオデシック距離を返す。
    ただし max_len_px を越えた時点で探索を打ち切る。
    """
    deg = get_degree_dict(coords)
    ends = [p for p, d in deg.items() if d == 1]

    # --- 端点が無い (= ループ路) ------------------------------
    if len(ends) < 2:
        ep0, ep1 = endpoints(coords)
        g_len = geodesic_length(set(map(tuple, coords)), ep0, ep1)
        return ep0, ep1, g_len

    # --- 端点がある場合 --------------------------------------
    best_len = -1.0
    best_pair = (ends[0], ends[1])
    coord_set = set(map(tuple, coords))
    for a, b in combinations(ends, 2):
        g = geodesic_length(coord_set, a, b)

        # 進捗表示
        print(f"[choose] try ({a}->{b})  geodesic={g:.1f}px")

        # 打ち切り判定
        if g > max_len_px:
            print(f"    └─ exceed {max_len_px}px → break")
            break

        if g > best_len:
            best_len = g
            best_pair = (a, b)

    return best_pair[0], best_pair[1], best_len


def extract_edges(skeleton_img: np.ndarray,
                  node_centers: List[Tuple[int, int, int]],
                  save_dir: str) -> np.ndarray:
    os.makedirs(save_dir, exist_ok=True)
    sk = erase_nodes(skeleton_img, node_centers)
    labeled, n_cc = label(sk, connectivity=2, return_num=True)
    edges = []
    for lab in range(1, n_cc+1):
        print(f"[edges] connected-component {lab}/{n_cc}")
        coords = np.column_stack(np.where(labeled == lab))
        if coords.size < 3:           # ★ 画素が少なすぎる成分をスキップ
            continue
        p0, p1, length_px = choose_longest_endpair(coords)

        # ★ 端点が取れなかった場合はスキップ
        if p0 is None or p1 is None:
            print("    └─ skip: no valid endpoints")
            continue

        if (p1[0]**2+p1[1]**2) < (p0[0]**2+p0[1]**2):
            p0, p1 = p1, p0
        eid = len(edges)
        np.save(os.path.join(save_dir, f"edge_{eid}.npy"),
                coords.astype(np.uint16))
        edges.append([eid, length_px, tuple(p0), tuple(p1)])
    return np.array(edges, dtype=object)


# -----------------------------------------------------------------------------
# 3. Graph 構築
# -----------------------------------------------------------------------------
def nearest_node(nodes: List[Tuple[int, int, int]], pt: Tuple[int, int]) -> int:
    arr = np.array([(n[0], n[1]) for n in nodes])
    d2 = ((arr[:, 0]-pt[0])**2 + (arr[:, 1]-pt[1])**2)
    return int(d2.argmin())


def build_graph(nodes: List[Tuple[int, int, int]],
                edges: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    for nid, (rx, cz, _) in enumerate(nodes):
        G.add_node(nid, array=(rx, cz))
    for eid, length_px, p0, p1 in edges:
        n0 = nearest_node(nodes, p0)
        n1 = nearest_node(nodes, p1)
        if n0 == n1:
            continue
        length_m = length_px * PX2M
        if G.has_edge(n0, n1):
            if length_m < G[n0][n1]['length']:
                G[n0][n1]['length'] = length_m
        else:
            G.add_edge(n0, n1, length=length_m, edge_id=int(eid))
    return G


def build_movement_graph(G: nx.Graph,
                         turn_rules: dict[int, dict[int, set[int]]]) -> nx.DiGraph:
    """ターン制限付き有向グラフ(半エッジ単位)を生成"""
    M = nx.DiGraph()

    # --- ① 半エッジ(状態)をノードとして追加 -----------------
    for u, v, data in G.edges(data=True):
        eid = data['edge_id']
        L = data['length'] / 2
        M.add_node((eid, u), end=v, length=L)
        M.add_node((eid, v), end=u, length=L)

    # --- ② 交差点内ターン遷移 --------------------------------
    for n in G.nodes():
        inc = list(G.edges(n, data=True))
        for e_in in inc:
            eid_in = e_in[2]['edge_id']
            src = (eid_in, n)
            allow = turn_rules.get(n, {}).get(eid_in, None)
            for e_out in inc:
                eid_out = e_out[2]['edge_id']
                if eid_out == eid_in:    # Uターン禁止
                    continue
                if allow is not None and eid_out not in allow:
                    continue
                dst = (eid_out, n)
                M.add_edge(src, dst, weight=0.0)  # ターン遷移（交差点内）→ 重みなし

    # --- ③ 半エッジ移動(区間走行) -----------------------------
    for state, attr in M.nodes(data=True):
        eid, n_from = state
        n_to = attr['end']
        M.add_edge(state, (eid, n_to), weight=attr['length'])  # 区間走行 → 実距離（メートル）を重みとして使用

    return M


# -----------------------------------------------------------------------------
# 4. 可視化
# -----------------------------------------------------------------------------
def visualize_edges(skeleton: np.ndarray,
                    edges: np.ndarray,
                    nodes: List[Tuple[int, int, int]],
                    out_png: str):

    # --- ① 画像を 90°CCW 回転 -------------------------------
    img = np.rot90(skeleton, k=1)          # shape = (10000,17000)

    fig, ax = plt.subplots(figsize=(50, 30))
    ax.imshow(img, cmap='gray', origin='upper')   # ← 転置も[::-1] も不要

    # 横軸 (=col) 幅は skeleton.shape[0] = 17000
    img = np.rot90(skeleton, k=1)          # shape = (10000,17000)
    H, W = img.shape                       # H=10000, W=17000

    # ② ★ オーバーレイ用の座標変換を 180°反転版に変更 ★
    def to_img_coord(rx: int, cz: int) -> Tuple[int, int]:
        """
        (row=X, col=Z)  →  背景に対して 180°反転した表示座標
        背景(90°CCW)での位置 (x'=W-1-rx, y'=cz) を
        (x,y)=(W-1-x', H-1-y') と反転すると
        最終的に  (x = rx        , y = H-1-cz)
        """
        return rx, H - 1 - cz

    # --- 変更後：距離[m] に応じたグラデーション ----------------
    lengths = np.array([length for _, length, *_ in edges], dtype=float)
    cmap = cm.get_cmap('jet')                          # jet 風
    norm = mcolors.Normalize(vmin=lengths.min(), vmax=lengths.max())

    for eid, length, p0, p1 in edges:
        x0, y0 = to_img_coord(*p0)
        x1, y1 = to_img_coord(*p1)
        color = cmap(norm(length))
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=1)

        xm, ym = (x0+x1)/2, (y0+y1)/2
        ax.text(xm, ym, f"{eid}", color=color, fontsize=3,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))
        # エッジ長さを ID の下に描画(px単位)
        ax.text(xm, ym + 20, f"{length:.1f}px", color='white', fontsize=3,
                ha='center', va='center')

    for nid, (rx, cz, _) in enumerate(nodes):
        xn, yn = to_img_coord(rx, cz)
        ax.plot(xn, yn, 'ro', markersize=3)
        ax.text(xn, yn-3, str(nid), color='yellow', fontsize=3, ha='center')

    ax.set_title("Manual Road Graph (edge & node IDs)")
    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[IMG] {out_png} を保存しました。")


def save_skeleton_image(skeleton: np.ndarray, out_path: str):
    """skeleton画像を可視化用に等倍で保存する

    Args:
        skeleton (np.ndarray): 2値のskeleton画像(dtype=uint8)
        out_path (str): 保存先のファイルパス(.png推奨)

    """
    img = (skeleton * 255).astype(np.uint8)   # 0 or 255 に変換
    Image.fromarray(np.rot90(img, k=1)).save(out_path)
    print(f"[IMG] skeleton を保存しました → {out_path}")


# -----------------------------------------------------------------------------
# 5. route -> driveline mask
# -----------------------------------------------------------------------------
def make_route_mask(x1_map: np.ndarray,
                    edge_ids: List[int],
                    edge_dir: str) -> np.ndarray:
    mask = np.zeros_like(x1_map, dtype=np.uint8)
    coords = np.zeros_like(x1_map, dtype=np.uint8)

    for eid in edge_ids:
        edge = np.load(os.path.join(edge_dir, f"edge_{eid}.npy"))
        coords[edge[:, 0], edge[:, 1]] = 1

    return coords


def print_connected_edges(G: nx.Graph, node_id: int):
    """指定ノードに接続するエッジをすべて表示する

    Args:
        G (nx.Graph): 構築済みグラフ
        node_id (int): node ID

    Raises:
        KeyError: ノードが存在しない場合
    """
    if node_id not in G:
        raise KeyError(f"ノード {node_id} はグラフに存在しません")

    print(f"[Node {node_id}] に接続しているエッジ:")
    for neighbor in G.neighbors(node_id):
        data = G[node_id][neighbor]
        eid = data.get('edge_id', '?')
        length = data.get('length', '?')
        print(f"  → Node {neighbor} (edge_id={eid}, length={length:.1f})")


def print_movements_at_node(MG: nx.DiGraph, node_id: int) -> None:
    """Movement-Graph で ─ 交差点 node_id 内の遷移一覧を表示する

    Movement-Graph のノードは  (edge_id, junction_id) のタプル。
    - 入口半エッジ  (eid_in , node_id)
    - 出口半エッジ  (eid_out, node_id)
    同一 junction_id を共有するノード間に張られた
    weight=0 のエッジが「ターン許可」を表す。

    Args:
        MG (nx.DiGraph): build_movement_graph() で構築した有向グラフ
        node_id (int)  : 対象交差点 ID
    Raises:
        KeyError: node_id が 1 つも存在しない場合
    """
    # ① junction node_id をもつ半エッジを列挙
    states = [s for s in MG.nodes if s[1] == node_id]
    if not states:
        raise KeyError(f"junction {node_id} は Movement-Graph に存在しません")

    print(f"[Junction {node_id}] ターン許可一覧")
    for s_in in states:
        eid_in = s_in[0]
        # weight==0 → 交差点内遷移
        out_eids = sorted(
            {s_out[0] for s_out in MG.successors(s_in) if MG[s_in][s_out]['weight'] == 0}
        )
        if out_eids:
            outs = ", ".join(map(str, out_eids))
            print(f"  ▸ IN  edge {eid_in:>4}  →  OUT {{{outs}}}")
        else:
            print(f"  ▸ IN  edge {eid_in:>4}  →  (遷移禁止)")


# %%
if __name__ == "__main__":
    os.makedirs(os.path.dirname(LABEL_IMG), exist_ok=True)
    os.makedirs(EDGE_DIR, exist_ok=True)

    road_x1 = load_x1_map(RAW_DAT)

    white_mask, nodes = load_mask(MASK_PNG)
    road_mask = road_mask_from_raw(RAW_DAT, white_mask)
    skeleton = skeletonize(road_mask).astype(np.uint8)
    save_skeleton_image(skeleton, "img/skeleton_eqscale.png")

    edges = extract_edges(skeleton, nodes, EDGE_DIR)
    G = build_graph(nodes, edges)

    # 保存
    with open(OUT_PKL, "wb") as f:
        pickle.dump(G, f)
    print(f"[PKL] {OUT_PKL} (nodes={G.number_of_nodes()}, edges={G.number_of_edges()})")

    visualize_edges(skeleton, edges, nodes, LABEL_IMG)

    # --- 例：ノード 0 → 3 をルーティングしてドライビングライン PNG 生成 ---
    src, dst = 54, 75

    # ---------------- Movement Graph 作成 ------------------------
    MG = build_movement_graph(G, TURN_RULES)

    # 出発・到着を半エッジで指定
    start_state = next(((eid, src) for _, _, eid in
                        (G.edges(src, data='edge_id'))), None)
    goal_state = next(((eid, dst) for _, _, eid in
                       (G.edges(dst, data='edge_id'))), None)

    start_state = (24, 54)
    goal_state = (14, 75)
    path_states = nx.shortest_path(MG, start_state, goal_state, weight='weight')

    # 通過エッジ列を抽出
    edge_ids = []
    edge_ids.append(start_state[0])  # 出発エッジを追加
    for s1, s2 in zip(path_states[:-1], path_states[1:]):
        if s1[0] != s2[0]:
            edge_ids.append(s2[0])

    print("[ROUTE] edges:", edge_ids)

    route_mask = make_route_mask(road_mask, edge_ids, EDGE_DIR)
    route_mask = down_scale_map(route_mask, 4, mode='max')
    route_mask = np.kron(route_mask, np.ones((4, 4), dtype=route_mask.dtype))
    routed_road = road_x1*route_mask

    np.save('tmp/routed_road.npy', routed_road)
    Image.fromarray((np.rot90(routed_road, k=1)).astype(np.uint8)).save('tmp/temp.png')
# %%
MG = build_movement_graph(G, TURN_RULES)

# 出発・到着を半エッジで指定
start_state = next(((eid, src) for _, _, eid in
                    (G.edges(src, data='edge_id'))), None)
goal_state = next(((eid, dst) for _, _, eid in
                   (G.edges(dst, data='edge_id'))), None)

start_state = (87, 72)
goal_state = (14, 75)
path_states = nx.shortest_path(MG, start_state, goal_state, weight='weight')

# 通過エッジ列を抽出
edge_ids = []
edge_ids.append(start_state[0])  # 出発エッジを追加
for s1, s2 in zip(path_states[:-1], path_states[1:]):
    if s1[0] != s2[0]:
        edge_ids.append(s2[0])

print("[ROUTE] edges:", edge_ids)

route_mask = make_route_mask(road_mask, edge_ids, EDGE_DIR)
route_mask = down_scale_map(route_mask, 4, mode='max')
route_mask = np.kron(route_mask, np.ones((4, 4), dtype=route_mask.dtype))
routed_road = road_x1*route_mask

np.save('tmp/routed_road.npy', routed_road)
Image.fromarray((np.rot90(routed_road, k=1)).astype(np.uint8)).save('tmp/temp.png')
# %%
