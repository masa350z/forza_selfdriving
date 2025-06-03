# %%
from __future__ import annotations

from typing import List, Tuple, Optional
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from typing import List, Tuple
import os
import pickle

import numpy as np
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from heapq import heappush, heappop
from itertools import combinations
from math import sqrt

import config
from modules import (
    down_scale_map, extract_driving_line_from_d32
)

# ---------- パス設定 ----------
MASK_PNG = "img/map_mask.png"                       # 透明+白+赤 の PNG
RAW_DAT = "map/road_map_x4.dat"            # 32-bit RAW (read-only)
EDGE_DIR = "map/road_edges"          # エッジ座標保存ディレクトリ
LABEL_IMG = "img/manual_edges_labeled.png"   # エッジIDラベル付き画像
GRAPH_MAP_DIR = 'map/graphmap/'
# ---------- しきい値 ----------
RED_THR = (200, 60, 60)
WHITE_THR = (200, 200, 200)
ERASE_MARGIN = 0
MAX_ROAD_LEN_PX = 20000
# -----------------------------

DIRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, sqrt(2)), (-1, 1, sqrt(2)),
        (1, -1, sqrt(2)),  (1, 1, sqrt(2))]


# -----------------------------------------------------------------------------
# デバッグ、可視化ツール
# -----------------------------------------------------------------------------
def visualize_edges(skeleton: np.ndarray,
                    edges: np.ndarray,
                    nodes: List[Tuple[int, int, int]],
                    out_png: str,
                    font_size: int = 8,
                    linewidth: int = 3,
                    node_radius: int = 8,
                    line_only: bool = False,
                    x_range: Optional[Tuple[int, int]] = None,
                    z_range: Optional[Tuple[int, int]] = None,
                    short_side_px: int = 10000,
                    dpi: int = 100) -> None:
    """
    skeleton とノード/エッジ ID を可視化（任意範囲＋短辺=10 000px 拡大保存）

    Parameters
    ----------
    skeleton : (X, Z) uint8
    edges    : ndarray  [[eid, length_px, p0, p1], ...]
    nodes    : list[(row, col, r)]
    out_png  : str
    x_range  : (xmin, xmax) 行範囲  [min,max)
    z_range  : (zmin, zmax) 列範囲  [min,max)
    font_size: int | None   テキストの絶対フォントサイズ(px)
    short_side_px : int     出力画像の短辺ピクセル数
    dpi      : int          Figure の DPI
    """
    # ---------- 1. トリミング ---------------------------------------------
    X, Z = skeleton.shape
    x0, x1 = x_range if x_range else (0, X)
    z0, z1 = z_range if z_range else (0, Z)
    if not (0 <= x0 < x1 <= X and 0 <= z0 < z1 <= Z):
        raise ValueError("x_range / z_range が不正です")

    sk_sub = skeleton[x0:x1, z0:z1]            # (Xsub, Zsub)
    img = np.rot90(sk_sub, k=1)                # (Zsub, Xsub) for imshow
    H, W = img.shape                           # 高=Zsub, 幅=Xsub

    # ---------- 2. Figure サイズ計算  (短辺=short_side_px) -----------------
    scale = short_side_px / min(W, H)
    out_W = int(round(W * scale))
    out_H = int(round(H * scale))
    fig_w_in = out_W / dpi
    fig_h_in = out_H / dpi

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])            # 余白ゼロ
    ax.imshow(img, cmap='gray', origin='upper', interpolation='nearest')

    # ---------- 3. 座標変換 -----------------------------------------------
    def to_img(rx: int, cz: int):
        """(row=X, col=Z) → 回転後座標 (描画対象外なら None)"""
        if x0 <= rx < x1 and z0 <= cz < z1:
            return rx - x0, H - 1 - (cz - z0)
        return None

    # ---------- 5. カラーマップ設定 ---------------------------------------
    lens = np.array([l for _, l, *_ in edges], float)
    cmap = cm.get_cmap('jet')
    norm = mcolors.Normalize(vmin=lens.min(), vmax=lens.max())

    # ---------- 6. エッジ --------------------------------------------------
    for eid, length, p0, p1 in edges:
        p0i, p1i = to_img(*p0), to_img(*p1)
        if p0i is None or p1i is None:
            continue
        (x0i, y0i), (x1i, y1i) = p0i, p1i
        col = cmap(norm(length))
        ax.plot([x0i, x1i], [y0i, y1i], color=col, linewidth=linewidth)

        if not line_only:
            xm, ym = (x0i + x1i) / 2, (y0i + y1i) / 2
            ax.text(xm, ym, f"{eid}", color=col,
                    fontsize=font_size, ha='center', va='center',
                    bbox=dict(fc='white', ec='none', alpha=.6, pad=1))

            ax.text(xm, ym + 12, f"{length:.1f}px", color='white',
                    fontsize=int(font_size * 0.9), ha='center', va='center')

    # ---------- 7. ノード --------------------------------------------------
    if not line_only:
        for nid, (rx, cz, _) in enumerate(nodes):
            pt = to_img(rx, cz)
            if pt is None:
                continue
            xn, yn = pt
            ax.plot(xn, yn, 'ro', ms=node_radius)
            ax.text(xn, yn - 5, str(nid), color='yellow',
                    fontsize=font_size, ha='center')

    # ---------- 8. 仕上げ --------------------------------------------------
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"[IMG] {out_png} を保存しました "
          f"→ {out_W}×{out_H}px (短辺={short_side_px}px, dpi={dpi})")


def save_skeleton_image(skeleton: np.ndarray, out_path: str):
    """skeleton画像を可視化用に等倍で保存する

    Args:
        skeleton (np.ndarray): 2値のskeleton画像(dtype=uint8)
        out_path (str): 保存先のファイルパス(.png推奨)

    """
    img = (skeleton * 255).astype(np.uint8)   # 0 or 255 に変換
    Image.fromarray(np.rot90(img, k=1)).save(out_path)
    print(f"[IMG] skeleton を保存しました → {out_path}")


def print_connected_edges(G: nx.Graph, node_id: int) -> None:
    """指定ノードに接続するエッジを列挙して表示する。

    Graph でも MultiGraph でも同じ呼び出しで使える。
    """
    if node_id not in G:
        raise KeyError(f"ノード {node_id} はグラフに存在しません")

    print(f"[Node {node_id}] に接続しているエッジ:")

    if isinstance(G, nx.MultiGraph):
        # ── parallel-edge 対応 ──────────────────
        for _, neighbor, key, attr in G.edges(node_id, keys=True, data=True):
            eid = attr.get('edge_id', key)
            length = attr.get('length', float('nan'))
            print(f"  → Node {neighbor} (edge_key={key}, edge_id={eid}, length={length:.1f})")
    else:
        # ── 単一エッジ Graph ────────────────────
        for neighbor, attr in G[node_id].items():
            eid = attr.get('edge_id', '?')
            length = attr.get('length', float('nan'))
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
            {s_out[0] for s_out in MG.successors(s_in) if MG[s_in][s_out].get('turn')}
        )
        if out_eids:
            outs = ", ".join(map(str, out_eids))
            print(f"  ▸ IN  edge {eid_in:>4}  →  OUT {{{outs}}}")
        else:
            print(f"  ▸ IN  edge {eid_in:>4}  →  (遷移禁止)")


# -----------------------------------------------------------------------------
# マップ保存、ロードツール
# -----------------------------------------------------------------------------
def save_graph_map(G, nodes, edges, skeleton, masked_map, output_dir=GRAPH_MAP_DIR):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    graph_path = os.path.join(output_dir + 'graph.pickle')
    node_path = os.path.join(output_dir + 'node.pickle')
    edge_path = os.path.join(output_dir + 'edge.pickle')
    skeleton_path = os.path.join(output_dir + 'skeleton.pickle')
    masked_map_path = os.path.join(output_dir + 'masked_map.pickle')

    save_skeleton_image(skeleton, "img/skeleton_eqscale.png")

    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    with open(node_path, "wb") as f:
        pickle.dump(nodes, f)
    with open(edge_path, "wb") as f:
        pickle.dump(edges, f)
    with open(skeleton_path, "wb") as f:
        pickle.dump(skeleton, f)
    with open(masked_map_path, "wb") as f:
        pickle.dump(masked_map, f)


def save_movement_graph_map(MG, output_dir=GRAPH_MAP_DIR):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    movement_graph = os.path.join(output_dir + 'movement_graph.pickle')

    with open(movement_graph, "wb") as f:
        pickle.dump(MG, f)


def load_graph_map(base_dir=GRAPH_MAP_DIR):
    graph_path = os.path.join(base_dir + 'graph.pickle')
    node_path = os.path.join(base_dir + 'node.pickle')
    edge_path = os.path.join(base_dir + 'edge.pickle')
    skeleton_path = os.path.join(base_dir + 'skeleton.pickle')

    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    with open(node_path, 'rb') as f:
        nodes = pickle.load(f)
    with open(edge_path, 'rb') as f:
        edges = pickle.load(f)
    with open(skeleton_path, 'rb') as f:
        skeleton = pickle.load(f)

    return G, nodes, edges, skeleton


def load_movement_graph_map(base_dir=GRAPH_MAP_DIR):
    movement_graph_path = os.path.join(base_dir + 'movement_graph.pickle')

    with open(movement_graph_path, 'rb') as f:
        MG = pickle.load(f)

    return MG

# ======================================================================
# 1. RAWマップデータから、交差点や修正マスクを適用したdrivinglineマップ作成
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

    np.save('map/intersection_mask.npy', red_flip.T)

    lbl, n_lbl = label(red_flip, connectivity=2, return_num=True)
    nodes: List[Tuple[int, int, int]] = []
    for reg in regionprops(lbl):
        cy, cx = reg.centroid
        radius = int(np.ceil(np.sqrt(reg.area / np.pi)))
        nodes.append((int(round(cx)), int(round(cy)), radius))

    return white_mask, nodes  # white_mask:(17000,10000)


def drivingline_mask_from_raw(raw_path: str, white_mask: np.ndarray) -> np.ndarray:
    X4 = config.MAP_SIZE_X * config.MAP_SCALE
    Z4 = config.MAP_SIZE_Z * config.MAP_SCALE
    raw = np.memmap(raw_path, dtype=np.uint32, mode='r', shape=(X4, Z4))
    dl_x4 = extract_driving_line_from_d32(raw)
    dl_x1 = down_scale_map(dl_x4, config.MAP_SCALE, mode='max')  # (17000,10000)
    drivingline_x1 = dl_x1.copy()
    masked_drivingline_x1 = (dl_x1 > config.ROAD_CENTER_DISTANCE_THRESHOLD).astype(np.uint8)
    masked_drivingline_x1[white_mask] = 0

    return masked_drivingline_x1, drivingline_x1  # (17000,10000)


# ======================================================================
# 2. エッジ（道路）、ノード（交差点）抽出
# ======================================================================
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
        if coords.size < 2:           # ★ 画素が少なすぎる成分をスキップ
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
    G = nx.MultiGraph()                       # ★ ここを MultiGraph に
    # ---- ノード追加 -------------------------------------------------
    for nid, (rx, cz, _) in enumerate(nodes):
        G.add_node(nid, array=(rx, cz))

    # ---- エッジ追加 -------------------------------------------------
    for eid, length_px, p0, p1 in edges:
        n0 = nearest_node(nodes, p0)
        n1 = nearest_node(nodes, p1)
        if n0 == n1:
            continue

        # edge_key = eid としてそのまま登録
        G.add_edge(
            n0, n1,
            key=int(eid),
            edge_id=int(eid),
            length=length_px
        )
    return G


def build_movement_graph(
        G: nx.Graph,
        turn_rules: dict[int, dict[int, set[int]]],
        junction_cost: float | dict[int, float] | None = None) -> nx.DiGraph:
    """
    半エッジ単位の Movement-Graph を生成する。

    Parameters
    ----------
    G : nx.Graph
        build_graph() で作った道路グラフ
    turn_rules : dict
        {junction_id: {in_eid: {out_eid, …}, …}}
    junction_cost : float | dict[int, float] | None, optional
        ・float  : すべての交差点に同じコストを課す
        ・dict   : {junction_id: cost, …} で個別設定
        ・None   : 追加コストなし（従来互換）
    """
    # ------------------------------------------------------------------
    # 0) junction_cost をアクセスしやすい dict にそろえる
    # ------------------------------------------------------------------
    if junction_cost is None:
        jc = {}
    elif isinstance(junction_cost, (int, float)):
        jc = {n: float(junction_cost) for n in G.nodes()}
    elif isinstance(junction_cost, dict):
        jc = {int(k): float(v) for k, v in junction_cost.items()}
    else:
        raise TypeError("junction_cost は float / dict / None のいずれか")

    M = nx.DiGraph()

    # ------------------------------------------------------------------
    # ① 半エッジ(状態)ノード
    # ------------------------------------------------------------------
    for u, v, data in G.edges(data=True):
        eid = data["edge_id"]
        half_len = data["length"] / 2          # 区間を２分割した距離
        M.add_node((eid, u), end=v, length=half_len)
        M.add_node((eid, v), end=u, length=half_len)

    # ------------------------------------------------------------------
    # ② 交差点内ターン遷移
    #    weight = junction_cost[その交差点]   (設定なければ 0)
    # ------------------------------------------------------------------
    for n in G.nodes():
        inc = list(G.edges(n, data=True))
        j_cost = jc.get(n, 0.0)

        for e_in in inc:
            eid_in = e_in[2]["edge_id"]
            src = (eid_in, n)
            allow = turn_rules.get(n, {}).get(eid_in)  # None → 全許可

            for e_out in inc:
                eid_out = e_out[2]["edge_id"]
                if eid_out == eid_in:
                    continue                    # U ターン禁止
                if allow is not None and eid_out not in allow:
                    continue

                dst = (eid_out, n)
                # ★ weight を j_cost にする（距離 0 + 交差点コスト）
                M.add_edge(src, dst,
                           weight=j_cost,
                           turn=True)

    # ------------------------------------------------------------------
    # ③ 半エッジ移動(区間走行) ：weight=実距離
    # ------------------------------------------------------------------
    for state, attr in M.nodes(data=True):
        eid, n_from = state
        n_to = attr["end"]
        M.add_edge(state, (eid, n_to), weight=attr["length"])

    return M


# -----------------------------------------------------------------------------
# 4. ルーティングされた部分だけをマスクしたマップ作成
# -----------------------------------------------------------------------------
def extract_routed_edges(MG, start_state, goal_state):
    path_states = nx.shortest_path(MG, start_state, goal_state, weight='weight')

    # 通過エッジ列を抽出
    edge_ids = []
    edge_ids.append(start_state[0])  # 出発エッジを追加
    for s1, s2 in zip(path_states[:-1], path_states[1:]):
        if s1[0] != s2[0]:
            edge_ids.append(s2[0])

    print("[ROUTE] edges:", edge_ids)

    return edge_ids


def concat_driveline(x1_map: np.ndarray,
                     edge_ids: List[int],
                     edge_dir: str) -> np.ndarray:

    driveline = np.zeros_like(x1_map, dtype=np.uint8)

    for eid in edge_ids:
        edge = np.load(os.path.join(edge_dir, f"edge_{eid}.npy"))
        driveline[edge[:, 0], edge[:, 1]] = 1

    return driveline


def ret_routed_drivingline(drivingline_x1, masked_drivingline_x1,
                           MG, start_state, goal_state, output_path='tmp/routed_road.npy'):
    edge_ids = extract_routed_edges(MG, start_state, goal_state)
    routed_drivingline = concat_driveline(masked_drivingline_x1, edge_ids, EDGE_DIR)

    routed_drivingline = down_scale_map(routed_drivingline, 4, mode='max')
    routed_drivingline = np.kron(routed_drivingline, np.ones((4, 4), dtype=routed_drivingline.dtype))
    routed_drivingline = drivingline_x1*routed_drivingline

    np.save(output_path, routed_drivingline)
    Image.fromarray((np.rot90(routed_drivingline, k=1)).astype(np.uint8)).save(output_path.replace('.npy', '.png'))

    return routed_drivingline


# %%
if __name__ == "__main__":
    os.makedirs(os.path.dirname(LABEL_IMG), exist_ok=True)
    os.makedirs(EDGE_DIR, exist_ok=True)

    # white_mask, nodes = load_mask(MASK_PNG)
    # masked_drivingline_x1, drivingline_x1 = drivingline_mask_from_raw(RAW_DAT, white_mask)
    # skeleton = skeletonize(masked_drivingline_x1).astype(np.uint8)

    # masked_map_path = os.path.join(GRAPH_MAP_DIR + 'masked_map.pickle')
    # with open(masked_map_path, "wb") as f:
    #     pickle.dump(masked_drivingline_x1, f)

    # edges = extract_edges(skeleton, nodes, EDGE_DIR)

    # -----------------------------------------------------------------------------
    # マップ手動修正
    # -----------------------------------------------------------------------------
    # additional_edges = [((313, 321), 438, 2.0),
    #                     ((205, 208), 586, 7.2),
    #                     ((122, 123), 104, 2.4),
    #                     ((122, 116), 94, 113.2),
    #                     ((123, 121), 103, 1.4),
    #                     ((97, 99), 161, 1324.7)]

    # G = build_graph(nodes, edges)
    # G.remove_edge(123, 116, key=94)
    # for (p0, eid, length) in additional_edges:
    #     G.add_edge(
    #         p0[0], p0[1],
    #         key=int(eid),
    #         edge_id=int(eid),
    #         length=length
    #     )

    # save_graph_map(G, nodes, edges, skeleton)

    TURN_RULES: dict[int, dict[int, set[int]]] = {
        # 例: 交差点 12 では edge 5 から 7,8 へ直進のみ許可
        # 12: {5: {7, 8}},

        # 立体交差
        256: {516: {453}, 453: {516}, 507: {483}, 483: {507}},
        278: {473: {441}, 441: {473}, 470: {462}, 462: {470}},
        296: {444: {421}, 421: {444}, 441: {419}, 419: {441}},
        293: {418: {416}, 416: {418}, 419: {410}, 410: {419}},
        285: {410: {405}, 405: {410}, 409: {394}, 394: {409}},
        308: {392: {390}, 390: {392}, 394: {375}, 375: {394}},
        328: {393: {380}, 380: {393}, 390: {391}, 391: {390}},
        341: {391: {386}, 386: {391}, 397: {389}, 389: {397}},
        348: {377: {369}, 369: {377}, 378: {365}, 365: {378}},
        309: {371: {367}, 367: {371}, 355: {370}, 370: {355}},
        383: {286: {274}, 274: {286}, 285: {276}, 276: {285}},

        78: {119: {117}, 117: {119}, 95: {121}, 121: {95}},
        81: {116: {117}, 117: {116}, 98: {120}, 120: {98}},

        97: {159: {158}, 158: {159}, 133: {161}, 161: {133}},
        100: {158: {153}, 153: {158}, 134: {160}, 160: {134}},

        99: {193: {192}, 193: {192}, 195: {161}, 161: {195}},
        101: {192: {190}, 190: {192}, 194: {160}, 160: {194}},

        109: {220: {218}, 218: {220}, 221: {210}, 210: {221}},
        110: {218: {216}, 216: {218}, 219: {209}, 209: {219}},

        137: {400: {395}, 395: {400}, 399: {264}, 264: {399}},
        140: {344: {395}, 395: {344}, 396: {263}, 263: {396}},

        160: {547: {544}, 544: {547}, 548: {542}, 542: {548}},
        162: {539: {544}, 544: {539}, 545: {541}, 541: {545}},

        174: {581: {579}, 579: {581}, 572: {582}, 582: {572}},
        176: {570: {579}, 579: {570}, 580: {571}, 571: {580}},

        25: {139: {141}, 141: {139}, 142: {129}, 129: {142}},
        16: {141: {136}, 136: {141}, 143: {74}, 74: {143}},

        1: {233: {180}, 180: {233}, 206: {234}, 234: {206}},
        2: {233: {236}, 236: {233}, 207: {237}, 237: {207}},

        # # 交差点
        311: {520: {519, 514}, 515: {520, 519}, 518: {520, 519, 514}, 514: {}},
        310: {456: {515, 512}, 514: {456, 512}, 513: {456, 515, 512}, 515: {}},
        319: {517: {513}, 519: {517}, 513: {}},
        305: {508: {518}, 512: {508}, 518: {}},

        259: {483: {478}, 474: {483}, 478: {}},
        264: {484: {481}, 482: {484}, 481: {}},
        269: {466: {468}, 472: {466}, 468: {}},
        260: {458: {467}, 465: {458}, 467: {}},
        262: {478: {482, 477}, 475: {482, 477}, 482: {}, 477: {}},
        265: {477: {472, 471}, 481: {472, 471}, 471: {}, 472: {}},
        263: {471: {465, 469}, 468: {465, 469}, 465: {}, 469: {}},
        261: {467: {474, 475}, 469: {474, 475}, 475: {}, 474: {}},

        241: {290: {488}, 489: {290}, 488: {}},
        243: {493: {489}, 494: {489}, 489: {}},
        246: {503: {494, 499}, 499: {503}, 494: {}},
        248: {507: {503, 504}, 503: {507}, 504: {507}},
        240: {488: {491, 492}, 491: {}, 492: {}},
        244: {500: {493, 499}, 499: {497}, 497: {499, 493}, 493: {}},
        242: {505: {500, 504}, 504: {505}, 500: {}},
        245: {510: {505, 506}, 506: {510}, 505: {510}},
        239: {492: {497, 498, 495}, 495: {497}, 497: {495}, 498: {}},
        238: {490: {506}, 501: {506}, 506: {}},
        236: {491: {496}, 495: {496}, 496: {495}},
        233: {502: {501, 496}, 501: {502}, 496: {502}},

        357: {463: {460}, 459: {463}, 460: {}},
        371: {461: {452}, 457: {461}, 452: {}},
        364: {361: {443}, 445: {361}, 443: {}},
        354: {437: {448}, 447: {437}, 448: {}},
        363: {460: {457, 451}, 455: {457, 451}, 451: {}, 457: {}},
        365: {452: {445, 450}, 451: {445, 450}, 450: {}, 445: {}},
        360: {450: {447, 449}, 443: {447, 449}, 447: {}, 449: {}},
        358: {448: {459, 455}, 449: {459, 455}, 459: {}, 455: {}},

        314: {456: {442}, 440: {456}, 442: {}},
        321: {442: {436, 433}, 438: {436, 433}, 433: {}},
        313: {434: {440, 438}, 432: {440, 434, 438}, 438: {}},
        318: {433: {426, 432}, 426: {432}, 432: {}},

        359: {354: {348}, 347: {354}, 348: {}},
        366: {348: {345}, 346: {345}, 345: {}},
        369: {338: {341}, 345: {341, 338}, 341: {}},
        368: {341: {337, 339}, 337: {}, 339: {}},
        370: {330: {336}, 337: {330}, 336: {}},
        367: {336: {340}, 367: {340}, 340: {}},
        362: {331: {342}, 340: {331, 342}, 342: {}},
        361: {342: {347, 346}, 347: {}, 346: {}},

        373: {330: {328}, 320: {330}, 328: {}},
        377: {328: {327, 323}, 321: {327, 323}, 327: {}, 323: {}},
        380: {327: {307}, 307: {324}, 324: {}},
        379: {324: {310, 313}, 323: {310, 313}, 310: {}, 313: {}},
        382: {310: {293}, 293: {309}, 309: {}},
        376: {313: {316, 312}, 309: {316, 312}, 316: {}, 312: {}},
        372: {303: {315}, 312: {303}, 315: {}},
        374: {315: {320, 321}, 316: {320, 321}, 320: {}, 321: {}},

        211: {634: {630}, 631: {630, 634}, 630: {}},
        212: {630: {627, 628}, 627: {628}, 628: {}},
        207: {628: {629, 623}, 623: {629}, 629: {}},
        206: {629: {632, 631}, 632: {631}, 631: {}},

        92: {611: {607}, 608: {611}, 607: {}},
        90: {607: {601}, 606: {601}, 601: {}},
        88: {609: {608, 606}, 608: {}, 606: {}},
        93: {589: {596}, 597: {589}, 596: {}},
        86: {596: {592}, 598: {592}, 592: {}},
        91: {601: {597, 598}, 597: {}, 598: {}},
        74: {590: {594}, 593: {590}, 594: {}},
        73: {594: {600}, 595: {600}, 600: {}},
        77: {592: {595, 590}, 590: {}, 595: {}},
        69: {612: {610}, 605: {612}, 610: {}},
        76: {610: {609}, 604: {609}, 609: {}},
        72: {600: {605, 604}, 605: {}, 604: {}},

        41: {59: {54}, 57: {59}, 54: {}},
        38: {54: {49}, 55: {49}, 49: {}},
        37: {58: {57, 55}, 57: {}, 55: {}},
        35: {49: {44, 47}, 47: {}, 44: {}},
        31: {44: {32}, 32: {46}, 46: {}},
        30: {46: {50}, 47: {50}, 50: {}},
        27: {50: {53, 52}, 53: {}, 52: {}},
        29: {53: {58}, 56: {58}, 58: {}},
        26: {60: {56}, 52: {60}, 56: {}},
        22: {61: {63}, 63: {60, 61}, 60: {63}},

        0: {543: {237}, 234: {543}},
        6: {205: {206}, 207: {205}},

        68: {61: {95, 97}, 95: {}, 97: {}},
        66: {97: {112, 111}, 112: {}, 111: {}},
        72: {122: {119, 123}, 112: {119, 123}, 119: {123, 122}, 123: {}},
        75: {125: {127}, 123: {127}, 127: {}},
        85: {127: {133}, 121: {133}, 133: {}},
        62: {124: {125, 122}, 122: {124}, 111: {124}, 125: {}},
        87: {134: {126, 120}, 126: {}, 120: {}},
        89: {126: {115, 118}, 115: {}, 118: {}},
        83: {118: {110, 116}, 116: {114, 110}, 114: {110, 116}, 110: {}},
        82: {110: {96}, 109: {96}, 96: {}},
        70: {96: {59}, 98: {59}, 59: {}},
        95: {93: {109, 114}, 115: {93}, 114: {93}, 109: {}},

        107: {195: {210, 211}, 210: {}, 211: {}},
        105: {211: {220, 228, 229}, 220: {228, 229}, 228: {220, 229}, 229: {}},
        112: {229: {264}, 221: {264}, 264: {}},
        112: {263: {217, 219}, 217: {}, 219: {}},
        114: {217: {212, 208, 216}, 216: {208, 212}, 212: {216, 208}, 208: {}},
        108: {208: {194}, 209: {194}, 194: {}},

        151: {399: {476, 480}, 476: {}, 480: {}},
        147: {567: {566, 554}, 480: {567}, 554: {567}, 566: {}},
        171: {566: {572}, 557: {572}, 572: {}},
        172: {571: {528, 556}, 528: {}, 556: {}},
        173: {528: {521}, 526: {521}, 521: {526, 485}, 485: {}},
        152: {485: {396}, 486: {396}, 396: {}},
        158: {476: {542}, 534: {542}, 542: {}},
        163: {548: {557, 555}, 555: {}, 557: {}},
        156: {547: {534, 554}, 555: {547}, 554: {547}, 534: {}},
        165: {556: {545}, 537: {545}, 545: {}},
        167: {539: {537, 526}, 526: {539}, 527: {539}, 537: {}},
        159: {541: {486, 527}, 527: {}, 486: {}},

        190: {615: {580}, 582: {615}, 580: {}},
    }

    G, nodes, edges, skeleton = load_graph_map()
    MG = load_movement_graph_map()

    # MG = build_movement_graph(G, TURN_RULES, junction_cost=15)
    # save_movement_graph_map(MG)

    # start_state = (136, 16)
    # goal_state = (667, 177)

    # routed_drivingline = ret_routed_drivingline(drivingline_x1, masked_drivingline_x1,
    #                                             MG, start_state, goal_state, output_path='tmp/routed_road.npy')
