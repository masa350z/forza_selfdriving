"""任意座標 → start_state / goal_state へ変換するユーティリティ。

Functions
---------
coord_to_state(...) : Forza 座標/配列座標 → (edge_id, junction_id)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

import config
from make_graph_map import load_graph_map, nearest_node


# --- グラフメタを一度ロードしてキャッシュ -----------------------------
_G, _NODES, _EDGES, _ = load_graph_map()
_EDGES_DICT = {int(eid): (tuple(p0), tuple(p1)) for eid, _, p0, p1 in _EDGES}


def _nearest_edge_and_junction(array_x: int, array_z: int,
                               nearest_edge_map: np.ndarray) -> Tuple[int, int]:
    """配列座標から最近接 edge_id と最寄り端点の junction_id を返す"""
    eid = int(nearest_edge_map[array_x, array_z])

    p0, p1 = _EDGES_DICT[eid]
    d0 = (array_x - p0[0]) ** 2 + (array_z - p0[1]) ** 2
    d1 = (array_x - p1[0]) ** 2 + (array_z - p1[1]) ** 2
    nearer_pt = p0 if d0 < d1 else p1

    junc_id = nearest_node(_NODES, nearer_pt)
    return eid, junc_id


# ──────────────────────────────────────────────────────────────────────
def coord_to_state(forza_x: float, forza_z: float,
                   nearest_edge_map: np.ndarray) -> Tuple[int, int]:
    """Forza 座標から Movement-Graph 用 state を返す

    Args:
        forza_x (float): Forza 座標系 X(m)
        forza_z (float): Forza 座標系 Z(m)
        nearest_edge_map (np.ndarray): `edge_map_builder.py` で作成した最近接エッジマップ(x1)

    Returns:
        (edge_id, junction_id): Movement-Graph の半エッジ状態
    """
    array_x = int((forza_x + config.MAP_OFFSET_X))         # x1 等倍
    array_z = int((forza_z + config.MAP_OFFSET_Z))
    if not (0 <= array_x < config.MAP_SIZE_X and 0 <= array_z < config.MAP_SIZE_Z):
        raise ValueError("座標がマップ外です")

    return _nearest_edge_and_junction(array_x, array_z, nearest_edge_map)
