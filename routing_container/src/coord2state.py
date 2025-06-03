"""Forza 座標 → Movement-Graph の半エッジ状態を返すユーティリティ。

       * coord_to_state(x,z) -> (edge_id, junction_id)
"""

from __future__ import annotations

from typing import Tuple, List
import pickle
import pathlib
import numpy as np

import config

# ---------- データロード（起動時 1 度だけ） ---------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"
with open(ROOT / "graphmap" / "node.pickle", "rb") as f:
    _NODES: List[Tuple[int, int, int]] = pickle.load(f)
with open(ROOT / "graphmap" / "edge.pickle", "rb") as f:
    _EDGES = pickle.load(f)

_EDGES_DICT = {int(eid): (tuple(p0), tuple(p1)) for eid, _, p0, p1 in _EDGES}
_NEAREST_MAP = np.load(ROOT / "nearest_edge_map_x1.npy")


# ---------- 内部関数 ---------------------------------------------------
def _nearest_node(nodes: List[Tuple[int, int, int]],
                  pt: Tuple[int, int]) -> int:
    arr = np.array([(n[0], n[1]) for n in nodes])
    d2 = ((arr[:, 0] - pt[0])**2 + (arr[:, 1] - pt[1])**2)
    return int(d2.argmin())


def _nearest_edge_and_junction(array_x: int, array_z: int
                               ) -> Tuple[int, int]:
    """配列座標 → 最近接 edge_id と junction_id"""
    eid = int(_NEAREST_MAP[array_x, array_z])

    p0, p1 = _EDGES_DICT[eid]
    d0 = (array_x - p0[0]) ** 2 + (array_z - p0[1]) ** 2
    d1 = (array_x - p1[0]) ** 2 + (array_z - p1[1]) ** 2
    nearer_pt = p0 if d0 < d1 else p1

    junc_id = _nearest_node(_NODES, nearer_pt)
    return eid, junc_id


# ---------- 公開関数 ---------------------------------------------------
def coord_to_state(forza_x: float, forza_z: float
                   ) -> Tuple[int, int]:
    """Forza 座標系 → 半エッジ状態 (edge_id, junction_id)

    Raises
    ------
    ValueError : マップ外座標
    """
    array_x = int(forza_x + config.MAP_OFFSET_X)
    array_z = int(forza_z + config.MAP_OFFSET_Z)

    if not (0 <= array_x < config.MAP_SIZE_X and
            0 <= array_z < config.MAP_SIZE_Z):
        raise ValueError("座標がマップ範囲外です")

    return _nearest_edge_and_junction(array_x, array_z)
