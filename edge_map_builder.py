"""各ピクセルに「最近接 edge_id」を割り当てた最近接エッジマップを作成する。

       * build_nearest_edge_map_x1(): 等倍 (17000x10000) を生成
       * build_nearest_edge_map_x4(): x4 拡大版を生成（自動運転用）

実行方法
--------
$ python edge_map_builder.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import config
from make_graph_map import load_graph_map  # nodes, edges, skeleton 取得用

EDGE_DIR = Path("map/road_edges")
OUT_X1 = Path("map/nearest_edge_map_x1.npy")
OUT_X4 = Path(f"map/nearest_edge_map_x{config.MAP_SCALE}.npy")


def _build_edge_id_surface(edges: np.ndarray) -> np.ndarray:
    """edge_i.npy 群から「edge_id が書かれた一次マップ」を作成する

    Returns:
        np.ndarray(dtype=int16, shape=(MAP_SIZE_X, MAP_SIZE_Z))
    """
    surf = np.full((config.MAP_SIZE_X, config.MAP_SIZE_Z), -1, np.int16)

    for eid, *_ in edges:
        coords = np.load(EDGE_DIR / f"edge_{eid}.npy")  # shape=(n,2)
        surf[coords[:, 0], coords[:, 1]] = int(eid)

    return surf


def build_nearest_edge_map_x1() -> np.ndarray:
    """最近接 edge_id を全画素へバラ撒いた x1 マップを生成して保存"""
    _, _, edges, _ = load_graph_map()

    # 1) edge_id が書かれた一次マップ
    edge_id_surf = _build_edge_id_surface(edges)

    # 2) 最近接インデックスを distance_transform_edt で取得
    #    入力 True=背景, False=エッジ なので invert
    back = (edge_id_surf == -1)
    _, (ix, iz) = distance_transform_edt(back, return_indices=True)
    nearest_edge_map = edge_id_surf[ix, iz]

    np.save(OUT_X1, nearest_edge_map)
    print(f"[SAVE] 最近接エッジマップ(x1) → {OUT_X1}")

    return nearest_edge_map


def build_nearest_edge_map_x4(nearest_x1: np.ndarray) -> None:
    """x4 拡大マップを保存する"""
    nearest_x4 = np.kron(nearest_x1, np.ones(
        (config.MAP_SCALE, config.MAP_SCALE), np.int16))
    np.save(OUT_X4, nearest_x4)
    print(f"[SAVE] 最近接エッジマップ(x{config.MAP_SCALE}) → {OUT_X4}")


if __name__ == "__main__":
    OUT_X1.parent.mkdir(parents=True, exist_ok=True)
    m_x1 = build_nearest_edge_map_x1()
    build_nearest_edge_map_x4(m_x1)
