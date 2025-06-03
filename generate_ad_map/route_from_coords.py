"""「現在地座標 → 目的地座標」の入力でルートを生成しrouted_road.npy を出力する。

Usage
-----
$ python route_from_coords.py --start_x -500 --start_z 1200 \
                              --goal_x  3500 --goal_z -2600
"""

from __future__ import annotations

import pickle
import argparse
from pathlib import Path

import numpy as np

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from make_graph_map import (load_graph_map, load_movement_graph_map,
                            ret_routed_drivingline)
from coord2state import coord_to_state


# ---------- パス設定 ----------
MASK_PNG = "../img/map_mask.png"                       # 透明+白+赤 の PNG
RAW_DAT = "../map/road_map_x4.dat"            # 32-bit RAW (read-only)
DRIVELINE_X1 = Path("../map/drivingline_map_x1.npy")
NEAREST_MAP_X1 = Path("../map/nearest_edge_map_x1.npy")
EDGE_DIR = Path("../map/road_edges")
OUT_NPY = Path("../tmp/routed_road.npy")
OUT_PNG = Path("../tmp/routed_road.png")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="任意座標 → ルートマップ生成スクリプト")
    ap.add_argument("--start_x", type=float, required=True, help="Forza X[m]")
    ap.add_argument("--start_z", type=float, required=True, help="Forza Z[m]")
    ap.add_argument("--goal_x", type=float, required=True, help="Forza X[m]")
    ap.add_argument("--goal_z", type=float, required=True, help="Forza Z[m]")
    args = ap.parse_args()

    # --- マップ・グラフをロード ----------------------------------------
    nearest_map = np.load(NEAREST_MAP_X1)

    drivingline_x1 = np.load(DRIVELINE_X1)
    with open('../map/graph../map/masked_map.pickle', 'rb') as f:
        masked_drivingline_x1 = pickle.load(f)

    _, _, _, _ = load_graph_map()          # Graph は内部キャッシュされ coord2state に利用
    MG = load_movement_graph_map()

    # --- 座標 → state へ変換 ------------------------------------------
    start_state = coord_to_state(args.start_x, args.start_z, nearest_map)
    goal_state = coord_to_state(args.goal_x, args.goal_z, nearest_map)

    print(f"[STATE] start={start_state}, goal={goal_state}")

    # --- 経路探索 & ルートマップ生成 -----------------------------------
    routed_drivingline = ret_routed_drivingline(drivingline_x1, masked_drivingline_x1,
                                                MG, start_state, goal_state, output_path='../tmp/routed_road.npy')


if __name__ == "__main__":
    main()
