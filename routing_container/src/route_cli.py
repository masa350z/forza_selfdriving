"""
標準入出力で経路を返す常駐サービス。

入力 : 1行JSON
        {"start_x": -500, "start_z": 1200,
         "goal_x":  3500, "goal_z": -2600}

出力 : 1行JSON
        {"edges": [136, 140, 287, ...]}

* 無効入力      → {"error": "..."} を返し継続
* EOF (Ctrl-D) → 終了
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
from typing import List

import networkx as nx

from coord2state import coord_to_state

# ──────────────────────────────────────────────────────────────
# データをメモリにロード(1度だけ)
# ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[1] / "data" / "graphmap"
with open(ROOT / "movement_graph.pickle", "rb") as f:
    MG: nx.DiGraph = pickle.load(f)


def shortest_edge_path(sx: float, sz: float, gx: float, gz: float) -> List[int]:
    s_state = coord_to_state(sx, sz)
    g_state = coord_to_state(gx, gz)

    states = nx.shortest_path(MG, s_state, g_state, weight="weight")

    edges: List[int] = [states[0][0]]
    for st in states[1:]:
        if st[0] != edges[-1]:
            edges.append(st[0])
    return edges


# ──────────────────────────────────────────────────────────────
# メインループ
# ──────────────────────────────────────────────────────────────
def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            edges = shortest_edge_path(
                req["start_x"], req["start_z"],
                req["goal_x"], req["goal_z"]
            )
            resp = {"edges": edges}
        except Exception as e:
            resp = {"error": str(e)}

        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
