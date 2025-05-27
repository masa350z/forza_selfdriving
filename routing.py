# %%
from make_quadrant_map import main as make_quadrant_map
import pickle
import networkx as nx
from PIL import Image
import numpy as np
from modules import down_scale_map


def shortest_path_by_distance(
        G: nx.Graph,
        src: int,
        dst: int):
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


def make_routed_x1_driveline(x1_map, road_edge_indexes, array_shape=(4250, 2500)):
    road_edge_map = np.zeros(array_shape, dtype=np.uint16)

    for i in road_edge_indexes:
        road_edge = np.load(f'map/road_edges/edge_{i}.npy')
        road_edge_map[road_edge[:, 0], road_edge[:, 1]] = 1

    div5_driving_line = down_scale_map(road_edge_map, 5, mode='max')

    x1_driveline = x1_map*div5_driving_line.repeat(20, axis=0).repeat(20, axis=1)

    return x1_driveline


def main(G, route_start_node, route_end_node):
    nodes, dist = shortest_path_by_distance(G, src=route_start_node,
                                            dst=route_end_node)
    road_edge_indexes = route_nodes_to_edges(nodes, G)
    print("edge_ids:", road_edge_indexes)
    print("nodes:", nodes, "edges:", road_edge_indexes, "distance[m]:", dist)

    x1_map = np.load('map/palacio_simple_x1.npy')
    routed_x1_driveline = make_routed_x1_driveline(x1_map, road_edge_indexes=road_edge_indexes)

    return routed_x1_driveline


# %%
if __name__ == '__main__':
    GRAPH_MAP_FILE = 'map/road_graph_.pickle'
    with open(GRAPH_MAP_FILE, 'rb') as f:
        G = pickle.load(f)

    # --- ルーティング ---
    route_start_node = 1
    route_end_node = 7

    routed_x1_driveline = main(G, route_start_node, route_end_node)
    routed_quadrant_map = make_quadrant_map(routed_x1_driveline, radius=10)
    np.save(f'map/quadrant_map_10_x1.npy', routed_quadrant_map)

# %%
Image.fromarray((routed_x1_driveline[int(800/1):int(2400/1), int(2500/1):int(4000/1)].T)[::-1].astype(np.uint8))
# %%
np.save(f'map/quadrant_map_10_x1.npy', routed_quadrant_map)
