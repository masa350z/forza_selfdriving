import numpy as np
from tqdm import tqdm
import config

MAP_SCALE = config.MAP_SCALE
MAP_OFFSET_X = config.MAP_OFFSET_X
MAP_OFFSET_Z = config.MAP_OFFSET_Z
ROAD_CENTER_DISTANCE_THRESHOLD = config.ROAD_CENTER_DISTANCE_THRESHOLD


def circle_offsets_by_quadrant(r: int):
    """半径 r の円周上にある整数オフセットを 4 象限ごとに返す。

    第 1 象限: (x ≥ 0, y ≥ 0)
    第 2 象限: (x <  0, y ≥ 0)
    第 3 象限: (x <  0, y <  0)
    第 4 象限: (x ≥ 0, y <  0)

    軸上(0 を含む)の点は、上記の「≥ / <」の条件に従って
    それぞれ第 1 または第 4 象限へ含める。

    Args:
        r (int): 半径 (ピクセル単位)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (q1, q2, q3, q4) いずれも shape=(m_i, 2), dtype=int
    """
    # --- 円周上の整数オフセットを生成 ---
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    dx = np.round(r * np.cos(angles)).astype(int)
    dz = np.round(r * np.sin(angles)).astype(int)

    # 重複を除去
    offsets = np.unique(np.stack([dx, dz], axis=1), axis=0)

    # --- 象限ごとに抽出 ---
    q1 = offsets[(offsets[:, 0] >= 0) & (offsets[:, 1] >= 0)]
    q2 = offsets[(offsets[:, 0] < 0) & (offsets[:, 1] >= 0)]
    q3 = offsets[(offsets[:, 0] < 0) & (offsets[:, 1] < 0)]
    q4 = offsets[(offsets[:, 0] >= 0) & (offsets[:, 1] < 0)]

    return q1, q2, q3, q4


def ret_candidate_points(x0, z0, offsets, W, H):
    candidate_points = np.array([[x0, z0]]) + offsets
    condition01 = candidate_points[:, 0] < W
    condition02 = candidate_points[:, 1] < H
    candidate_points = candidate_points[condition01 & condition02]

    return candidate_points


def ret_offset_value(road_map, x0, z0, radius, W, H):
    offsets01, offsets02, offsets03, offsets04 = circle_offsets_by_quadrant(radius)

    candidate_points01 = ret_candidate_points(x0, z0, offsets01, W, H)
    candidate_points02 = ret_candidate_points(x0, z0, offsets02, W, H)
    candidate_points03 = ret_candidate_points(x0, z0, offsets03, W, H)
    candidate_points04 = ret_candidate_points(x0, z0, offsets04, W, H)

    value_array01 = road_map[candidate_points01[:, 0], candidate_points01[:, 1]]
    value_array02 = road_map[candidate_points02[:, 0], candidate_points02[:, 1]]
    value_array03 = road_map[candidate_points03[:, 0], candidate_points03[:, 1]]
    value_array04 = road_map[candidate_points04[:, 0], candidate_points04[:, 1]]

    max_value01 = np.max(value_array01)
    max_value02 = np.max(value_array02)
    max_value03 = np.max(value_array03)
    max_value04 = np.max(value_array04)

    offset_value01 = offsets01[value_array01 == max_value01][0]*(max_value01 != 0)
    offset_value02 = offsets02[value_array02 == max_value02][0]*(max_value02 != 0)
    offset_value03 = offsets03[value_array03 == max_value03][0]*(max_value03 != 0)
    offset_value04 = offsets04[value_array04 == max_value04][0]*(max_value04 != 0)

    return offset_value01, offset_value02, offset_value03, offset_value04


def main(road_map, radius):
    W, H = road_map.shape

    quadrant_map = np.zeros((W, H, 4, 2), dtype=np.int16)  # NE, SE, SW, NW

    max_retry = 10

    road_coords = np.column_stack(np.where(road_map > ROAD_CENTER_DISTANCE_THRESHOLD))  # [x, z]形式

    for x0, z0 in tqdm(road_coords, desc="探索中"):
        filled = 0
        for retry in range(max_retry + 1):
            r = radius + retry

            offset_values = ret_offset_value(road_map, x0, z0, r, W, H)

            for q in range(4):
                if not np.array_equal(offset_values[q], np.zeros(2)):
                    quadrant_map[x0, z0, q] = offset_values[q]
                    filled += 1

            if filled >= 2:
                break

    return quadrant_map


# %%
if __name__ == '__main__':
    road_map = np.load(f'map/palacio_oval_x1.npy')

    for RADIUS in [10, 20, 30, 50, 100]:
        quadrant_map = main(road_map, radius=RADIUS)
        np.save(f'map/quadrant_map_{RADIUS}_x1.npy', quadrant_map)

# %%
