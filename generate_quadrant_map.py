import numpy as np
from tqdm import tqdm
import config


MAP_OFFSET_X = config.MAP_OFFSET_X
MAP_OFFSET_Z = config.MAP_OFFSET_Z
ROAD_CENTER_VALUE_THRESHOLD = config.ROAD_CENTER_VALUE_THRESHOLD

road_map = np.load('map/palacio_oval.npy')
road_map = road_map > ROAD_CENTER_VALUE_THRESHOLD


def classify_quadrant(dx, dy):
    if dx >= 0 and dy < 0:
        return 1  # SE
    elif dx >= 0 and dy >= 0:
        return 0  # NE
    elif dx < 0 and dy >= 0:
        return 3  # NW
    elif dx <= 0 and dy <= 0:
        return 2  # SW

def find_quadrant_intersections(road_map, radius, max_retry=10):
    """
    与えられた道路マップの各道路ピクセルに対して、
    半径 'radius' の円周上にある他の道路ピクセルとの交点を探し、
    それを4象限（NE, SE, SW, NW）に分類して記録する。

    各象限で交点が見つからない場合は、radiusを+1しながら最大max_retry回まで探索を行う。

    Args:
        road_map (np.ndarray): shape=(H, W)、0または1からなる道路マップ
        radius (int): 最初に探索する円の半径（ピクセル単位）
        max_retry (int): 半径拡大の最大試行回数
        MAP_OFFSET_X (int): ワールド座標からマップ座標へのX方向オフセット
        MAP_OFFSET_Z (int): ワールド座標からマップ座標へのZ方向オフセット

    Returns:
        np.ndarray: shape=(H, W, 4, 2)、各象限における最初の交点の(x, y)座標（ローカル座標）
                    見つからなければ [0, 0] が入る
    """
    H, W = road_map.shape
    output = np.zeros((H, W, 4, 2), dtype=np.int16)  # NE, SE, SW, NW

    road_coords = np.column_stack(np.where(road_map == 1))  # [y, x]形式

    for y0, x0 in tqdm(road_coords, desc="探索中"):
        filled = [False] * 4  # 各象限の埋まり状態

        for retry in range(max_retry + 1):
            r = radius + retry

            # 半径rの円周上の整数オフセットを生成
            angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
            dx = np.round(r * np.cos(angles)).astype(int)
            dy = np.round(r * np.sin(angles)).astype(int)
            offsets = np.unique(np.stack([dx, dy], axis=1), axis=0)

            for dx, dy in offsets:
                x = x0 + dx
                y = y0 + dy
                if 0 <= x < W and 0 <= y < H and road_map[y, x] == 1:
                    q = classify_quadrant(dx, dy)
                    if not filled[q]:
                        output[y0, x0, q] = [x - MAP_OFFSET_X, y - MAP_OFFSET_Z]
                        filled[q] = True

            if sum(filled) >= 2:  # 2象限以上埋まったら終了
                break

    return output


for RADIUS in [10, 30, 50, 100]:
    quadrant_map = find_quadrant_intersections(road_map, radius=RADIUS)

    np.save(f'map/quadrant_map_{RADIUS}.npy', quadrant_map)
