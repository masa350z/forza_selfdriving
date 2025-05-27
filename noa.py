# %%
import numpy as np
import config
from modules import normalize_angle_rad, compute_radius, find_nearest_road_pixel, convert_forzaposition_to_arraycoord, UDP_Reader
import time


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

    return np.array([offset_value01, offset_value02, offset_value03, offset_value04])


class CarModel:
    def __init__(self, roadmap_x4, intersection_mask, right_hand_traffic=True,
                 search_radius=30, search_retries=10, search_retry_step=1):

        self.road_map_x4 = roadmap_x4
        self.route_map_x1 = None
        self.intersection_mask_x1 = intersection_mask

        self.map_offset_x = config.MAP_OFFSET_X
        self.map_offset_z = config.MAP_OFFSET_Z
        self.map_scale = config.MAP_SCALE
        self.road_center_distance_threshold = config.ROAD_CENTER_DISTANCE_THRESHOLD

        self.reader = UDP_Reader()
        self.data = None
        self.drivingline_error = None
        self.radius = None
        self.nearest_coord_x4 = None
        self.target_coord_x1 = None
        self.offset_target_coordinate = None
        self.pos_x = None
        self.pos_z = None
        self.yaw = None
        self.right_hand_traffic = right_hand_traffic

        self.search_radius = search_radius  # 半径 (ピクセル単位)
        self.search_retries = search_retries  # 探索の最大試行回数
        self.search_retry_step = search_retry_step  # 探索半径の増加量 (ピクセル単位)

        self.is_intersection = False

    def update_route_map(self, route_map_x1):
        """
        ルートマップを更新するメソッド。

        Args:
            route_map_x1 (np.ndarray): ルートマップ (x1)。
        """
        self.route_map_x1 = route_map_x1

    def update(self):
        data = self.reader.read_data()
        self.data = data
        self.speed_mps = data['Speed']
        self.yaw = data['Yaw']
        self.pos_x = data['PositionX']
        self.pos_z = data['PositionZ']
        self.drivingline_error = data['NormalizedDrivingLine']

        self.radius = compute_radius(data)

        self.array_coord_x_x4, self.array_coord_z_x4 = convert_forzaposition_to_arraycoord(self.pos_x,
                                                                                     self.pos_z,
                                                                                     self.map_offset_x,
                                                                                     self.map_offset_z,
                                                                                     self.map_scale)
        self.array_coord_x_x1 = int(self.array_coord_x_x4/self.map_scale)
        self.array_coord_z_x1 = int(self.array_coord_z_x4/self.map_scale)
        
        self.is_intersection = self.intersection_mask_x1[self.array_coord_x_x1][self.array_coord_z_x1]

        self.nearest_coord_x4, _ = find_nearest_road_pixel(self.road_map_x4,
                                                           self.array_coord_x_x4,
                                                           self.array_coord_z_x4,
                                                           self.road_center_distance_threshold,
                                                           window_size=50*self.map_scale)

        if self.nearest_coord_x4 is not None:

            # 目標点
            self.target_coord_x1 = self.calc_next_point_differ(int(self.nearest_coord_x4[0]/self.map_scale),
                                                               int(self.nearest_coord_x4[1]/self.map_scale))

            if not self.target_coord_x1 is None:
                self.target_coord_x1[0] += int(self.nearest_coord_x4[0]/self.map_scale)
                self.target_coord_x1[1] += int(self.nearest_coord_x4[1]/self.map_scale)

                offset = self.compute_offset_vector()

                self.offset_target_coordinate = (
                    self.target_coord_x1[0]+offset[0], self.target_coord_x1[1]+offset[1])

    def calc_yaw_error(self):
        if self.target_coord_x1 is not None:
            target_x, target_z = self.offset_target_coordinate

            target_yaw = np.arctan2(target_x - self.array_coord_x_x4/self.map_scale,
                                    target_z - self.array_coord_z_x4/self.map_scale)

            yaw_error = normalize_angle_rad(self.yaw - target_yaw)

            return yaw_error
        else:
            return None

    def lateral_offset_signed(self):
        """
        現在位置が、理想→目標に向かう単位進行ベクトルに対して
        どれだけ右(または左)に離れているかを符号付きで返す。

        Returns:
            float: 符号付き横方向距離(正＝通行側にずれている)
        """

        if self.target_coord_x1 is not None:
            target_x, target_z = self.target_coord_x1[0]*self.map_scale, self.target_coord_x1[1]*self.map_scale
            ideal_x, ideal_z = self.nearest_coord_x4[0], self.nearest_coord_x4[1]

            # 理想 → 目標 方向ベクトル
            dx = target_x - ideal_x
            dz = target_z - ideal_z

            # 正規化(単位ベクトル)
            norm = np.hypot(dx, dz)
            if norm == 0:
                return 0.0  # 進行方向が定義できない

            dx /= norm
            dz /= norm

            # 右90度回転ベクトル(右手系: (dz, -dx))
            normal_x = dz
            normal_z = -dx
            if not self.right_hand_traffic:
                normal_x *= -1
                normal_z *= -1

            # 現在位置 → 理想位置ベクトル
            vx = self.array_coord_x_x4 - ideal_x
            vz = self.array_coord_z_x4 - ideal_z

            # 横方向距離＝内積(現在位置と進行線の法線)
            lateral_distance = vx * normal_x + vz * normal_z

            return lateral_distance

        else:
            return None

    def compute_offset_vector(self, offset=3, right_hand_traffic=True):
        """
        現在地から目標地点に向かう直線に対して、右または左に offset 分だけずらした位置への差分 (dx, dz) を返す

        Args:
            current_x (float): 現在のX座標
            current_z (float): 現在のZ座標
            target_x (float): 目標のX座標
            target_z (float): 目標のZ座標
            offset (float): 右または左にずらす距離(ピクセル単位)
            right_hand_traffic (bool): Trueなら右側通行(右へオフセット)、Falseなら左側通行

        Returns:
            tuple[float, float]: (dx, dz) 差分ベクトル
        """
        ideal_x, ideal_z = self.nearest_coord_x4[0]/self.map_scale, self.nearest_coord_x4[1]/self.map_scale

        # 進行方向ベクトル
        target_x, target_z = self.target_coord_x1

        dx = target_x - ideal_x
        dz = target_z - ideal_z

        # 単位ベクトルに正規化
        norm = np.sqrt(dx**2 + dz**2)
        if norm == 0:
            return (0.0, 0.0)  # 進行方向なし

        dx /= norm
        dz /= norm

        # 垂直方向ベクトル(右90度回転: (dz, -dx)、左なら (-dz, dx))
        if right_hand_traffic:
            offset_dx = dz * offset
            offset_dz = -dx * offset
        else:
            offset_dx = -dz * offset
            offset_dz = dx * offset

        return (offset_dx, offset_dz)

    def calc_next_point_differ(self, pos_x, pos_z):
        if self.route_map_x1 is not None:
            for i in range(self.search_retries):
                # 探索半径を増やす
                radius = self.search_radius + i * self.search_retry_step  # 探索半径を増やす
                next_candidate_points = ret_offset_value(self.route_map_x1, pos_x, pos_z, radius,
                                                         W=config.MAP_SIZE_X,
                                                         H=config.MAP_SIZE_Z)
                if np.sum(np.abs(next_candidate_points)) != 0:
                    angle = np.arctan2(next_candidate_points[:, 0], next_candidate_points[:, 1]) - self.yaw
                    angle = np.abs(normalize_angle_rad(angle))

                    # ★ [0,0] の候補に対して angle を巨大値に置き換える ★
                    zero_mask = np.all(next_candidate_points == 0, axis=1)
                    angle[zero_mask] = 1e9

                    min_angle = np.min(angle)

                    if min_angle > np.pi*(3/4):
                        continue

                    next_point_differ = next_candidate_points[angle == min_angle][0]

                    return next_point_differ
        else:
            return None


# %%


if __name__ == "__main__":
    MAP_SCALE = config.MAP_SCALE          # 例: 4

    # --- データ読み込み --------------------------------------
    INTERSECTION_MASK = np.load('tmp/intersection_mask.npy')
    ROAD_MAP_X4 = np.load(f'map/palacio_simple_x{MAP_SCALE}.npy')      # x4
    ROUTE_MASK_X1 = (np.load('tmp/routed_road.npy') > 0).astype(np.uint8)
    ROUTE_DIST_X1 = np.load('tmp/routed_road.npy')                       # 同じファイル
