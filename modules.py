from multiprocessing import shared_memory
from scipy.spatial import cKDTree
import numpy as np
import struct
import config


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



class UDP_Reader:
    def __init__(self):
        self.SHM_NAME = config.SHM_NAME
        self.SHM_BUFFER_SIZE = config.SHM_BUFFER_SIZE

        try:
            self.shm = shared_memory.SharedMemory(name=self.SHM_NAME)
            print("共有メモリに接続しました")
        except FileNotFoundError:
            print("共有メモリが存在しません")
            exit(1)

    def read_data(self):
        raw = bytes(self.shm.buf[:self.SHM_BUFFER_SIZE])
        data = decode_forza_udp(raw)

        return data



class CarModel:
    def __init__(self, right_hand_traffic=True,
                 search_radius=30, search_retries=10, search_retry_step=1):

        self.route_map_x1 = None

        self.map_offset_x = config.MAP_OFFSET_X
        self.map_offset_z = config.MAP_OFFSET_Z
        self.map_scale = config.MAP_SCALE
        self.road_center_distance_threshold = config.ROAD_CENTER_DISTANCE_THRESHOLD
        self.right_hand_traffic = right_hand_traffic
        self.is_menu = True

        self.reader = UDP_Reader()
        self.data = None
        self.pos_x = None
        self.pos_z = None
        self.yaw = None
        self.drivingline_error = None

        self.radius = None
        self.nearest_point_x4 = None

        self.search_radius = search_radius  # 短期探索半径
        self.search_retries = search_retries  # 探索の最大試行回数
        self.search_retry_step = search_retry_step  # 探索半径の増加量 (ピクセル単位)

        self.yaw_error = None
        self.lateral_error = None

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
        udp_sum_value = 0
        for i in list(self.data.keys()):
            udp_sum_value += self.data[i]

        if udp_sum_value == self.data['TimestampMS']: ## メニュー画面の場合
            self.is_menu = True
        else:
            self.is_menu = False
        
        self.speed_mps = data['Speed']
        self.yaw = data['Yaw']
        self.pos_x = data['PositionX']
        self.pos_z = data['PositionZ']
        self.drivingline_error = data['NormalizedDrivingLine']

        self.radius = compute_radius(data)

    def ret_yaw_error(self, target_point_x1):
        if not target_point_x1 is None:
            target_point_x1[0] += int(self.nearest_point_x4[0]/self.map_scale)
            target_point_x1[1] += int(self.nearest_point_x4[1]/self.map_scale)

            offset = self.compute_offset_vector(target_point_x1)

            target_point = (target_point_x1[0]+offset[0], target_point_x1[1]+offset[1])
            yaw_error = self.calc_yaw_error(target_point)

            return yaw_error
        else:
            return None

    def calc_yaw_error(self, target_point):
        if target_point is not None:
            target_x, target_z = target_point

            target_yaw = np.arctan2(target_x - self.array_coord_x_x4/self.map_scale,
                                    target_z - self.array_coord_z_x4/self.map_scale)

            yaw_error = normalize_angle_rad(self.yaw - target_yaw)

            return yaw_error
        else:
            return None

    def lateral_offset_signed(self, target_point_x1):
        """
        現在位置が、理想→目標に向かう単位進行ベクトルに対して
        どれだけ右(または左)に離れているかを符号付きで返す。

        Returns:
            float: 符号付き横方向距離(正＝通行側にずれている)
        """

        if target_point_x1 is not None:
            target_x, target_z = target_point_x1[0]*self.map_scale, target_point_x1[1]*self.map_scale
            ideal_x, ideal_z = self.nearest_point_x4[0], self.nearest_point_x4[1]

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

    def compute_offset_vector(self, target_point, offset=2, right_hand_traffic=True):
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
        ideal_x, ideal_z = self.nearest_point_x4[0]/self.map_scale, self.nearest_point_x4[1]/self.map_scale

        # 進行方向ベクトル
        target_x, target_z = target_point

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

    def calc_next_point_differ(self, base_radius, pos_x, pos_z):
        if self.route_map_x1 is not None:
            for i in range(base_radius - 1):
                radius = base_radius - i
                if radius > 10:
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

            for i in range(self.search_retries):
                # 探索半径を増やす
                radius = base_radius + i * self.search_retry_step  # 探索半径を増やす
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


def decode_forza_udp(data, buffer_offset=12):
    # Sledフォーマットの解析
    sled_data = struct.unpack(
        'iIfffffffffffffffffffffffffffiiiiffffffffffffffffffffiiiii',
        data[:232]  # Sledフォーマットは最初の232バイト
    )

    # Dashフォーマットの解析
    dash_data = struct.unpack(
        'fffffffffffffffffHBBBBBBbbb',
        data[232 + buffer_offset:311 + buffer_offset]  # Dashフォーマットはその後の79バイト
    )

    # Sledフォーマットのデコード
    decoded_data = {
        'IsRaceOn': sled_data[0],
        'TimestampMS': sled_data[1],
        'EngineMaxRpm': sled_data[2],
        'EngineIdleRpm': sled_data[3],
        'CurrentEngineRpm': sled_data[4],
        'AccelerationX': sled_data[5],
        'AccelerationY': sled_data[6],
        'AccelerationZ': sled_data[7],
        'VelocityX': sled_data[8],
        'VelocityY': sled_data[9],
        'VelocityZ': sled_data[10],
        'AngularVelocityX': sled_data[11],
        'AngularVelocityY': sled_data[12],
        'AngularVelocityZ': sled_data[13],
        'Yaw': sled_data[14],
        'Pitch': sled_data[15],
        'Roll': sled_data[16],
        'NormalizedSuspensionTravelFrontLeft': sled_data[17],
        'NormalizedSuspensionTravelFrontRight': sled_data[18],
        'NormalizedSuspensionTravelRearLeft': sled_data[19],
        'NormalizedSuspensionTravelRearRight': sled_data[20],
        'TireSlipRatioFrontLeft': sled_data[21],
        'TireSlipRatioFrontRight': sled_data[22],
        'TireSlipRatioRearLeft': sled_data[23],
        'TireSlipRatioRearRight': sled_data[24],
        'WheelRotationSpeedFrontLeft': sled_data[25],
        'WheelRotationSpeedFrontRight': sled_data[26],
        'WheelRotationSpeedRearLeft': sled_data[27],
        'WheelRotationSpeedRearRight': sled_data[28],
        'WheelOnRumbleStripFrontLeft': sled_data[29],
        'WheelOnRumbleStripFrontRight': sled_data[30],
        'WheelOnRumbleStripRearLeft': sled_data[31],
        'WheelOnRumbleStripRearRight': sled_data[32],
        'WheelInPuddleDepthFrontLeft': sled_data[33],
        'WheelInPuddleDepthFrontRight': sled_data[34],
        'WheelInPuddleDepthRearLeft': sled_data[35],
        'WheelInPuddleDepthRearRight': sled_data[36],
        'SurfaceRumbleFrontLeft': sled_data[37],
        'SurfaceRumbleFrontRight': sled_data[38],
        'SurfaceRumbleRearLeft': sled_data[39],
        'SurfaceRumbleRearRight': sled_data[40],
        'TireSlipAngleFrontLeft': sled_data[41],
        'TireSlipAngleFrontRight': sled_data[42],
        'TireSlipAngleRearLeft': sled_data[43],
        'TireSlipAngleRearRight': sled_data[44],
        'TireCombinedSlipFrontLeft': sled_data[45],
        'TireCombinedSlipFrontRight': sled_data[46],
        'TireCombinedSlipRearLeft': sled_data[47],
        'TireCombinedSlipRearRight': sled_data[48],
        'SuspensionTravelMetersFrontLeft': sled_data[49],
        'SuspensionTravelMetersFrontRight': sled_data[50],
        'SuspensionTravelMetersRearLeft': sled_data[51],
        'SuspensionTravelMetersRearRight': sled_data[52],
        'CarOrdinal': sled_data[53],
        'CarClass': sled_data[54],
        'CarPerformanceIndex': sled_data[55],
        'DrivetrainType': sled_data[56],
        'NumCylinders': sled_data[57],
        'PositionX': dash_data[0],
        'PositionY': dash_data[1],
        'PositionZ': dash_data[2],
        'Speed': dash_data[3],
        'Power': dash_data[4],
        'Torque': dash_data[5],
        'TireTempFrontLeft': dash_data[6],
        'TireTempFrontRight': dash_data[7],
        'TireTempRearLeft': dash_data[8],
        'TireTempRearRight': dash_data[9],
        'Boost': dash_data[10],
        'Fuel': dash_data[11],
        'Distance': dash_data[12],
        'BestLapTime': dash_data[13],
        'LastLapTime': dash_data[14],
        'CurrentLapTime': dash_data[15],
        'CurrentRaceTime': dash_data[16],
        'LapNumber': dash_data[17],
        'RacePosition': dash_data[18],
        'Accel': dash_data[19],
        'Brake': dash_data[20],
        'Clutch': dash_data[21],
        'HandBrake': dash_data[22],
        'Gear': dash_data[23],
        'Steer': dash_data[24],
        'NormalizedDrivingLine': dash_data[25],
        'NormalizedAIBrakeDifference': dash_data[26],
    }

    return decoded_data


def normalize_angle_rad(angle):
    """角度を [-π, π] に正規化"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def compute_radius(data):
    vel = np.array(
        [data['VelocityX'], data['VelocityY'], data['VelocityZ']])
    ang = np.array(
        [data['AngularVelocityX'], data['AngularVelocityY'], data['AngularVelocityZ']])

    v = np.linalg.norm(vel)
    w = np.linalg.norm(ang)
    return float('inf') if w == 0 else v / w


def convert_forzaposition_to_arraycoord(forza_pos_x, forza_pos_z, offset_x, offset_z, scale):
    array_coord_x = int((forza_pos_x + offset_x)*scale)
    array_coord_z = int((forza_pos_z + offset_z)*scale)

    return array_coord_x, array_coord_z


def convert_arraycoord_to_forzaposition(array_coord_x, array_coord_z, offset_x, offset_z, scale):
    forza_pos_x = int((array_coord_x/scale) - offset_x)
    forza_pos_z = int((array_coord_z/scale) - offset_z)

    return forza_pos_x, forza_pos_z


def find_nearest_road_pixel(road_map, pos_x, pos_z, threshold, window_size=50):
    """
    自車座標の近傍 (±window_size) の範囲で最も近い道路ピクセルを探索

    Args:
        road_map (np.ndarray): shape=(W, H), 値は 0 or 1 の2Dマップ
        pos_x (int): 自車のX座標(列)
        pos_z (int): 自車のZ座標(行)
        window_size (int): 探索半径(ピクセル)

    Returns:
        tuple: ((x, z), 距離) 最寄りピクセルと距離
    """
    W, H = road_map.shape

    # 探索範囲を切り出し
    x0 = max(int(pos_x) - window_size, 0)
    x1 = min(int(pos_x) + window_size + 1, W)
    z0 = max(int(pos_z) - window_size, 0)
    z1 = min(int(pos_z) + window_size + 1, H)

    submap = road_map[x0:x1, z0:z1]

    # 道路ピクセル座標を抽出
    rel_coords = np.column_stack(np.where(submap > threshold))  # (x, z) 相対座標
    if len(rel_coords) == 0:
        return None, float('inf')  # 範囲内に道路なし

    # 相対 → 絶対座標に変換
    abs_coords = rel_coords + [x0, z0]  # shape=(N, 2)

    # 最近傍探索
    tree = cKDTree(abs_coords)
    distance, index = tree.query([pos_x, pos_z])  # 現在地

    nearest = abs_coords[index]
    return (nearest[0], nearest[1]), distance  # (x, z), 距離


def compress_info32(driving_line: int, is_dirt: int, pos_y: int,
                    radius: int, yaw: int) -> int:
    """指定のビット順で 32bit に圧縮する"""
    if not (0 <= driving_line <= 0x7F):
        raise ValueError("driving_line must be 0-127 (7bit)")
    if is_dirt not in (0, 1):
        raise ValueError("is_dirt must be 0 or 1 (1bit)")
    if not (0 <= pos_y <= 0xFF):
        raise ValueError("pos_y must be 0-255 (8bit)")
    if not (0 <= radius <= 0xFF):
        raise ValueError("radius must be 0-255 (8bit)")
    if not (0 <= yaw <= 0xFF):
        raise ValueError("yaw must be 0-255 (8bit)")

    return (
        (yaw << 24) |
        (radius << 16) |
        (pos_y << 8) |
        (is_dirt << 7) |
        driving_line
    )


def extract_driving_line_from_d32(d32: np.ndarray) -> np.ndarray:
    """uint32配列から driving_line (7bit) を抽出"""
    if d32.dtype != np.uint32:
        raise ValueError("dtype は np.uint32 である必要があります")
    return (d32 & 0x7F).astype(np.uint8)


def extract_is_dirt_from_d32(d32: np.ndarray) -> np.ndarray:
    """uint32配列から is_dirt (1bit) を抽出"""
    if d32.dtype != np.uint32:
        raise ValueError("dtype は np.uint32 である必要があります")
    return ((d32 >> 7) & 0x1).astype(np.uint8)


def extract_pos_y_from_d32(d32: np.ndarray) -> np.ndarray:
    """uint32配列から pos_y (8bit) を抽出"""
    if d32.dtype != np.uint32:
        raise ValueError("dtype は np.uint32 である必要があります")
    return ((d32 >> 8) & 0xFF).astype(np.uint8)


def extract_radius_from_d32(d32: np.ndarray) -> np.ndarray:
    """uint32配列から radius (8bit) を抽出"""
    if d32.dtype != np.uint32:
        raise ValueError("dtype は np.uint32 である必要があります")
    return ((d32 >> 16) & 0xFF).astype(np.uint8)


def extract_yaw_from_d32(d32: np.ndarray) -> np.ndarray:
    """uint32配列から yaw (8bit) を抽出"""
    if d32.dtype != np.uint32:
        raise ValueError("dtype は np.uint32 である必要があります")
    return ((d32 >> 24) & 0xFF).astype(np.uint8)


def down_scale_map(road_map, down_scale, mode='max'):
    xlen, zlen = road_map.shape
    resized_xlen = int(xlen / down_scale)
    resized_zlen = int(zlen / down_scale)

    new_shape = (
        resized_xlen, down_scale,
        resized_zlen, down_scale
    )

    # reshapeしてから max を取る（axis=(1, 3) は downscale 部分）
    # road_map = extract_driving_line_from_d32(road_map)
    resized_img = road_map[:resized_xlen*down_scale, :resized_zlen*down_scale].reshape(new_shape)
    if mode == 'max':
        resized_map = resized_img.max(axis=(1, 3))
    elif mode == 'min':
        resized_map = resized_img.min(axis=(1, 3))
    else:
        resized_map = np.average(resized_img, axis=(1, 3))

    return resized_map
