from multiprocessing import shared_memory
from scipy.spatial import cKDTree
import numpy as np
import struct
import config


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
    def __init__(self, roadmap, quadrant_map, right_hand_traffic=True):

        self.road_map = roadmap
        self.quadrant_map = quadrant_map
        self.map_offset_x = config.MAP_OFFSET_X
        self.map_offset_z = config.MAP_OFFSET_Z
        self.map_scale = config.MAP_SCALE
        self.road_center_distance_threshold = config.ROAD_CENTER_DISTANCE_THRESHOLD

        self.reader = UDP_Reader()
        self.data = None
        self.drivingline_error = None
        self.radius = None
        self.nearest_coord = None
        self.target_coord_per4 = None
        self.offset_target_coordinate = None
        self.pos_x = None
        self.pos_z = None
        self.yaw = None
        self.right_hand_traffic = right_hand_traffic

    def update(self):
        data = self.reader.read_data()
        self.data = data
        self.speed_mps = data['Speed']
        self.yaw = data['Yaw']
        self.pos_x = data['PositionX']
        self.pos_z = data['PositionZ']
        self.drivingline_error = data['NormalizedDrivingLine']

        self.radius = compute_radius(data)

        self.array_coord_x, self.array_coord_z = convert_forzaposition_to_arraycoord(self.pos_x,
                                                                                     self.pos_z,
                                                                                     self.map_offset_x,
                                                                                     self.map_offset_z,
                                                                                     self.map_scale)

        self.nearest_coord, _ = find_nearest_road_pixel(self.road_map,
                                                        self.array_coord_x,
                                                        self.array_coord_z,
                                                        self.road_center_distance_threshold,
                                                        window_size=50*self.map_scale)

        if self.nearest_coord is not None:
            # 目標点
            self.target_coord_per4 = calc_next_point_differ(self.quadrant_map,
                                                            int(self.nearest_coord[0]/self.map_scale),
                                                            int(self.nearest_coord[1]/self.map_scale),
                                                            self.yaw)
            if not self.target_coord_per4 is None:
                self.target_coord_per4[0] += int(self.nearest_coord[0]/self.map_scale)
                self.target_coord_per4[1] += int(self.nearest_coord[1]/self.map_scale)

                offset = self.compute_offset_vector()

                self.offset_target_coordinate = (
                    self.target_coord_per4[0]+offset[0], self.target_coord_per4[1]+offset[1])

    def calc_yaw_error(self):
        if self.target_coord_per4 is not None:
            target_x, target_z = self.offset_target_coordinate

            target_yaw = np.arctan2(target_x - self.array_coord_x/self.map_scale,
                                    target_z - self.array_coord_z/self.map_scale)

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

        if self.target_coord_per4 is not None:
            target_x, target_z = self.target_coord_per4[0]*self.map_scale, self.target_coord_per4[1]*self.map_scale
            ideal_x, ideal_z = self.nearest_coord[0], self.nearest_coord[1]

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
            vx = self.array_coord_x - ideal_x
            vz = self.array_coord_z - ideal_z

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
        ideal_x, ideal_z = self.nearest_coord[0]/self.map_scale, self.nearest_coord[1]/self.map_scale

        # 進行方向ベクトル
        target_x, target_z = self.target_coord_per4

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


def calc_next_point_differ(quadrant_map, pos_x, pos_z, yaw):
    if not (0 <= pos_x < quadrant_map.shape[0] and 0 <= pos_z < quadrant_map.shape[1]):
        return None

    candidates = quadrant_map[pos_x, pos_z]
    if np.sum(candidates > 0) != 0:
        candidates = candidates[np.sum(np.abs(candidates), axis=1) != 0]
        dx = candidates[:, 0]
        dz = candidates[:, 1]

        angle = np.arctan2(dx, dz) - yaw
        angle = np.abs(normalize_angle_rad(angle))

        min_angle = np.min(angle)
        if min_angle < np.pi*(3/4)*2:
            return candidates[angle == min_angle][0]
        else:
            return None
    else:
        return None


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
