import numpy as np
from scipy.spatial import cKDTree
import struct
from multiprocessing import shared_memory
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


def find_nearest_road_pixel(road_map, pos_x, pos_z, window_size=50):
    """
    自車座標の近傍 (±window_size) の範囲で最も近い道路ピクセルを探索

    Args:
        road_map (np.ndarray): shape=(H, W), 値は 0 or 1 の2Dマップ
        pos_x (int): 自車のX座標（列）
        pos_z (int): 自車のZ座標（行）
        window_size (int): 探索半径（ピクセル）

    Returns:
        tuple: ((x, z), 距離) 最寄りピクセルと距離
    """
    H, W = road_map.shape

    # 探索範囲を切り出し
    x0 = max(int(pos_x) - window_size, 0)
    x1 = min(int(pos_x) + window_size + 1, W)
    z0 = max(int(pos_z) - window_size, 0)
    z1 = min(int(pos_z) + window_size + 1, H)

    submap = road_map[z0:z1, x0:x1]

    # 道路ピクセル座標を抽出
    rel_coords = np.column_stack(np.where(submap == 1))  # (z, x) 相対座標
    if len(rel_coords) == 0:
        return None, float('inf')  # 範囲内に道路なし

    # 相対 → 絶対座標に変換
    abs_coords = rel_coords + [z0, x0]  # shape=(N, 2)

    # 最近傍探索
    tree = cKDTree(abs_coords)
    distance, index = tree.query([pos_z, pos_x])  # 現在地

    nearest = abs_coords[index]
    return (nearest[1], nearest[0]), distance  # (x, z), 距離


def calc_next_point(quadrant_map, pos_x, pos_z, yaw, MAP_OFFSET_X, MAP_OFFSET_Z):
    # 座標変換（整数ピクセル）
    ix = int(pos_x + MAP_OFFSET_X)
    iz = int(pos_z + MAP_OFFSET_Z)

    if not (0 <= iz < quadrant_map.shape[0] and 0 <= ix < quadrant_map.shape[1]):
        return None

    candidates = quadrant_map[iz, ix]
    if np.sum(candidates) != 0:
        candidates = candidates[candidates[:, 0] != 0]
        dx = candidates[:, 0] - pos_x
        dz = candidates[:, 1] - pos_z

        angle = np.arctan2(dx, dz) - yaw
        angle = np.abs(normalize_angle_rad(angle))

        min_angle = np.min(angle)
        if min_angle < np.pi*(3/4)*2:
            return candidates[angle == min_angle][0]
        else:
            return None
    else:
        return None


def extract_driving_line_from_d16(d16: np.ndarray) -> np.ndarray:
    """uint16圧縮マップからdriving_line (7bit) 情報だけを取り出す

    Args:
        d16 (np.ndarray): shape=(H, W), dtype=uint16 の2次元配列

    Returns:
        np.ndarray: shape=(H, W), dtype=uint8 の driving_line 抽出結果
    """
    if d16.dtype != np.uint16:
        raise ValueError("入力配列は dtype=uint16 である必要があります")

    driving_line_map = ((d16 >> 9) & 0x7F).astype(np.uint8)

    return driving_line_map
