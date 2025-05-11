# %%
import time
import numpy as np
from simple_pid import PID
from pyxinput import vController
from modules import decode_forza_udp
from multiprocessing import shared_memory
import config

from modules import calc_next_point, find_nearest_road_pixel, normalize_angle_rad

def calc_yaw_error(pos_x, pos_z, yaw):
    nearest_pixel, distance = find_nearest_road_pixel(road_map,
                                                      pos_x + MAP_OFFSET_X,
                                                      pos_z + MAP_OFFSET_Z)

    if nearest_pixel is not None:
        nearest_pixel_x = nearest_pixel[0]
        nearest_pixel_z = nearest_pixel[1]

        # 目標点を地図相対座標に変換
        target = calc_next_point(quadrant_map,
                                 nearest_pixel_x - MAP_OFFSET_X,
                                 nearest_pixel_z - MAP_OFFSET_Z,
                                 yaw,
                                 MAP_OFFSET_X,
                                 MAP_OFFSET_Z)

        target_x = target[0]
        target_z = target[1]

        target_yaw = np.arctan2(target_x - pos_x,
                                target_z - pos_z)

        yaw_error = normalize_angle_rad(yaw - target_yaw)

        return yaw_error
    else:
        return None


MAP_OFFSET_X = config.MAP_OFFSET_X
MAP_OFFSET_Z = config.MAP_OFFSET_Z
SHM_NAME = config.SHM_NAME
SHM_BUFFER_SIZE = config.SHM_BUFFER_SIZE
ROAD_CENTER_VALUE_THRESHOLD = config.ROAD_CENTER_VALUE_THRESHOLD


# === 設定 ===
MAP_FILE = 'map/palacio_oval.npy'
QUAD_FILE = 'map/quadrant_map_30.npy'

MPH_TO_MPS = 0.44704
TARGET_SPEED = 40 * MPH_TO_MPS
RADIUS_THRESH = 35
STEER_OFFSET = 0

THROTTLE_MAX = 0.6


# === マップ読み込み ===
road_map = np.load(MAP_FILE) > ROAD_CENTER_VALUE_THRESHOLD
quadrant_map = np.load(QUAD_FILE)

# === 共有メモリ接続 ===
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    print("共有メモリに接続しました")
except FileNotFoundError:
    print("共有メモリが存在しません")
    exit(1)

# %%
# PIDゲイン
# S_KP, S_KI, S_KD = 0.1 / 127.0, 0.5 / 127.0, 8.0 / 127.0
# T_KP, T_KI, T_KD = 1, 1, 1

S_KP, S_KI, S_KD = 3, 1, 1
T_KP, T_KI, T_KD = 0.05, 0.05, 0.05

# 仮想コントローラ初期化
controller = vController()

steer_pid = PID(S_KP, S_KI, S_KD, setpoint=0)
steer_pid.output_limits = (-1, 1)

steer01_pid = PID(5.0/127.0, 0.0, 8.0/127.0, setpoint=0)
steer01_pid.output_limits = (-1, 1)

throttle_pid = PID(T_KP, T_KI, T_KD, setpoint=TARGET_SPEED)
throttle_pid.output_limits = (0, THROTTLE_MAX)


def compute_radius(data):
    vel = np.array(
        [data['VelocityX'], data['VelocityY'], data['VelocityZ']])
    ang = np.array(
        [data['AngularVelocityX'], data['AngularVelocityY'], data['AngularVelocityZ']])

    v = np.linalg.norm(vel)
    w = np.linalg.norm(ang)
    return float('inf') if w == 0 else v / w


def control_loop():
    input_init = False
    target_dt = 1.0 / 60  # 60Hz ≒ 16.666ms

    while True:
        loop_start = time.time()
        try:
            raw = bytes(shm.buf[:SHM_BUFFER_SIZE])
            data = decode_forza_udp(raw)

            speed = data['Speed']
            yaw = data['Yaw']
            pos_x = data['PositionX']
            pos_z = data['PositionZ']

            norm_line = data['NormalizedDrivingLine']

            yaw_error = calc_yaw_error(pos_x, pos_z, yaw)

            if (yaw_error is not None) and (pos_x != 0) and (pos_z != 0):
                radius = compute_radius(data)

                steer_feedback = steer_pid(yaw_error)
                steer = steer_feedback

                steer_feedback01 = steer01_pid(norm_line)
                steer01 = -steer_feedback01

                throttle = throttle_pid(speed) if radius > RADIUS_THRESH else 0

                # steer = steer+steer01

                # steer = steer if steer < 1 else 1
                # steer = steer if steer > -1 else -1

                # controller.set_value('AxisLx', steer)
                # controller.set_value('TriggerR', throttle)

                input_init = False

            else:
                if not input_init:
                    controller.set_value('AxisLx', 0)
                    controller.set_value('TriggerR', 0)
                    input_init = True

                # print(
                #     f"[OK] speed={speed:.2f}, steer={steer:.2f}, throttle={throttle:.2f}, radius={radius:.1f}")
        except Exception as e:
            pass
            # print("受信エラー:", e)

            # 60Hz周期を維持するためにスリープ
        elapsed = time.time() - loop_start
        sleep_time = target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


# === 実行 ===
if __name__ == "__main__":
    control_loop()
