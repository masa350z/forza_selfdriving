import time
import numpy as np
from simple_pid import PID
from pyxinput import vController
from forza_udp_decoder import decode_forza_udp

from multiprocessing import shared_memory

SHM_NAME = 'forza_shm'
BUFFER_SIZE = 1024

try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    print("共有メモリに接続しました")
except FileNotFoundError:
    print("共有メモリが存在しません")
    exit(1)

# === 設定 ===
MPH_TO_MPS = 0.44704
TARGET_SPEED = 30 * MPH_TO_MPS
RADIUS_THRESH = 75
STEER_OFFSET = 0

THROTTLE_MAX=0.6

# PIDゲイン
# S_KP, S_KI, S_KD = 0.1 / 127.0, 0.5 / 127.0, 8.0 / 127.0
# T_KP, T_KI, T_KD = 1, 1, 1

S_KP, S_KI, S_KD = 5.0 / 127.0, 0.0, 8.0 / 127.0
T_KP, T_KI, T_KD = 0.05, 0.05, 0.05

# 仮想コントローラ初期化
controller = vController()

# PID初期化
pid_steer_straight = PID(0.2 / 127, 0.0, 0.2 / 127, setpoint=0)
pid_steer_medium   = PID(0.4 / 127, 0.1 / 127, 0.4 / 127, setpoint=0)
pid_steer_sharp    = PID(0.6 / 127, 0.2 / 127, 0.8 / 127, setpoint=0)

# 出力制限は共通
for pid in [pid_steer_straight, pid_steer_medium, pid_steer_sharp]:
    pid.output_limits = (-1, 1)

steer_pid = PID(S_KP, S_KI, S_KD, setpoint=0)
steer_pid.output_limits = (-1, 1)

throttle_pid = PID(T_KP, T_KI, T_KD, setpoint=TARGET_SPEED)
throttle_pid.output_limits = (0, THROTTLE_MAX)

def select_pid_by_radius(radius):
    if radius > 100:         # 直線（ほぼ曲率0）
        return pid_steer_straight
    elif radius > 50:        # 中カーブ
        return pid_steer_medium
    else:                    # 急カーブ
        return pid_steer_sharp


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
            raw = bytes(shm.buf[:BUFFER_SIZE])
            data = decode_forza_udp(raw)

            speed = data['Speed']
            norm_line = data['NormalizedDrivingLine']

            if norm_line != 0:
                radius = compute_radius(data)

                lateral_error = norm_line + STEER_OFFSET

                steer_feedback = steer_pid(lateral_error)
                # steer_feedback = select_pid_by_radius(radius)(lateral_error)
                steer = -steer_feedback

                throttle = throttle_pid(speed) if radius > RADIUS_THRESH else 0

                controller.set_value('AxisLx', steer)
                controller.set_value('TriggerR', throttle)

                input_init = False
            
            else:
                if not input_init:
                    controller.set_value('AxisLx', 0)
                    controller.set_value('TriggerR', 0)
                    input_init = True

            print(
                f"[OK] speed={speed:.2f}, steer={steer:.2f}, throttle={throttle:.2f}, radius={radius:.1f}")
        except Exception as e:
            print("受信エラー:", e)

        # 60Hz周期を維持するためにスリープ
        elapsed = time.time() - loop_start
        sleep_time = target_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# === 実行 ===
if __name__ == "__main__":
    control_loop()
