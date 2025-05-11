import time
import numpy as np
from simple_pid import PID
from pyxinput import vController

from modules import compute_radius, UDP_Reader
import config


SHM_NAME = config.SHM_NAME
SHM_BUFFER_SIZE = config.SHM_BUFFER_SIZE
ROAD_CENTER_VALUE_THRESHOLD = config.ROAD_CENTER_VALUE_THRESHOLD

road_map = np.load('map/palacio_oval.npy')
road_map = road_map > ROAD_CENTER_VALUE_THRESHOLD

reader = UDP_Reader()

# === 設定 ===
MPH_TO_MPS = 0.44704
TARGET_SPEED = 30 * MPH_TO_MPS
RADIUS_THRESH = 75
STEER_OFFSET = 0

THROTTLE_MAX = 0.6

S_KP, S_KI, S_KD = 5.0 / 127.0, 0.0, 8.0 / 127.0
T_KP, T_KI, T_KD = 0.05, 0.05, 0.05

# 仮想コントローラ初期化
controller = vController()

steer_pid = PID(S_KP, S_KI, S_KD, setpoint=0)
steer_pid.output_limits = (-1, 1)

throttle_pid = PID(T_KP, T_KI, T_KD, setpoint=TARGET_SPEED)
throttle_pid.output_limits = (0, THROTTLE_MAX)


def control_loop():
    input_init = False
    target_dt = 1.0 / 60  # 60Hz ≒ 16.666ms

    while True:
        loop_start = time.time()
        try:
            data = reader.read_data()

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

            # print(
            #     f"[OK] speed={speed:.2f}, steer={steer:.2f}, throttle={throttle:.2f}, radius={radius:.1f}")
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
