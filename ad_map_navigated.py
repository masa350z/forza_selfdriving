from pyxinput import vController
from simple_pid import PID
import numpy as np
import time

from modules import CarModel
import config


class NOA_MODEL(CarModel):
    def __init__(self, roadmap, quadrant_map, roadcenter_distance=10, max_throttle=0.6,
                 right_hand_traffic=True, running_herz=60):
        super().__init__(roadmap, quadrant_map, right_hand_traffic)

        self.controller = vController()  # 仮想コントローラ初期化
        target_speed_mps = config.TARGET_SPEED_MPH * config.MPH_TO_MPS
        self.radius_thresh = config.RADIUS_THRESH

        self.steer_pid = PID(3, 0, 1, setpoint=0)
        self.steer_pid.output_limits = (-1, 1)

        self.steer01_pid = PID(0.04, 0., 0.03, setpoint=roadcenter_distance)
        self.steer01_pid.output_limits = (-1, 1)

        self.throttle_pid = PID(0.05, 0.05, 0.05, setpoint=target_speed_mps)
        self.throttle_pid.output_limits = (0, max_throttle)

        self.target_dt = 1/running_herz
        self.input_init = False

    def run(self):
        while True:
            try:
                loop_start = time.time()

                self.mono_run()

                elapsed = time.time() - loop_start
                sleep_time = self.target_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except:
                pass

    def mono_run(self):
        self.update()
        yaw_error = self.calc_yaw_error()
        lateral_error = self.lateral_offset_signed()

        if (yaw_error is not None) and (lateral_error is not None) and (self.pos_x != 0) and (self.pos_z != 0):
            throttle = self.throttle_pid(self.speed_mps)
            throttle = throttle if self.radius > self.radius_thresh else 0

            steer00 = self.steer_pid(yaw_error)
            steer01 = self.steer01_pid(lateral_error)

            steer = steer00+steer01
            # steer = steer00

            steer = steer if steer < 1 else 1
            steer = steer if steer > -1 else -1

            self.controller.set_value('AxisLx', steer)
            self.controller.set_value('TriggerR', throttle)

            print(f"lateral_error={lateral_error:.2f} yaw={yaw_error:.2f} steer={steer:.2f}, throttle={throttle:.2f}", end='\r')

            self.input_init = False

        else:
            if not self.input_init:
                self.controller.set_value('AxisLx', 0)
                self.controller.set_value('TriggerR', 0)
                self.input_init = True


if __name__ == "__main__":
    MAP_SCALE = config.MAP_SCALE
    ROAD_MAP = np.load(f'map/oval_backup/palacio_oval_x{MAP_SCALE}.npy')
    # ROAD_MAP = np.load(f'map/palacio_simple_x{MAP_SCALE}.npy')
    QUADRANT_MAP = np.load('map/oval_backup/quadrant_map_10_x1.npy')
    # QUADRANT_MAP = np.load('map/quadrant_map_temp.npy')

    ad_car = NOA_MODEL(roadmap=ROAD_MAP, quadrant_map=QUADRANT_MAP)
    ad_car.run()
