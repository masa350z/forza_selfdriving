# from pyxinput import vController
from simple_pid import PID
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from modules import CarModel, UDPControllerClient
import config


class GAME_NNAVIGATED_MODEL(CarModel):
    def __init__(self, roadcenter_distance=0, max_throttle=0.6,
                 right_hand_traffic=True, running_herz=60):
        super().__init__(right_hand_traffic=right_hand_traffic)

        # self.controller = vController()  # 仮想コントローラ初期化
        self.controller = UDPControllerClient()
        target_speed_mps = config.TARGET_SPEED_MPH * config.MPH_TO_MPS
        self.radius_thresh = config.RADIUS_THRESH

        self.steer_pid = PID(5.0/127.0, 0.0, 8.0/127.0, setpoint=0)
        self.steer_pid.output_limits = (-1, 1)

        self.throttle_pid = PID(0.05, 0.05, 0.05, setpoint=target_speed_mps)
        self.throttle_pid.output_limits = (0, max_throttle)

        self.target_dt = 1/running_herz
        self.input_init = False

        self.roadcenter_distance = roadcenter_distance

    def run(self):
        while True:
            loop_start = time.time()

            self.mono_run()

            elapsed = time.time() - loop_start
            sleep_time = self.target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def mono_run(self):
        self.update()
        if (self.drivingline_error != 0) and (self.pos_x != 0) and (self.pos_z != 0):
            lateral_error = self.drivingline_error + self.roadcenter_distance

            steer_feedback = self.steer_pid(lateral_error)
            steer = -steer_feedback

            throttle = self.throttle_pid(self.speed_mps)
            throttle = throttle if self.radius > self.radius_thresh else 0

            self.controller.set_value('AxisLx', steer)
            self.controller.set_value('TriggerR', throttle)

            print(f"lateral_error={lateral_error:.2f}  steer={steer:.2f}, throttle={throttle:.2f}", end='\r')

            self.input_init = False

        else:
            if not self.input_init:
                self.controller.set_value('AxisLx', 0)
                self.controller.set_value('TriggerR', 0)
                self.input_init = True


if __name__ == "__main__":
    ad_car = GAME_NNAVIGATED_MODEL(roadcenter_distance=0)
    ad_car.run()
