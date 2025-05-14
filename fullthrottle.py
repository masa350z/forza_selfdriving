from pyxinput import vController
from simple_pid import PID
import numpy as np
import time

from modules import CarModel
import config


class FULLTHROTTLE_MODEL(CarModel):
    def __init__(self, roadmap, quadrant_map, roadcenter_distance=0,
                 right_hand_traffic=True, running_herz=60):
        super().__init__(roadmap, quadrant_map, right_hand_traffic)

        self.controller = vController()  # 仮想コントローラ初期化
        target_speed_mps = 300/1.6 * config.MPH_TO_MPS

        self.throttle_pid = PID(0.05, 0.05, 0.05, setpoint=target_speed_mps)
        self.throttle_pid.output_limits = (0, 1)

        self.target_dt = 1/running_herz
        self.input_init = False

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
        if (self.pos_x != 0) and (self.pos_z != 0) and (self.drivingline_error != 0):
            a = self.data['TireSlipRatioFrontLeft']
            b = self.data['TireSlipRatioFrontRight']
            c = self.data['TireSlipRatioRearLeft']
            d = self.data['TireSlipRatioRearRight']

            slip = (a+b+c+d)/4

            throttle = self.throttle_pid(self.speed_mps)
            throttle = throttle if self.radius > 50 else 0
            if slip > 0.3:
                throttle = 0
            else:
                self.controller.set_value('TriggerR', throttle)

            print(f"throttle={throttle: .2f}", end='\r')

            self.input_init = False

        else:
            if not self.input_init:
                self.controller.set_value('TriggerR', 0)
                self.input_init = True


if __name__ == "__main__":
    MAP_SCALE = config.MAP_SCALE
    ROAD_MAP = np.load(f'map/palacio_oval_x{MAP_SCALE}.npy')
    QUADRANT_MAP = np.load('map/quadrant_map_10_x1.npy')

    ad_car = FULLTHROTTLE_MODEL(roadmap=ROAD_MAP, quadrant_map=QUADRANT_MAP, roadcenter_distance=0)
    ad_car.run()
