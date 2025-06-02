from pyxinput import vController
from simple_pid import PID
import numpy as np
import time

from modules import CarModel, find_nearest_road_pixel, convert_forzaposition_to_arraycoord
import config


class STEERING:
    def __init__(self, roadcenter_distance_x4=10, max_steer_history=10):
        self.steer_forward_error_pid = PID(3, 0, 1, setpoint=0)
        self.steer_forward_error_pid.output_limits = (-1, 1)

        self.steer_lateral_error_pid = PID(0.04, 0., 0.03, setpoint=roadcenter_distance_x4)
        self.steer_lateral_error_pid.output_limits = (-0.5, 0.5)

        self.steer_history = []
        self.max_steer_history = max_steer_history

    def ret_steer_value(self, yaw_error, lateral_error):
        if yaw_error is not None:
            steer_forward = self.steer_forward_error_pid(yaw_error)
        else:
            return None
        if lateral_error is not None:
            steer_lateral = self.steer_lateral_error_pid(lateral_error)
        else:
            return None

        self.steer_history.append(steer_forward + steer_lateral)
        self.steer_history = self.steer_history[-self.max_steer_history:]

        steer = sum(self.steer_history)/len(self.steer_history)

        steer = steer if steer < 1 else 1
        steer = steer if steer > -1 else -1

        return steer


class NOA_MODEL(CarModel):
    def __init__(self, roadmap, roadcenter_distance_x4=10, max_throttle=0.6,
                 right_hand_traffic=True, running_herz=60):

        super().__init__(right_hand_traffic=right_hand_traffic,
                         search_radius=20, search_retries=5, search_retry_step=10)

        self.road_map_x4 = roadmap

        self.controller = vController()  # 仮想コントローラ初期化
        target_speed_mps = config.TARGET_SPEED_MPH * config.MPH_TO_MPS
        self.radius_thresh = config.RADIUS_THRESH

        self.steering = STEERING(roadcenter_distance_x4)

        self.throttle_pid = PID(0.05, 0.05, 0.05, setpoint=target_speed_mps)
        self.throttle_pid.output_limits = (0, max_throttle)

        self.target_dt = 1/running_herz
        self.input_init = False

    def run(self):
        while True:
            # try:
            loop_start = time.time()

            self.mono_run()

            elapsed = time.time() - loop_start
            sleep_time = self.target_dt - elapsed
            if sleep_time > 0:
                # print(sleep_time)
                time.sleep(sleep_time)
            # except Exception as e:
            #     print(e)

    def mono_run(self):
        self.update()

        if not self.is_menu:  # メニュー画面でない場合
            self.array_coord_x_x4, self.array_coord_z_x4 = convert_forzaposition_to_arraycoord(self.pos_x,
                                                                                               self.pos_z,
                                                                                               self.map_offset_x,
                                                                                               self.map_offset_z,
                                                                                               self.map_scale)
            self.array_coord_x_x1 = int(self.array_coord_x_x4/self.map_scale)
            self.array_coord_z_x1 = int(self.array_coord_z_x4/self.map_scale)

            self.nearest_point_x4, _ = find_nearest_road_pixel(self.road_map_x4,
                                                               self.array_coord_x_x4,
                                                               self.array_coord_z_x4,
                                                               self.road_center_distance_threshold,
                                                               window_size=50*self.map_scale)

            if self.nearest_point_x4 is not None:
                target_point_x1 = self.calc_next_point_differ(self.search_radius,
                                                              int(self.nearest_point_x4[0]/self.map_scale),
                                                              int(self.nearest_point_x4[1]/self.map_scale))

                self.yaw_error = self.ret_yaw_error(target_point_x1)
                self.lateral_error = self.lateral_offset_signed(target_point_x1)

            throttle = self.throttle_pid(self.speed_mps)
            throttle = throttle if self.radius > self.radius_thresh else 0

            steer = self.steering.ret_steer_value(self.yaw_error, self.lateral_error)

            self.controller.set_value('AxisLx', steer)
            self.controller.set_value('TriggerR', throttle)

            lateral_error = self.lateral_error if self.lateral_error is not None else 999
            yaw_error = self.yaw_error if self.yaw_error is not None else 999

            print(f"lateral_error={lateral_error:.2f} yaw_short={yaw_error:.2f} steer={steer:.2f}, throttle={throttle:.2f}", end='\r')
            # print(f"lateral_error={lateral_error:.2f} yaw_short={yaw_error:.2f} steer={steer:.2f}, throttle={throttle:.2f}")

            self.input_init = False

        else:
            if not self.input_init:
                self.controller.set_value('AxisLx', 0)
                self.controller.set_value('TriggerR', 0)
                self.input_init = True


if __name__ == "__main__":
    MAP_SCALE = config.MAP_SCALE
    ROAD_MAP = np.load(f'map/drivingline_map_x{MAP_SCALE}.npy')
    ROUTE_DIST_X1 = np.load('tmp/routed_road.npy')

    ad_car = NOA_MODEL(roadmap=ROAD_MAP, roadcenter_distance_x4=8, running_herz=60)
    ad_car.update_route_map(ROUTE_DIST_X1)

    try:
        ad_car.run()
    except KeyboardInterrupt:
        print("\n[終了] Ctrl+C により終了します")
