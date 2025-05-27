import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from modules import calc_next_point_differ, find_nearest_road_pixel, \
    UDP_Reader, convert_forzaposition_to_arraycoord
import config

ROAD_CENTER_DISTANCE_THRESHOLD = 110
MAP_OFFSET_X = config.MAP_OFFSET_X
MAP_OFFSET_Z = config.MAP_OFFSET_Z
SHM_NAME = config.SHM_NAME

# === 設定 ===
MAP_FILE_X4 = 'map/palacio_simple_x4.npy'
MAP_FILE_X1 = 'map/palacio_simple_x1.npy'

# QUAD_FILE01 = 'map/oval_backup/quadrant_map_10_x1.npy'
QUAD_FILE01 = 'map/quadrant_map_temp.npy'
# QUAD_FILE02 = 'map/oval_backup/quadrant_map_30_x1.npy'
# QUAD_FILE03 = 'map/oval_backup/quadrant_map_50_x1.npy'
# QUAD_FILE04 = 'map/oval_backup/quadrant_map_100_x1.npy'

WINDOW = 200  # 表示範囲 半サイズ

# === マップ読み込み ===
road_map_x4 = np.load(MAP_FILE_X4)
road_map_x1 = np.load(MAP_FILE_X1)
quadrant_map01 = np.load(QUAD_FILE01)
# quadrant_map02 = np.load(QUAD_FILE02)
# quadrant_map03 = np.load(QUAD_FILE03)
# quadrant_map04 = np.load(QUAD_FILE04)


reader = UDP_Reader()


# === 描画用 ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


def init_axes(ax):
    img = ax.imshow(np.zeros((WINDOW*2, WINDOW*2), dtype=np.uint8),
                    cmap='gray', vmin=0, vmax=1, origin='lower')
    car_dot, = ax.plot([], [], color='white', marker='o', markersize=4)
    nearest_dot, = ax.plot([], [], color='fuchsia', marker='o', markersize=5)
    goal01_dot, = ax.plot([], [], color='aqua', marker='o', markersize=4)
    goal02_dot, = ax.plot([], [], color='lime', marker='o', markersize=4)
    goal03_dot, = ax.plot([], [], color='orange', marker='o', markersize=4)
    goal04_dot, = ax.plot([], [], color='crimson', marker='o', markersize=4)
    ax.set_title("Local Navigation View")
    return img, car_dot, nearest_dot, goal01_dot, goal02_dot, goal03_dot, goal04_dot


# 2つの描画セット
img1, car_dot1, nearest_dot1, g01_1, g02_1, g03_1, g04_1 = init_axes(ax1)
img2, car_dot2, nearest_dot2, g01_2, g02_2, g03_2, g04_2 = init_axes(ax2)


def set_circle_x4(quadrant_map, goal_dot, nearest_pixel_x, nearest_pixel_z, yaw, x0, z0):
    nearest_pixel_x = int(nearest_pixel_x/4)
    nearest_pixel_z = int(nearest_pixel_z/4)

    x0 = int((x0+WINDOW)/4)
    z0 = int((z0+WINDOW)/4)

    target = calc_next_point_differ(quadrant_map,
                                    nearest_pixel_x,
                                    nearest_pixel_z,
                                    yaw)

    if target is not None:
        gx = ((nearest_pixel_x - x0) + target[0])*4 + WINDOW
        gz = ((nearest_pixel_z - z0) + target[1])*4 + WINDOW

        goal_dot.set_data([gx], [gz])

    else:
        goal_dot.set_data([], [])


def set_circle_x1(quadrant_map, goal_dot, nearest_pixel_x, nearest_pixel_z, yaw, x0, z0):
    x0 = x0+WINDOW
    z0 = z0+WINDOW

    target = calc_next_point_differ(quadrant_map,
                                    nearest_pixel_x,
                                    nearest_pixel_z,
                                    yaw)

    if target is not None:
        gx = ((nearest_pixel_x - x0) + target[0]) + WINDOW
        gz = ((nearest_pixel_z - z0) + target[1]) + WINDOW

        goal_dot.set_data([gx], [gz])

    else:
        goal_dot.set_data([], [])


def update_x4(array_coord_x, array_coord_z, yaw):
    nearest_coord, _ = find_nearest_road_pixel(
        road_map_x4, array_coord_x, array_coord_z, ROAD_CENTER_DISTANCE_THRESHOLD
    )

    if nearest_coord is not None:
        nearest_pixel_x, nearest_pixel_z = nearest_coord

        x0, x1 = array_coord_x - WINDOW, array_coord_x + WINDOW
        z0, z1 = array_coord_z - WINDOW, array_coord_z + WINDOW

        extract = np.zeros((2*WINDOW, 2*WINDOW), dtype=np.uint8)
        if 0 <= x0 and x1 <= road_map_x4.shape[0] and 0 <= z0 and z1 <= road_map_x4.shape[1]:
            extract = road_map_x4[x0:x1, z0:z1].astype(np.uint8)

        img1.set_data(extract.T)
        car_dot1.set_data([WINDOW], [WINDOW])
        nearest_dot1.set_data([nearest_pixel_x - x0], [nearest_pixel_z - z0])

        for quadrant_map, g0 in [(quadrant_map01, g01_1),
                                 #  (quadrant_map02, g02_1),
                                 #  (quadrant_map03, g03_1),
                                 #  (quadrant_map04, g04_1),
                                 ]:

            set_circle_x4(quadrant_map, g0,
                          nearest_pixel_x,
                          nearest_pixel_z,
                          yaw, x0, z0)


def update_x1(array_coord_x, array_coord_z, yaw):
    nearest_coord, _ = find_nearest_road_pixel(
        road_map_x1, array_coord_x, array_coord_z, ROAD_CENTER_DISTANCE_THRESHOLD
    )

    if nearest_coord is not None:
        nearest_pixel_x, nearest_pixel_z = nearest_coord

        x0, x1 = array_coord_x - WINDOW, array_coord_x + WINDOW
        z0, z1 = array_coord_z - WINDOW, array_coord_z + WINDOW

        extract = np.zeros((2*WINDOW, 2*WINDOW), dtype=np.uint8)
        if 0 <= x0 and x1 <= road_map_x1.shape[0] and 0 <= z0 and z1 <= road_map_x1.shape[1]:
            extract = road_map_x1[x0:x1, z0:z1].astype(np.uint8)

        img2.set_data(extract.T)
        car_dot2.set_data([WINDOW], [WINDOW])
        nearest_dot2.set_data([nearest_pixel_x - x0], [nearest_pixel_z - z0])

        for quadrant_map, g0 in [(quadrant_map01, g01_2),
                                 #  (quadrant_map02, g02_2),
                                 #  (quadrant_map03, g03_2),
                                 #  (quadrant_map04, g04_2),
                                 ]:

            set_circle_x1(quadrant_map, g0,
                          nearest_pixel_x,
                          nearest_pixel_z,
                          yaw, x0, z0)


# === アニメーション更新 ===
def update(frame):
    data = reader.read_data()

    pos_x = data['PositionX']
    pos_z = data['PositionZ']
    yaw = data['Yaw']

    array_coord_x_x4, array_coord_z_x4 = convert_forzaposition_to_arraycoord(
        pos_x, pos_z, MAP_OFFSET_X, MAP_OFFSET_Z, 4
    )
    array_coord_x_x1, array_coord_z_x1 = convert_forzaposition_to_arraycoord(
        pos_x, pos_z, MAP_OFFSET_X, MAP_OFFSET_Z, 1
    )

    update_x4(array_coord_x_x4, array_coord_z_x4, yaw)
    update_x1(array_coord_x_x1, array_coord_z_x1, yaw)

    return [
        img1, car_dot1, nearest_dot1, g01_1, g02_1, g03_1, g04_1,
        img2, car_dot2, nearest_dot2, g01_2, g02_2, g03_2, g04_2
    ]


# === 実行 ===
ani = animation.FuncAnimation(fig, update, interval=100)
plt.show()
