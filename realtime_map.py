import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import shared_memory

from modules import calc_next_point, find_nearest_road_pixel, decode_forza_udp
import config

ROAD_CENTER_VALUE_THRESHOLD = config.ROAD_CENTER_VALUE_THRESHOLD
MAP_OFFSET_X = config.MAP_OFFSET_X
MAP_OFFSET_Z = config.MAP_OFFSET_Z
SHM_NAME = config.SHM_NAME
SHM_BUFFER_SIZE = config.SHM_BUFFER_SIZE

# === 設定 ===
MAP_FILE = 'map/palacio_oval.npy'
QUAD_FILE01 = 'map/quadrant_map_10.npy'
QUAD_FILE02 = 'map/quadrant_map_30.npy'
QUAD_FILE03 = 'map/quadrant_map_50.npy'
QUAD_FILE04 = 'map/quadrant_map_100.npy'

WINDOW = 200  # 表示範囲 半サイズ（→ 500×500）

# === マップ読み込み ===
road_map = np.load(MAP_FILE) > ROAD_CENTER_VALUE_THRESHOLD
quadrant_map01 = np.load(QUAD_FILE01)
quadrant_map02 = np.load(QUAD_FILE02)
quadrant_map03 = np.load(QUAD_FILE03)
quadrant_map04 = np.load(QUAD_FILE04)

# === 共有メモリ接続 ===
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    print("共有メモリに接続しました")
except FileNotFoundError:
    print("共有メモリが存在しません")
    exit(1)


# === 描画用 ===
fig, ax = plt.subplots()
img = ax.imshow(np.zeros((WINDOW*2, WINDOW*2), dtype=np.uint8),
                cmap='gray', vmin=0, vmax=1, origin='lower')

nearest_dot, = ax.plot([], [], color='fuchsia', marker='o', markersize=5)

car_dot, = ax.plot([], [], color='white', marker='o', markersize=4)

goal01_dot, = ax.plot([], [], color='aqua', marker='o', markersize=4)
goal02_dot, = ax.plot([], [], color='lime', marker='o', markersize=4)
goal03_dot, = ax.plot([], [], color='orange', marker='o', markersize=4)
goal04_dot, = ax.plot([], [], color='crimson', marker='o', markersize=4)

ax.set_title("Local Navigation View (Live)")


def set_circle(quadrant_map, goal_dot, nearest_pixel_x, nearest_pixel_z, yaw, x0, z0):
    # 目標点を地図相対座標に変換
    target = calc_next_point(quadrant_map,
                            nearest_pixel_x - MAP_OFFSET_X,
                            nearest_pixel_z - MAP_OFFSET_Z,
                            yaw,
                            MAP_OFFSET_X,
                            MAP_OFFSET_Z)

    if target is not None:
        gx = target[0] + MAP_OFFSET_X - x0
        gz = target[1] + MAP_OFFSET_Z - z0
        goal_dot.set_data([gx], [gz])

    else:
        goal_dot.set_data([], [])

# === アニメーション更新 ===
def update(frame):
    raw = bytes(shm.buf[:SHM_BUFFER_SIZE])
    data = decode_forza_udp(raw)

    pos_x = data['PositionX']
    pos_z = data['PositionZ']
    yaw = data['Yaw']

    nearest_pixel, distance = find_nearest_road_pixel(road_map,
                                                      pos_x + MAP_OFFSET_X,
                                                      pos_z + MAP_OFFSET_Z)

    if nearest_pixel is not None:
        nearest_pixel_x = nearest_pixel[0]
        nearest_pixel_z = nearest_pixel[1]

        cx = int(pos_x + MAP_OFFSET_X)
        cz = int(pos_z + MAP_OFFSET_Z)

        # マップ切り出し
        z0, z1 = cz - WINDOW, cz + WINDOW
        x0, x1 = cx - WINDOW, cx + WINDOW

        extract = np.zeros((2*WINDOW, 2*WINDOW), dtype=np.uint8)
        if 0 <= z0 and z1 <= road_map.shape[0] and 0 <= x0 and x1 <= road_map.shape[1]:
            extract = road_map[z0:z1, x0:x1].astype(np.uint8)

        img.set_data(extract)

        # 自車点（中心）
        car_dot.set_data([WINDOW], [WINDOW])
        nearest_dot.set_data([nearest_pixel_x - x0], [nearest_pixel_z- z0])

        set_circle(quadrant_map01, goal01_dot, nearest_pixel_x, nearest_pixel_z, yaw, x0, z0)
        set_circle(quadrant_map02, goal02_dot, nearest_pixel_x, nearest_pixel_z, yaw, x0, z0)
        set_circle(quadrant_map03, goal03_dot, nearest_pixel_x, nearest_pixel_z, yaw, x0, z0)
        set_circle(quadrant_map04, goal04_dot, nearest_pixel_x, nearest_pixel_z, yaw, x0, z0)

    return [img, car_dot, goal01_dot]


# === 実行 ===
ani = animation.FuncAnimation(fig, update, interval=100)
plt.show()
