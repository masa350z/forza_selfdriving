import numpy as np
import os
from modules import decode_forza_udp
from multiprocessing import shared_memory

# === 定数定義 ===
MAP_FILE = "map/road_map.dat"
DTYPE = np.uint16
SCALE = 10
MAP_SIZE_X = 17000 * SCALE
MAP_OFFSET_X = 10000 * SCALE
MAP_SIZE_Z = 10000 * SCALE
MAP_OFFSET_Z = 4300 * SCALE
SHM_NAME = 'Global\\forza_shm'
SHM_BUFFER_SIZE = 1024
DOWN_SCALE = 100  # 可視化用の圧縮率


# === 圧縮・展開関数 ===
def compress_info16(driving_line, is_dirt, pos_y):
    if not (0 <= driving_line <= 0x7F):
        raise ValueError("driving_line must be in 0〜127 (7bit)")
    if not (is_dirt in (0, 1)):
        raise ValueError("is_dirt must be 0 or 1")
    if not (0 <= pos_y <= 0xFF):
        raise ValueError("pos_y must be in 0〜255 (8bit)")
    return (driving_line << 9) | (is_dirt << 8) | pos_y


def extract_driving_line(d16_map):
    return ((d16_map >> 9) & 0x7F).astype(np.uint8)


# === 共有メモリ接続 ===
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    print("共有メモリに接続しました")
except FileNotFoundError:
    print("共有メモリが存在しません")
    exit(1)

# === マップ初期化（memmap） ===
if not os.path.exists(MAP_FILE):
    fp = np.memmap(MAP_FILE, dtype=DTYPE, mode='w+',
                   shape=(MAP_SIZE_Z, MAP_SIZE_X))
    fp.flush()

road_map = np.memmap(MAP_FILE, dtype=DTYPE, mode='r+',
                     shape=(MAP_SIZE_Z, MAP_SIZE_X))


# === マップ更新 ===
def update_road_map(pos_x, pos_z, compressed_value):
    ix = int(pos_x * SCALE + MAP_OFFSET_X)
    iz = int(pos_z * SCALE + MAP_OFFSET_Z)
    if 0 <= ix < MAP_SIZE_X and 0 <= iz < MAP_SIZE_Z:
        road_map[iz, ix] = compressed_value


# === 自動記録ループ ===
def main_loop():
    input_init = False

    try:
        while True:
            raw = bytes(shm.buf[:SHM_BUFFER_SIZE])
            data = decode_forza_udp(raw)
            driving_line = data['NormalizedDrivingLine']
            pos_x = data['PositionX']
            pos_y = data['PositionY']
            pos_z = data['PositionZ']

            if pos_x == 0 and pos_y == 0 and pos_z == 0 and driving_line == 0:
                if not input_init:
                    print('保存中')
                    road_map.flush()
                    print('保存完了')
                    input_init = True

            else:
                pos_z = data['PositionZ']
                pos_y = int(data['PositionY'] / 4)
                driving_line = 127 - int(abs(driving_line))
                is_dirt = int(data['SurfaceRumbleFrontLeft'] != 0)

                compressed = compress_info16(driving_line, is_dirt, pos_y)
                update_road_map(pos_x, pos_z, compressed)

                input_init = False
            print(f"({pos_x:.2f}, {pos_z:.2f}, {pos_y}, {driving_line})", end='\r')

    except KeyboardInterrupt:
        print("\n[終了] Ctrl+C により終了します")


# === 実行 ===
if __name__ == "__main__":
    main_loop()
