from PIL import Image
import numpy as np
import os
from multiprocessing import shared_memory
from tqdm import tqdm


def extract_driving_line_from_d16(d16: np.ndarray) -> np.ndarray:
    """uint16圧縮マップからdriving_line (7bit) 情報だけを取り出す

    Args:
        d16 (np.ndarray): shape=(H, W), dtype=uint16 の2次元配列

    Returns:
        np.ndarray: shape=(H, W), dtype=uint8 の driving_line 抽出結果
    """
    if d16.dtype != np.uint16:
        raise ValueError("入力配列は dtype=uint16 である必要があります")

    driving_line_map = ((d16 >> 9) & 0x7F).astype(np.uint8)
    return driving_line_map


# === 定数定義 ===
MAP_FILE = "map/road_map.dat"
OUTPUT_FILE = "img/road_map_view.png"
DTYPE = np.uint16
SCALE = 10
MAP_SIZE_X = 17000 * SCALE
OFFSET_X = 10000 * SCALE
MAP_SIZE_Z = 10000 * SCALE
OFFSET_Z = 4300 * SCALE
SHM_NAME = 'forza_shm'
BUFFER_SIZE = 1024
DOWN_SCALE = 400  # 可視化用の圧縮率


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


xlen, zlen = road_map.shape
resized_xlen = int(xlen / DOWN_SCALE)
resized_zlen = int(zlen / DOWN_SCALE)

while True:
    resized_img = np.zeros((resized_xlen, resized_zlen), dtype=np.uint8)
    for i in tqdm(range(resized_zlen)):
        for j in range(resized_xlen):
            block = road_map[
                j * DOWN_SCALE:(j + 1) * DOWN_SCALE,
                i * DOWN_SCALE:(i + 1) * DOWN_SCALE
            ]
            resized_img[j, i] = np.max(extract_driving_line_from_d16(block))

    Image.fromarray(resized_img[::-1].astype(np.uint8)).save(OUTPUT_FILE)
    print(f"[保存] {OUTPUT_FILE}, {OUTPUT_FILE}")
