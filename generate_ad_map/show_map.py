from PIL import Image
import numpy as np
import os
from tqdm import tqdm

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from modules import extract_driving_line_from_d32


# === 定数定義 ===
MAP_FILE = "../map/road_map_x4.dat"
OUTPUT_FILE = "../img/road_map_view.png"
DTYPE = np.uint32
SCALE = 4
MAP_SIZE_X = 17000 * SCALE
OFFSET_X = 10000 * SCALE
MAP_SIZE_Z = 10000 * SCALE
OFFSET_Z = 4300 * SCALE
DOWN_SCALE = 100  # 可視化用の圧縮率


# === マップ初期化（memmap） ===
if not os.path.exists(MAP_FILE):
    fp = np.memmap(MAP_FILE, dtype=DTYPE, mode='w+',
                   shape=(MAP_SIZE_X, MAP_SIZE_Z))
    fp.flush()

road_map = np.memmap(MAP_FILE, dtype=DTYPE, mode='r+',
                     shape=(MAP_SIZE_X, MAP_SIZE_Z))

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
            resized_img[j, i] = np.max(extract_driving_line_from_d32(block))

    Image.fromarray(resized_img[:, ::-1].T.astype(np.uint8)).save(OUTPUT_FILE)
    print(f"[保存] {OUTPUT_FILE}, {OUTPUT_FILE}")
