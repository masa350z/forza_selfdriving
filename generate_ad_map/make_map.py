from PIL import Image
import numpy as np
import sys
import os

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from modules import UDP_Reader, compute_radius, compress_info32, convert_forzaposition_to_arraycoord, extract_driving_line_from_d32
import config


output_x4map_path = '../map/drivingline_map_x4.npy'
output_x1map_path = '../map/drivingline_map_x1.npy'
output_x1img_path = '../img/drivingline_map_x1.png'

MAP_SCALE = config.MAP_SCALE
map_source_path = f"../map/road_map_x{MAP_SCALE}.dat"

scaled_mapsize_x = config.MAP_SIZE_X*MAP_SCALE
scaled_mapsize_z = config.MAP_SIZE_Z*MAP_SCALE


def init_roadmap_memmap(map_source_path, map_dtype=np.uint32):
    # === マップ初期化(memmap) ===
    if not os.path.exists(map_source_path):
        fp = np.memmap(map_source_path, dtype=map_dtype, mode='w+',
                       shape=(scaled_mapsize_x, scaled_mapsize_z))
        fp.flush()

    road_map = np.memmap(map_source_path, dtype=map_dtype, mode='r+',
                         shape=(scaled_mapsize_x, scaled_mapsize_z))

    return road_map


def update_road_map(pos_x, pos_z, compressed_value):
    # === マップ更新 ===
    ix, iz = convert_forzaposition_to_arraycoord(pos_x, pos_z,
                                                 config.MAP_OFFSET_X, config.MAP_OFFSET_Z,
                                                 MAP_SCALE)

    if 0 <= ix < scaled_mapsize_x and 0 <= iz < scaled_mapsize_z:
        road_map[ix, iz] = compressed_value


def convert_dat_to_npy(map_dtype=np.uint32):
    road_map = np.memmap(map_source_path, dtype=map_dtype,
                         mode='r+', shape=(scaled_mapsize_x, scaled_mapsize_z))
    np.save(map_source_path.replace('dat', 'npy'), road_map)

    return road_map


def down_scale_map(road_map, down_scale):
    xlen, zlen = road_map.shape
    resized_xlen = int(xlen / down_scale)
    resized_zlen = int(zlen / down_scale)

    new_shape = (
        resized_xlen, down_scale,
        resized_zlen, down_scale
    )

    # reshapeしてから max を取る(axis=(1, 3) は downscale 部分)
    road_map = extract_driving_line_from_d32(road_map)
    resized_img = road_map[:resized_xlen*down_scale, :resized_zlen*down_scale].reshape(new_shape)
    resized_map = resized_img.max(axis=(1, 3))

    return resized_map


def make_scaled_map(npy_map, scale, output_npy_path, output_png_path=None):
    resized_map = down_scale_map(npy_map, scale)
    np.save(output_npy_path, resized_map)
    if output_png_path is not None:
        Image.fromarray((resized_map.T[::-1]*2).astype(np.uint8)).save(output_png_path)


def make_map(road_map):
    reader = UDP_Reader()
    input_init = False

    while True:
        data = reader.read_data()
        driving_line = data['NormalizedDrivingLine']
        pos_x = data['PositionX']
        pos_y = data['PositionY']
        pos_z = data['PositionZ']
        yaw = data['Yaw']
        radius = compute_radius(data)
        speed = data['Speed']/config.MPH_TO_MPS*config.MPH_TO_KMPH  # これでkm/hになる
        if radius == float('inf'):
            radius = 255
        else:
            radius = int(radius)

        if (pos_x == 0 and pos_y == 0 and pos_z == 0 and driving_line == 0):
            if not input_init:
                print('保存中')
                road_map.flush()
                print('保存完了')
                input_init = True

        elif speed >= 1:
            pos_y = int(data['PositionY'] / 4)
            driving_line = 127 - int(abs(driving_line))
            is_dirt = int(data['SurfaceRumbleFrontLeft'] != 0)
            radius = radius if radius <= 255 else 255
            yaw = int((yaw+np.pi)/(2*np.pi)*255)
            yaw = yaw if yaw <= 255 else 255

            compressed = compress_info32(
                driving_line, is_dirt, pos_y, radius, yaw)
            update_road_map(pos_x, pos_z, compressed)

            input_init = False

            print(
                f"({pos_x:.2f}, {pos_z:.2f}, {pos_y}, {driving_line}, {radius}, {yaw})", end='\r')


# === 実行 ===
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'convert':
            # 変換モード
            print("変換モード")
            convert_dat_to_npy()
        else:
            print("変換モードオプション: convert")
    else:
        road_map = init_roadmap_memmap(map_source_path)
        try:
            make_map(road_map)
        except KeyboardInterrupt:
            print("\n[終了] Ctrl+C により終了します")
            road_map.flush()

            print("npyに変換中")
            npy_map = convert_dat_to_npy()
            print("変換完了")

            print("4倍精度マップ作成中")
            make_scaled_map(npy_map, 1, output_x4map_path)
            print("4倍精度マップ作成完了")

            print("等倍マップ作成中")
            make_scaled_map(npy_map, MAP_SCALE, output_x1map_path, output_x1img_path)
            print("等倍マップ作成完了")
