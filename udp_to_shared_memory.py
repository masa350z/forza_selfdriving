"""UDPで受信したバイナリを共有メモリに書き込むプログラム"""

import socket
from multiprocessing import shared_memory
import config

SHM_NAME = config.SHM_NAME
SHM_BUFFER_SIZE = config.SHM_BUFFER_SIZE
FORZA_UDP_PORT = config.FORZA_UDP_PORT

# 共有メモリ作成(既に存在する場合は再作成)
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_BUFFER_SIZE)
except FileExistsError:
    existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
    existing_shm.close()
    existing_shm.unlink()
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=SHM_BUFFER_SIZE)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', FORZA_UDP_PORT))
print(f"UDPポート {FORZA_UDP_PORT} をリッスン中...")

try:
    while True:
        data, _ = sock.recvfrom(SHM_BUFFER_SIZE)
        shm.buf[:len(data)] = data
        shm.buf[len(data):] = b'\x00' * (SHM_BUFFER_SIZE - len(data))  # 残りをゼロクリア
except KeyboardInterrupt:
    print("終了処理中...")
finally:
    shm.close()
    shm.unlink()
    sock.close()
