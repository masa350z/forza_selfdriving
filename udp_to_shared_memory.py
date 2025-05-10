"""UDPで受信したバイナリを共有メモリに書き込むプログラム"""

import socket
import time
from multiprocessing import shared_memory

SHM_NAME = 'forza_shm'
BUFFER_SIZE = 1024  # UDP受信サイズ
UDP_PORT = 5555     # Forza等の送信側に合わせて設定

# 共有メモリ作成（既に存在する場合は再作成）
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=BUFFER_SIZE)
except FileExistsError:
    existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
    existing_shm.close()
    existing_shm.unlink()
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=BUFFER_SIZE)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', UDP_PORT))
print(f"UDPポート {UDP_PORT} をリッスン中...")

try:
    while True:
        data, _ = sock.recvfrom(BUFFER_SIZE)
        shm.buf[:len(data)] = data
        shm.buf[len(data):] = b'\x00' * (BUFFER_SIZE - len(data))  # 残りをゼロクリア
except KeyboardInterrupt:
    print("終了処理中...")
finally:
    shm.close()
    shm.unlink()
    sock.close()
