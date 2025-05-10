"""Windows単体で完結するForza Horizon 5自動運転システム

   - UDPでForzaのテレメトリを受信
   - PID制御でステア・スロットル演算
   - pyxinput + ViGEm で仮想Xboxコントローラに出力
"""
from forza_udp_decoder import decode_forza_udp
from multiprocessing import shared_memory

SHM_NAME = 'forza_shm'
BUFFER_SIZE = 1024

try:
    shm = shared_memory.SharedMemory(name=SHM_NAME)
    print("共有メモリに接続しました")
except FileNotFoundError:
    print("共有メモリが存在しません")
    exit(1)



while True:
    raw = bytes(shm.buf[:BUFFER_SIZE])
    data = decode_forza_udp(raw)

    a = data['PositionX']
    b = data['PositionY']
    c = data['PositionZ']
    d = data['NormalizedDrivingLine']
    
    # a = data['SurfaceRumbleFrontLeft']
    # b = data['SurfaceRumbleFrontRight']
    # c = data['SurfaceRumbleRearLeft']
    # d = data['Steer']
    
    

    print(f" ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f})", end='\r', flush=True)
