"""WSL からの制御信号を受信し仮想コントローラ(vController)へ流す"""

import socket
import struct
from pyxinput import vController
import config

PORT = config.CONTROL_PORT

vc = vController()                          # ViGEm + pyxinput 必須
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
print(f"vController サーバ起動 : UDP {PORT} で待受")

try:
    while True:
        dat, _ = sock.recvfrom(8)           # steer, throttle = 2×float32
        steer, throttle = struct.unpack("ff", dat)

        # 範囲クリッピング
        steer = max(-1.0, min(1.0, steer))
        throttle = max(0.0, min(1.0, throttle))

        vc.set_value("AxisLx", steer)
        vc.set_value("TriggerR", throttle)
except KeyboardInterrupt:
    pass
