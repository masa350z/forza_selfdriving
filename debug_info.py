from modules import UDP_Reader
import time


reader = UDP_Reader()

while True:
    data = reader.read_data()

    a = data['PositionX']
    b = data['PositionY']
    c = data['PositionZ']
    d = data['Yaw']
    # radius = compute_radius(data)

    # a = data['TireSlipRatioFrontLeft']
    # b = data['TireSlipRatioFrontRight']
    # c = data['TireSlipRatioRearLeft']
    # d = data['TireSlipRatioRearRight']

    print(f" ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f})", end='\r')
    # print('=====================')
    # print(data)
    # print('------------------------')
    # temp = 0
    # for i in list(data.keys()):
    #     temp += data[i]
    # print(temp)
    # print(temp == data['TimestampMS'])
    time.sleep(1)
