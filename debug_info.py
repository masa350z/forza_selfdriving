from modules import UDP_Reader, compute_radius

reader = UDP_Reader()

while True:
    data = reader.read_data()

    # a = data['PositionX']
    # b = data['PositionY']
    # c = data['PositionZ']
    # d = data['Speed']/0.44704*1.609
    # radius = compute_radius(data)

    a = data['TireSlipRatioFrontLeft']
    b = data['TireSlipRatioFrontRight']
    c = data['TireSlipRatioRearLeft']
    d = data['TireSlipRatioRearRight']

    print(f" ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f})", end='\r')
