from modules import UDP_Reader

reader = UDP_Reader()

while True:
    data = reader.read_data()

    a = data['PositionX']
    b = data['PositionY']
    c = data['PositionZ']
    d = data['NormalizedDrivingLine']
    
    # a = data['SurfaceRumbleFrontLeft']
    # b = data['SurfaceRumbleFrontRight']
    # c = data['SurfaceRumbleRearLeft']
    # d = data['Steer']
    
    

    print(f" ({a:.2f}, {b:.2f}, {c:.2f}, {d:.2f})", end='\r')
