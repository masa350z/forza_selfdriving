import struct

k_kmh = (2.2*1.02)*1.6
k_gforce = 1000/(60*60)

# フォーマットに基づいてアンパックする関数
def decode_forza_udp(data, buffer_offset=12):
    # Sledフォーマットの解析
    sled_data = struct.unpack(
        'iIfffffffffffffffffffffffffffiiiiffffffffffffffffffffiiiii',
        data[:232]  # Sledフォーマットは最初の232バイト
    )

    # Dashフォーマットの解析
    dash_data = struct.unpack(
        'fffffffffffffffffHBBBBBBbbb',
        data[232 + buffer_offset:311 + buffer_offset]  # Dashフォーマットはその後の79バイト
    )

    # Sledフォーマットのデコード
    decoded_data = {
        'IsRaceOn': sled_data[0],
        'TimestampMS': sled_data[1],
        'EngineMaxRpm': sled_data[2],
        'EngineIdleRpm': sled_data[3],
        'CurrentEngineRpm': sled_data[4],
        'AccelerationX': sled_data[5],
        'AccelerationY': sled_data[6],
        'AccelerationZ': sled_data[7],
        'VelocityX': sled_data[8],
        'VelocityY': sled_data[9],
        'VelocityZ': sled_data[10],
        'AngularVelocityX': sled_data[11],
        'AngularVelocityY': sled_data[12],
        'AngularVelocityZ': sled_data[13],
        'Yaw': sled_data[14],
        'Pitch': sled_data[15],
        'Roll': sled_data[16],
        'NormalizedSuspensionTravelFrontLeft': sled_data[17],
        'NormalizedSuspensionTravelFrontRight': sled_data[18],
        'NormalizedSuspensionTravelRearLeft': sled_data[19],
        'NormalizedSuspensionTravelRearRight': sled_data[20],
        'TireSlipRatioFrontLeft': sled_data[21],
        'TireSlipRatioFrontRight': sled_data[22],
        'TireSlipRatioRearLeft': sled_data[23],
        'TireSlipRatioRearRight': sled_data[24],
        'WheelRotationSpeedFrontLeft': sled_data[25],
        'WheelRotationSpeedFrontRight': sled_data[26],
        'WheelRotationSpeedRearLeft': sled_data[27],
        'WheelRotationSpeedRearRight': sled_data[28],
        'WheelOnRumbleStripFrontLeft': sled_data[29],
        'WheelOnRumbleStripFrontRight': sled_data[30],
        'WheelOnRumbleStripRearLeft': sled_data[31],
        'WheelOnRumbleStripRearRight': sled_data[32],
        'WheelInPuddleDepthFrontLeft': sled_data[33],
        'WheelInPuddleDepthFrontRight': sled_data[34],
        'WheelInPuddleDepthRearLeft': sled_data[35],
        'WheelInPuddleDepthRearRight': sled_data[36],
        'SurfaceRumbleFrontLeft': sled_data[37],
        'SurfaceRumbleFrontRight': sled_data[38],
        'SurfaceRumbleRearLeft': sled_data[39],
        'SurfaceRumbleRearRight': sled_data[40],
        'TireSlipAngleFrontLeft': sled_data[41],
        'TireSlipAngleFrontRight': sled_data[42],
        'TireSlipAngleRearLeft': sled_data[43],
        'TireSlipAngleRearRight': sled_data[44],
        'TireCombinedSlipFrontLeft': sled_data[45],
        'TireCombinedSlipFrontRight': sled_data[46],
        'TireCombinedSlipRearLeft': sled_data[47],
        'TireCombinedSlipRearRight': sled_data[48],
        'SuspensionTravelMetersFrontLeft': sled_data[49],
        'SuspensionTravelMetersFrontRight': sled_data[50],
        'SuspensionTravelMetersRearLeft': sled_data[51],
        'SuspensionTravelMetersRearRight': sled_data[52],
        'CarOrdinal': sled_data[53],
        'CarClass': sled_data[54],
        'CarPerformanceIndex': sled_data[55],
        'DrivetrainType': sled_data[56],
        'NumCylinders': sled_data[57],
        'PositionX': dash_data[0],
        'PositionY': dash_data[1],
        'PositionZ': dash_data[2],
        'Speed': dash_data[3],
        'Power': dash_data[4],
        'Torque': dash_data[5],
        'TireTempFrontLeft': dash_data[6],
        'TireTempFrontRight': dash_data[7],
        'TireTempRearLeft': dash_data[8],
        'TireTempRearRight': dash_data[9],
        'Boost': dash_data[10],
        'Fuel': dash_data[11],
        'Distance': dash_data[12],
        'BestLapTime': dash_data[13],
        'LastLapTime': dash_data[14],
        'CurrentLapTime': dash_data[15],
        'CurrentRaceTime': dash_data[16],
        'LapNumber': dash_data[17],
        'RacePosition': dash_data[18],
        'Accel': dash_data[19],
        'Brake': dash_data[20],
        'Clutch': dash_data[21],
        'HandBrake': dash_data[22],
        'Gear': dash_data[23],
        'Steer': dash_data[24],
        'NormalizedDrivingLine': dash_data[25],
        'NormalizedAIBrakeDifference': dash_data[26],
    }

    return decoded_data

def decode_forza_speed(data):
    """
    Forza MotorsportのUDPデータから速度(km/h)を抽出して返す関数。

    内部で decode_forza_udp を用いて生のデータをデコードし、
    'Speed' フィールドを km/h に換算して返す。

    Args:
        data (bytes): ForzaからのUDP生データ。

    Returns:
        float: 速度(km/h)。
    """
    raw_data = decode_forza_udp(data)

    speed_km_h = raw_data['Speed'] * k_kmh

    return speed_km_h 

def decode_forza_gforce(data):
    """
    Forza MotorsportのUDPデータから前後G、左右Gを抽出して返す関数。

    内部で decode_forza_udp を用いて生のデータをデコードし、
    'AccelerationZ' と 'AccelerationX' を用いて前後方向、左右方向の
    G-forceを計算する。

    Args:
        data (bytes): ForzaからのUDP生データ。

    Returns:
        tuple: (forward_backward_g, side_to_side_g)
            - forward_backward_g (float): 前後方向のG値 (G)
            - side_to_side_g (float): 左右方向のG値 (G)
    """
    raw_data = decode_forza_udp(data)
    
    # 前後方向GをZ軸から算出 (符号反転などは車両座標系に合わせる)
    forward_backward_g = -raw_data['AccelerationZ'] * k_kmh * k_gforce / 9.8

    # 左右方向GをX軸から算出
    side_to_side_g = -raw_data['AccelerationX'] * k_kmh * k_gforce / 9.8

    return forward_backward_g, side_to_side_g