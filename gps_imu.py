import numpy as np
import pandas as pd

def find_steady_window(data, window_size=6, step_size=2, sample_rate=50):
    num_samples = window_size * sample_rate
    step_samples = step_size * sample_rate
    min_std = float('inf')
    steady_window = None
    
    for i in range(0, len(data) - num_samples, step_samples):
        window = data[i:i + num_samples]
        std = np.std(window)
        
        if std < min_std:
            min_std = std
            steady_window = window
            
    return steady_window.mean(axis=0)

"""This is only for accelerometer data, Assume you don't have Gyro data and Magnetometer data.""" 
def calculate_euler_angles(ax, ay, az):
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    print(roll," ",pitch)
    return roll, pitch

def complementary_filter(prev_angles, gyro_rates, dt, alpha):
    # Gyro integration
    gyro_angles = prev_angles + gyro_rates * dt
    # Combine gyro and accel angles using a complementary filter
    fused_angles = alpha * gyro_angles + (1 - alpha) * prev_angles
    return fused_angles

def reorient_step_1(data, roll, pitch):
    r_matrix = np.array([[1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)],
                         [0, np.sin(roll), np.cos(roll)]])
    
    p_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                         [0, 1, 0],
                         [-np.sin(pitch), 0, np.cos(pitch)]])
    
    rp_matrix = p_matrix @ r_matrix
    reoriented_data = data @ rp_matrix.T
    return reoriented_data

def calculate_yaw_angle(reoriented_data, threshold):
    horizontal_data = reoriented_data[:, :2]
    accel_mag = np.linalg.norm(horizontal_data, axis=1)
    max_index = np.argmax(accel_mag)
    
    if accel_mag[max_index] > threshold:
        ax, ay = horizontal_data[max_index]
        yaw = np.arctan2(ay, ax)
        return yaw
    else:
        return None

def reorient_step_2(reoriented_data, yaw):
    y_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                         [np.sin(yaw), np.cos(yaw), 0],
                         [0, 0, 1]])
    
    reoriented_data_final = reoriented_data @ y_matrix.T
    return reoriented_data_final

def reorient_acceleration_data(data, threshold=1):
    # Algorithm A1
    gravity = find_steady_window(data)
    roll, pitch = calculate_euler_angles(*gravity)
    reoriented_data = reorient_step_1(data, roll, pitch)

    # Algorithm A2
    yaw = calculate_yaw_angle(reoriented_data, threshold)
    
    if yaw is not None:
        reoriented_data_final = reorient_step_2(reoriented_data, yaw)
    else:
        reoriented_data_final = reoriented_data
        
    return reoriented_data_final

def calculate_euler_angles_imu(acc_data, gyro_data, mag_data, dt, alpha=0.98):
    acc_angles = np.zeros_like(acc_data)
    fused_angles = np.zeros_like(acc_data)
    
    for i, (ax, ay, az) in enumerate(acc_data):
        roll, pitch = calculate_euler_angles(ax, ay, az)
        acc_angles[i] = np.array([roll, pitch])
        
    for i in range(1, len(acc_data)):
        gyro_rates = gyro_data[i] * np.pi / 180  # Convert to radians
        fused_angles[i] = complementary_filter(fused_angles[i - 1], gyro_rates, dt, alpha)
        
    yaw_angles = np.zeros(len(acc_data))
    
    for i, (mx, my, mz) in enumerate(mag_data):
        yaw_angles[i] = np.arctan2(my, mx)
    
    euler_angles = np.column_stack((fused_angles, yaw_angles))
    return euler_angles

def reorient_imu_data(data, euler_angles):
    r_matrix = np.array([[1, 0, 0],
                         [0, np.cos(euler_angles[:, 0]), -np.sin(euler_angles[:, 0])],
                         [0, np.sin(euler_angles[:, 0]), np.cos(euler_angles[:, 0])]])
    
    p_matrix = np.array([[np.cos(euler_angles[:, 1]), 0, np.sin(euler_angles[:, 1])],
                         [0, 1, 0],
                         [-np.sin(euler_angles[:, 1]), 0, np.cos(euler_angles[:, 1])]])
    
    y_matrix = np.array([[np.cos(euler_angles[:, 2]), -np.sin(euler_angles[:, 2]), 0],
                         [np.sin(euler_angles[:, 2]), np.cos(euler_angles[:, 2]), 0],
                         [0, 0, 1]])
    
    rp_matrix = p_matrix @ r_matrix
    rpy_matrix = y_matrix @ rp_matrix
    reoriented_data = data @ rpy_matrix.transpose((0, 2, 1))
    
    return reoriented_data

def reorient_acceleration_data_imu(acc_data, gyro_data, mag_data, dt, alpha=0.98):
    euler_angles = calculate_euler_angles_imu(acc_data, gyro_data, mag_data, dt, alpha)
    reoriented_data = reorient_imu_data(acc_data, euler_angles)
    return reoriented_data

# Read raw IMU data from CSV
imu_data = pd.read_csv("./NYCU_GPS/lab1/2_after.csv").to_numpy()
gyro_data = imu_data[:, 1:4]
acc_data = imu_data[:, 5:8]
print(acc_data)
mag_data = imu_data[:, 9:12]
print(mag_data)
# Read raw acceleration data (ax, ay, az) from CSV


# Reorient acceleration data
reoriented_data = reorient_acceleration_data_imu(acc_data, gyro_data, mag_data,dt=0.1, alpha=0.98)
np.savetxt("./NYCU_GPS/lab1/reoriented_data.csv", reoriented_data, delimiter=",")
