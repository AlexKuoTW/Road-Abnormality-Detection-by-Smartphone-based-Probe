import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

class GPSVis(object):
    """
        Class for GPS data visualization using pre-downloaded OSM map in image format.
    """
    def __init__(self, data_path, map_path, points):
        """
        :param data_path: Path to file containing GPS records.
        :param map_path: Path to pre-downloaded OSM map in image format.
        :param points: Upper-left, and lower-right GPS points of the map (lat1, lon1, lat2, lon2).
        """
        self.data_path = data_path
        self.points = points
        self.map_path = map_path

        self.result_image = Image
        self.x_ticks = []
        self.y_ticks = []

    def plot_map(self, output='save', save_as='resultMap.png'):
        """
        Method for plotting the map. You can choose to save it in file or to plot it.
        :param output: Type 'plot' to show the map or 'save' to save it.
        :param save_as: Name and type of the resulting image.
        :return:
        """
        self.get_ticks()
        fig, axis1 = plt.subplots(figsize=(10, 10))
        axis1.imshow(self.result_image)
        axis1.set_xlabel('Longitude')
        axis1.set_ylabel('Latitude')
        axis1.set_xticklabels(self.x_ticks)
        axis1.set_yticklabels(self.y_ticks)
        axis1.grid()
        if output == 'save':
            plt.savefig(save_as)
        else:
            plt.show()

    def create_image(self, color, width):
        """
        Create the image that contains the original map and the GPS records.
        :param color: Color of the GPS records.
        :param width: Width of the drawn GPS records.
        :return:
        """
        data = pd.read_csv(self.data_path, names=['LATITUDE', 'LONGITUDE', 'label'], sep=',')
        self.result_image = Image.open(self.map_path, 'r')
        img_points = []
        gps_data = tuple(zip(data['LATITUDE'].values, data['LONGITUDE'].values, data['label'].values))
        draw = ImageDraw.Draw(self.result_image)
        # for d in gps_data:
        #     # for i in range(0, len(data['label'])):
        #     x1, y1 = self.scale_to_img(d, (self.result_image.size[0], self.result_image.size[1]))
        #     draw.ellipse([(x1-2, y1-2),(x1+2, y1+2)], fill=color, width=width)
        #     #print(img_points)
        for lat, lon, size in gps_data:
            x1, y1 = self.scale_to_img((lat, lon), (self.result_image.size[0], self.result_image.size[1]))
            draw.ellipse([(x1-size*2.3, y1-size/1.1), (x1+size*2.3, y1+size/1.1)], fill=color, width=width)

    def scale_to_img(self, lat_lon, h_w):
        """
        Conversion from latitude and longitude to the image pixels.
        It is used for drawing the GPS records on the map image.
        :param lat_lon: GPS record to draw (lat1, lon1).
        :param h_w: Size of the map image (w, h).
        :return: Tuple containing x and y coordinates to draw on map image.
        """
        # https://gamedev.stackexchange.com/questions/33441/how-to-convert-a-number-from-one-min-max-set-to-another-min-max-set/33445
        old = (self.points[2], self.points[0])
        new = (0, h_w[1])
        y = ((lat_lon[0] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        old = (self.points[1], self.points[3])
        new = (0, h_w[0])
        x = ((lat_lon[1] - old[0]) * (new[1] - new[0]) / (old[1] - old[0])) + new[0]
        # y must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        return x, h_w[1] - y

    def get_ticks(self):
        """
        Generates custom ticks based on the GPS coordinates of the map for the matplotlib output.

        """
        self.x_ticks = [map(
            lambda x: round(x, 4),
            np.linspace(self.points[1], self.points[3], num=7))]
        y_ticks = [map(
            lambda x: round(x, 4),
            np.linspace(self.points[2], self.points[0], num=8))]
        # Ticks must be reversed because the orientation of the image in the matplotlib.
        # image - (0, 0) in upper left corner; coordinate system - (0, 0) in lower left corner
        self.y_ticks = sorted(y_ticks, reverse=True)


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
    # print(roll," ",pitch)
    return roll, pitch

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





"""
Offline
"""
# Read raw IMU data from CSV
# df = pd.read_csv("./NYCU_GPS/lab1/2_before_align.csv").to_numpy()
# data = df[:, 4:7]
# print(data)

# # Reorient acceleration data
# reoriented_data = reorient_acceleration_data(data)
# np.savetxt("./NYCU_GPS/lab1/reoriented_data.csv", reoriented_data, delimiter=",")



"""
Try
"""
# import numpy as np

# # 假設加速度數據為 (ax, ay, az)，磁場數據為 (mx, my, mz)
# acceleration = np.array([0.08349609375, -0.9655303955, -0.2501373291]).reshape((3, 1))
# magnetic_field = np.array([51.40674973, -14.74022388, -7.376216888]).reshape((3, 1))

# # 計算車子的旋轉矩陣 R，假設使用歐拉積分法
# yaw = 0.1008
# pitch = 0.0246
# roll = -0.0907
# R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
#               [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
#               [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])

# # 將加速度數據投影到車子的xyz座標系統中
# projected_acceleration = np.dot(R, acceleration)

# # 將磁場數據投影到車子的xyz座標系統中
# # projected_magnetic_field = np.dot(R, magnetic_field)

# # 取出投影後的加速度數據和磁場數據在車子的xyz座標系統中的值
# ax_car, ay_car, az_car = projected_acceleration.flatten()
# # mx_car, my_car, mz_car = projected_magnetic_field.flatten()

# print('ax_car =', ax_car, 'ay_car =', ay_car, 'az_car =', az_car)
# # print('mx_car =', mx_car, 'my_car =', my_car, 'mz_car =', mz_car)



# import numpy as np

# # Euler angles (in radians) calculated from accelerometer sensor data
# yaw = 0.1008
# pitch = 0.0246
# roll = -0.0907

# # Define the transformation matrix
# Rx = np.array([[1, 0, 0],
#                [0, np.cos(roll), -np.sin(roll)],
#                [0, np.sin(roll), np.cos(roll)]])
# Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                [0, 1, 0],
#                [-np.sin(pitch), 0, np.cos(pitch)]])
# Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                [np.sin(yaw), np.cos(yaw), 0],
#                [0, 0, 1]])
# R = Rz.dot(Ry).dot(Rx) # Combine the rotation matrices in the order of RzRyRx

# # Accelerometer signal data values in the device coordinate system
# x = 0.08349609375
# y = -0.9655303955
# z = -0.2501373291

# # Create a 3x1 column vector of the signal data values
# V_device = np.array([[x], [y], [z]])

# # Multiply the transformation matrix with the signal data values
# V_local_level = R.dot(V_device)

# # Print the transformed signal data values in the local-level coordinate system
# print(V_local_level.shape())

# #  np.array([0.08349609375, -0.9655303955, -0.2501373291]).reshape((3, 1))