import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from io import StringIO
from PIL import Image, ImageDraw
from gpsvis import reorient_acceleration_data
from gpsvis import GPSVis
from label_zscore import Labels
from clustering import Clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

# from label_zdiff import Labels

warnings.filterwarnings("ignore")
csv = 3 # TODO: Change the csv number here
print("Start processing round {}.csv".format(csv))

"""
Read the data and save the data before processing
fill the 0 in rf of empty data
"""
round = pd.read_csv("./NYCU_GPS/lab1/{}.csv".format(csv))
r = round.ffill()
r.fillna(0, inplace=True)
# r.to_csv("./NYCU_GPS/lab1/{}_before_align.csv".format(csv), index=False)
# r.plot(x='Timestamp', y=['Gyroscope_x','Gyroscope_y','Gyroscope_z'])
# r.plot(x='Timestamp', y=['Accelerometer_x','Accelerometer_y','Accelerometer_z'])
# r.plot(x='Timestamp', y=['Accelerometer_z'],xlim=(1677898055220,1677898265775),ylim=(-1,-0.5)) #1
# r.plot(x='Timestamp', y=['Accelerometer_z'],xlim=(1677898362819,1677898567342),ylim=(-1.25,0.5)) #2

"""
Using the gravity to align the Z-axis of the acceleration data with that of the vehicle.
Save the reoriented data back to r dataframe
"""
df_numpy = r.to_numpy()
data = df_numpy[:, 4:7]
reoriented_data = reorient_acceleration_data(data)
# print(reoriented_data)
r['Accelerometer_vehicle_x'] = reoriented_data[:, 0]
r['Accelerometer_vehicle_y'] = reoriented_data[:, 1]
r['Accelerometer_vehicle_z'] = reoriented_data[:, 2]
# r.to_csv("./NYCU_GPS/lab1/2_after_align.csv", index=False)



######### Task ######### 
"""
ms to s
Dind the indices of outliers using z-score method and
replace the outliers with the mean value
"""
r['Time'] = r['Timestamp']/1000 
z_scores = np.abs((r['Accelerometer_vehicle_z'] - np.mean(r['Accelerometer_vehicle_z'])) / np.std(r['Accelerometer_vehicle_z']))
outlier_indices = np.where(z_scores > 5)[0]
# print(outlier_indices)
# print(r.iloc[outlier_indices[0:9]])
## drop the outliers
# cleaned_data = r['Accelerometer_y'].drop(outlier_indices)
cleaned_data = r['Accelerometer_vehicle_z'].copy()
cleaned_data.loc[outlier_indices] = np.nan
cleaned_data = cleaned_data.fillna(cleaned_data.mean())


"""
Take the average of every 20 sec for all Accelerometer_vehicle_z
as a vertical upward vector and called by g0
Then fill the latest g0 into last 0.xx sec
"""
j = 0
average_list = []
r['Accelerometer_Average'] = np.zeros(len(r['Accelerometer_vehicle_z']))
for i in range(0, len(cleaned_data), 1):
    # Every 20 sec reset the g0
    if (r['Time'][i] - r['Time'][j]) >= 20:
        g0 = cleaned_data[j:i].mean()
        # Fill the g0 into r['Accelerometer_Average'][i:j]
        r['Accelerometer_Average'][j:i] = g0
        average_list.append(g0)
        j = i
r['Accelerometer_Average'][j:] = g0


"""
save the new csv file
plot the r['Accelerometer_Average'] and r['Accelerometer_vehicle_z']
"""
# r.to_csv("./NYCU_GPS/lab1/2_average.csv", index=True)
# r.plot(x = 'Time', y = ['Accelerometer_vehicle_z', 'Accelerometer_Average'])
# plt.show()


"""
Use the z-score thresold to find the abnormal value in r['Accelerometer_vehicle_z']
Each 0.5 sec detect the abnormal value then sbve the result into result_df
"""
label_df = Labels()
col_name = 'Accelerometer_vehicle_z'
thold = r['Accelerometer_Average']
filter_df = label_df.cusum_filter(r,r[col_name].mean(),col_name)
result_df = r.loc[filter_df['event'], :]
# abnormal value of r merged with result_df
result_df = pd.merge(result_df, filter_df, how='inner', on=['Time'])
result_df = result_df.sort_values('label')
# drop the result_df['value'] == 0
result_df = result_df[result_df['value'] != 0]
# result_df.to_csv("./NYCU_GPS/lab1/result_df.csv", index=False)

"""
Create data.txt and record the result_df['Latitude','Longitude'] into it
"""
Latitude = pd.DataFrame({'Latitude':result_df['Latitude'],
                        'Longitude':result_df['Longitude'],
                        'labels':result_df['label']})
Latitude.to_csv("./NYCU_GPS/lab1/data{}.txt".format(csv), index=False, header=False, sep=',')


"""
GPSVis implementation
"""

vis = GPSVis(data_path='./NYCU_GPS/lab1/data{}.txt'.format(csv),
             map_path='./NYCU_GPS/lab1/map.jpg',  # Path to map downloaded from the OSM.
             points=(24.789664521572398, 120.99518219793899, 24.787270842564535, 120.99897215920797)) # Two coordinates of the map (upper left, lower right)

# use the for loop to read the labels in data.txt [24.7886260815816,120.99526551742817,2]
# eg,. like 2 is labels 
# then use the GPSVis to plot the points
vis.create_image(color=(255, 0, 0), width = 3)  # Set the color and the width of the GNSS tracks.
vis.plot_map(output='save', save_as='./NYCU_GPS/lab1/resultMap_{}.png'.format(csv))



"""
Merge the 3 of txt file into one txt file
"""
# def load_and_merge_data(files, output_file):
#     data_frames = []
#     for file in files:
#         df = pd.read_csv(file, names=['LATITUDE', 'LONGITUDE'], sep=',')
#         data_frames.append(df)
#     merged_data = pd.concat(data_frames, ignore_index=True)
#     # Save the merged data to a .txt file
#     merged_data.to_csv(output_file, index=False, header=False, sep=',')
#     return merged_data
def merge_txt_files(input_files, output_file, path='./NYCU_GPS/lab1/'):
    with open(path+output_file, 'w') as outfile:
        for file in input_files:
            with open(file, 'r') as infile:
                for line in infile:
                    outfile.write(line)

data_files = ['./NYCU_GPS/lab1/data1.txt', './NYCU_GPS/lab1/data2.txt', './NYCU_GPS/lab1/data3.txt']
output_file = 'merged_data.txt'
data = merge_txt_files(data_files, output_file)
# print(data)


"""
GPSVis implementation again
"""
# vis = GPSVis(data_path='./NYCU_GPS/lab1/{}'.format(output_file),
#              map_path='./NYCU_GPS/lab1/map.jpg',  # Path to map downloaded from the OSM.
#              points=(24.789664521572398, 120.99518219793899, 24.787270842564535, 120.99897215920797)) # Two coordinates of the map (upper left, lower right)

# vis.create_image(color=(255, 0, 0), width=3)  # Set the color and the width of the GNSS tracks.
# vis.plot_map(output='save', save_as='./NYCU_GPS/lab1/resultMap_{}.png'.format(output_file))


"""
Write back the merge txt file into pd.DataFrame type
"""
# Transfer data to pd.DataFrame type
with open('./NYCU_GPS/lab1/'+output_file, 'r') as file:
    lines = file.readlines()
cleaned_lines = [line.strip() for line in lines]
txt_content = "\n".join(cleaned_lines)
txt_file = StringIO(txt_content)
cluster_df = pd.read_csv(txt_file, names=['LATITUDE', 'LONGITUDE', 'label'], sep=',')
# cluster_df.to_csv("./NYCU_GPS/lab1/cluster_df.csv", index=False)


"""
clustering using the Hierarchical Method:
First we need to calculate the distance between each point
References the clustering.py 
"""
# ================Try=================
# def get_color_by_size(size, max_size):
#     intensity = int(255 * (1 - (size / max_size)))
#     return (intensity, 255, intensity)

# def get_width_by_size(size, max_size, min_width=5, max_width=20):
#     width = ((size / max_size) * (max_width - min_width)) + min_width
#     return int(width)

# def plot_different_sizes(vis, clusters):
#     max_cluster_size = max([len(cluster) for cluster in clusters])

#     for cluster in clusters:
#         centroid = np.mean(cluster, axis=0)
#         width = get_width_by_size(len(cluster), max_cluster_size)
#         vis.plot_clustered_points(centroids=[centroid], color=(0, 255, 0), width=width)
# ================Try=================

n_clusters = 15 # TODO: 聚類的個數
color = (255, 0, 0)  # Color
cluster = Clustering(cluster_df, map_path='./NYCU_GPS/lab1/map.jpg',
                     points=(24.789664521572398, 120.99518219793899, 24.787270842564535, 120.99897215920797))

cluster.create_image(color, n_clusters)
# cluster.result_image.show()
cluster.plot_map(output='save', save_as='./NYCU_GPS/lab1/Cluster_result.png')
print("Finish processing round {}.csv".format(csv))