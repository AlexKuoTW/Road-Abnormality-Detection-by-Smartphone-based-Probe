import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

class Clustering:
    def __init__(self, data_path, map_path, points):
        self.data_path = data_path
        self.map_path = map_path
        self.points = points

    def hierarchical_clustering(self, data, n_clusters):
        # data = pd.read_csv(self.data_path, names=['LATITUDE', 'LONGITUDE'], sep=',')
        print(data.iloc[:,0:2])
        Z = linkage(data.iloc[:,0:2], method='ward')
        labels = fcluster(Z, t=n_clusters, criterion='maxclust')
        return labels

    def create_image(self, color, n_clusters, min_width=2, max_width=18):
        labels = self.hierarchical_clustering(self.data_path, n_clusters)
        
        data = self.data_path
        data['Cluster'] = labels
        # print(data['Cluster'].value_counts())
        cluster_sizes = data['Cluster'].value_counts().to_dict()
        print(cluster_sizes)
        self.result_image = Image.open(self.map_path, 'r')
        draw = ImageDraw.Draw(self.result_image)
        colors = [(np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)) for _ in range(n_clusters)]
        for _, row in data.iterrows():
            lat, lon, cluster_id = row['LATITUDE'], row['LONGITUDE'], row['Cluster']
            x1, y1 = self.scale_to_img((lat, lon), (self.result_image.size[0], self.result_image.size[1]))

            # Compute the width based on the cluster size
            cluster_size = cluster_sizes[cluster_id]
            width = np.interp(cluster_size, (data['Cluster'].value_counts().min(), data['Cluster'].value_counts().max()), (min_width, max_width))

            draw.ellipse([(x1 - width / 2, y1 - width / 2), (x1 + width / 2, y1 + width / 2)], fill=colors[int(cluster_id) - 1], width=int(width))


        """
        這是更改顏色
        """
        # self.result_image = Image.open(self.map_path, 'r')
        # draw = ImageDraw.Draw(self.result_image)
        for _, row in data.iterrows():
            lat, lon, cluster_id = row['LATITUDE'], row['LONGITUDE'], row['Cluster']
            x1, y1 = self.scale_to_img((lat, lon), (self.result_image.size[0], self.result_image.size[1]))
            # draw.ellipse([(x1 - 2, y1 - 2), (x1 + 2, y1 + 2)], fill=colors[cluster_id - 1])
            draw.ellipse([(x1 - 2, y1 - 2), (x1 + 2, y1 + 2)], fill=colors[int(cluster_id) - 1])

        """
        這是更改橢圓大小
        """
        # for cluster_id, size in cluster_sizes.items():
        #     cluster_data = data[data['Cluster'] == cluster_id]
        #     gps_data = tuple(zip(cluster_data['LATITUDE'].values, cluster_data['LONGITUDE'].values))
        #     width = np.interp(size, (data['Cluster'].value_counts().min(), data['Cluster'].value_counts().max()), (min_width, max_width))
        #     for d in gps_data:
        #         x1, y1 = self.scale_to_img(d, (self.result_image.size[0], self.result_image.size[1]))
        #         draw.ellipse([(x1 - width / 2, y1 - width / 2), (x1 + width / 2, y1 + width / 2)], fill=color, width=int(width))

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