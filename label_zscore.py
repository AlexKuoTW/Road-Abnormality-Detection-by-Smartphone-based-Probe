import pandas as pd
import numpy as np

class Labels(object):
	"""Class for creating Labels"""
	def __init__(self):
		pass


	def cusum_filter(self,df,thold,col_name):
		"""
		使用 Z-DIFF 过滤器对数据进行处理，找到超出阈值的事件并返回。

		Args:
			data (pandas.DataFrame): 要处理的数据。
			column (str): 要处理的列名。
			window_size (int): 滚动窗口大小。
			threshold (float): 阈值。

		Returns:
			包含事件和值的 pandas.DataFrame。
		"""

		# 计算滚动均值和标准差
		# rolling_mean = df[col_name].rolling(window_size).mean()
		# rolling_std = df[col_name].rolling(window_size).std()
		
		# 计算 Z-DIFF 值
		zscore = np.abs((df['Accelerometer_vehicle_z'] - np.mean(df['Accelerometer_vehicle_z'])) / np.std(df['Accelerometer_vehicle_z']))
		# zscore.to_csv("./NYCU_GPS/lab1/zscore.csv", index=False)
		
		# # 查找超出阈值的事件
		# index = zscore[zscore > 5].index.tolist()
		# values = df.loc[index, col_name].tolist()

		# # 创建结果的 DataFrame
		# result_df = pd.DataFrame({
		# 	'event': index,
		# 	'value': values
		# })
		events_2 = zscore[zscore > 1.5].index.tolist() # 小坑洞
		events_3 = zscore[zscore > 3].index.tolist() # 水溝蓋
		events_4 = zscore[zscore > 5].index.tolist() # 大坑洞
		events_5 = zscore[zscore > 6].index.tolist() # 減速丘
		events = list(set(events_2 + events_3 + events_4 + events_5))
		
		# labels the events
		labels = []
		for event in events:
			if event in events_5:
				labels.append(5)
			elif event in events_4:
				labels.append(4)
			elif event in events_3:
				labels.append(3)
			elif event in events_2:
				labels.append(2)

		# 创建结果的 DataFrame
		result_df = pd.DataFrame({
			'event': events,
			'Time': df.loc[events, 'Time'].tolist(),
			'value': df.loc[events, col_name].tolist(),
			'label': labels
		})

		return result_df



