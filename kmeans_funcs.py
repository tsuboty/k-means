# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import random as rd

#ファイルの読み込み
def readData(url):
	return pd.read_csv(url)

# 初期中心点の設定
def setInitialPoint(cluster_num,data):
	center = []
	for i in range(cluster_num):
		randomint = rd.randrange(0,len(data))
		x = data["x"][randomint]
		y = data["y"][randomint]
		ar = [x,y]
		center.append(ar)
	return center

#ユークリッド距離の計算
def calcMinDistance(centers,data):
	dist = []
	i = 0
	for p in centers:
		d =  (data["x"] - p[0])**2 + (data["y"] - p[1])**2
		dist.append(d)
	return dist

#所属ラベルの計算
def calcBelong(dist, min_column,label,CLUSTER_NUM):


	for i in range(len(dist)):
		for j in range(CLUSTER_NUM):
			if (dist.iat[i,j] == min_column[i]):
				label[i] = j
	return label


#重心を再計算
def calcCenter(data,dist,label):
	new_center = np.array([[0.0,0.0]] * (len(dist.ix[0,:])))
	num_points = [0] * len(dist.ix[0,:])

	for i in range(len(data)):
		new_center[int(label.at[i]),0] += data.ix[i:i].x
		new_center[int(label.at[i]),1] += data.ix[i:i].y
		num_points[int(label.at[i])] += 1

	for i in range(len(new_center)):
		new_center[i,0] = float(new_center[i,0]) / num_points[i]
		new_center[i,1] = new_center[i,1] * 1.0 / num_points[i]


	return new_center

