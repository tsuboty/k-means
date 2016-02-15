# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import random as rd
from kmeans_funcs import *

##########################################
# main routine ##############################
##########################################

#########初期値の設定###############
#dataFrame化
data = readData("./data/data.csv")
# data = np.array(data)
# 初期値の設定
centers = setInitialPoint(3,data)
# 所属ベクトル初期設定
label = pd.Series(np.ones(len(data)))

#################################
#################################


def k_means_one(centers,data,label):
	#データ距離の計算
	dist = calcMinDistance(centers,data)
	# データフレーム化
	dist = pd.DataFrame(dist).T
	# 列比較で最小値をもとめる
	min_column = dist.min(axis=1)
	#所属を求める。
	label = calcBelong(dist,min_column,label)
	#中心の再計算
	new_center = calcCenter(data,dist,label)

	return (new_center,label)

centers,label = k_means_one(centers,data,label)

# 50回繰り返す
for i in range(50):
	centers,label = k_means_one(centers,data,label)

# クラスター化

clusters = [[0],[0],[0]]
for i in range(len(label)):
	index = int(label[i])
	clusters[index].append(data.ix[i:i])
	plt.plot

c_set = ["r","g","b"]
for i in range(len(clusters)):
	for point in clusters[i][1:]:
		plt.scatter(point.x,point.y,c=c_set[i],s=80)

plt.scatter(centers[:,0],centers[:,1],c="b",s=120)
plt.show()