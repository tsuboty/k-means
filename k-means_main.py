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
data = readData("./data.csv")
# data = np.array(data)
# 初期値の設定
centers = setInitialPoint(3,data)
# 所属ベクトル初期設定
label = pd.Series(np.ones(len(data)))

##################
##################


def k_means_one(centers,data):
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


plt.scatter(data["x"],data["y"],c="r")
plt.scatter(new_center[:,0],new_center[:,1],c="b",s=120)
plt.show()