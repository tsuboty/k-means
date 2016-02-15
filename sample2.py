#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans, kmeans2, whiten

def main():
    # ２種類の２次元正規分布に基づきサンプリングして観測データを生成
    mean1 = [0,0]; cov1 = [[4,0],[0,100]]; N1 = 1000
    X1 = np.random.multivariate_normal(mean1,cov1,N1)
    mean2 = [10,-10]; cov2 = [[1,20],[20,50]]; N2 = 1000
    X2 = np.random.multivariate_normal(mean2,cov2,N2)
    X = np.concatenate((X1,X2))

    # 描画
    x,y = X.T
    plt.plot(x,y,'k.'); plt.axis('equal'); plt.show()

    # kmeans2でクラスタリング
    whitened = whiten(X) # 正規化（各軸の分散を一致させる）
    centroid, label = kmeans2(whitened, k=2) # kmeans2
    C1 = []; C2 = [] # クラスタ保存用
    for i in range(len(X)):
        if label[i] == 0:
            C1 += [whitened[i]]
        elif label[i] == 1:
            C2 += [whitened[i]]

    # 描画
    x,y = zip(*C1)
    plt.plot(x, y, 'r.')
    x,y = zip(*C2)
    plt.plot(x, y, 'g.')
    x,y = centroid.T
    plt.plot(x, y, 'bx')
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()