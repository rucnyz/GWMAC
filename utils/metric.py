# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 11:02
# @Author  : nieyuzhou
# @File    : metric.py
# @Software: PyCharm
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler


def cluster_accuracy(y_predict, y_true):
    """
    Hungarian algorithm
    """
    D = max(y_predict.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype = np.int64)
    for i in range(y_predict.size):
        cost[y_predict[i], y_true[i]] += 1
    ind = linear_sum_assignment(np.max(cost) - cost)
    ind = np.array(ind).T
    return sum([cost[i, j] for i, j in ind]) * 1.0 / y_predict.size


def normalize(x, name):
    if name == "rt":
        scaler = MaxAbsScaler()
    else:
        scaler = MinMaxScaler([0, 1])
    norm_x = scaler.fit_transform(x)
    return norm_x
