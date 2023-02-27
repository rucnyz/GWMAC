# -*- coding: utf-8 -*-
# @Time    : 2022/3/26 8:55
# @Author  : nieyuzhou
# @File    : extract_data.py
# @Software: PyCharm
import scipy.io as sio
import numpy as np
import torch
from mvlearn.datasets import load_UCImultifeature

from utils.metric import normalize


def get_each_view_sample(x, label, args, a, test, idx):
    np.random.seed(args.seed + idx)
    rate = a * (1 - args.test_rate)
    index = np.random.choice(int(args.nums * rate), int(args.nums * rate), replace = False)
    rest = np.arange(int(args.nums * rate), args.nums)
    if test == 1:
        index = rest
    elif args.test_rate == 0:
        # do not use test set
        index = np.concatenate([index, rest])
    return torch.tensor(normalize(x[index], args.data), dtype = torch.float32), torch.tensor(label[index])


def get_movie_data():
    actors = np.loadtxt("./data/my_movies/M2.txt")
    keywords = np.loadtxt("./data/my_movies/M1.txt")
    movies = np.loadtxt("./data/my_movies/Mact.txt").astype("int64")
    data = [actors, keywords]
    return data, movies


def get_hw_data():
    full_data, full_labels = load_UCImultifeature(views = [0, 1, 2, 3, 4, 5])
    full_labels = full_labels.astype("int64")
    # np.random.seed(1)
    # full_data.append(np.random.randn(2000, 150))
    return full_data, full_labels


def get_mat_data(name, classes = 0):
    if classes != 0:  # cal
        dataset = sio.loadmat("./data/Caltech101_" + str(classes) + ".mat")  # read mat
    else:
        if name == "orl":
            dataset = sio.loadmat("./data/" + name + ".mat")
            dataset_y = sio.loadmat("./data/" + name + "_y.mat")
            dataset["Y"] = dataset_y["y"][0, 0]
        else:
            dataset = sio.loadmat("./data/" + name + ".mat")
    full_data = np.squeeze(dataset["X"])
    full_labels = np.squeeze(dataset["Y"])
    full_labels = (full_labels - np.min(full_labels)).astype("int64")
    data = []
    np.random.seed(1)
    index = np.random.permutation(len(full_labels))
    for i in range(len(full_data)):
        data.append(full_data[i][index].astype("float64"))
    return data, full_labels[index]


def get_pro_data():
    dataset = sio.loadmat("./data/prokaryotic.mat")
    truth = dataset["truth"].squeeze().astype("int64")
    data = [dataset["text"].astype("float64"), dataset["proteome_comp"].astype("float64"),
            dataset["gene_repert"].astype("float64")]
    return data, truth


def get_3s_data():
    dataset = sio.loadmat("./data/3-sources.mat")
    truth = dataset["truth"].squeeze().astype("int64")
    data = [dataset["bbc"].A.astype("float64"), dataset["guardian"].A.astype("float64"),
            dataset["reuters"].A.astype("float64")]
    return data, truth


def get_data(args):
    if args.data == "hw":
        full_data, full_labels = get_hw_data()
    elif args.data == "cal7":
        full_data, full_labels = get_mat_data("cal", 7)
        # full_data, full_labels = get_first_two_cal_data("cal", 7)  # test two modalities cases
    elif args.data == "cal20":
        full_data, full_labels = get_mat_data("cal", 20)
    elif args.data == "rt":
        full_data, full_labels = get_mat_data("Reuters")
    elif args.data == "movie":
        full_data, full_labels = get_movie_data()
    elif args.data == "pro":
        full_data, full_labels = get_pro_data()
    elif args.data == "3s":
        full_data, full_labels = get_3s_data()
    else:  # orl
        full_data, full_labels = get_mat_data("orl")
        if args.test_rate == 0:
            args.alpha = args.alpha - 0.5
            if args.alpha < 0:
                args.alpha = 1
    return full_data, full_labels


def get_first_two_cal_data(name, classes = 0):
    full_data, full_labels = get_mat_data(name, classes)
    index_label = np.where((full_labels == 0) | (full_labels == 1))
    data = []
    for i in range(len(full_data)):
        data.append(full_data[i][index_label])
    labels = full_labels[index_label]
    return data, labels
