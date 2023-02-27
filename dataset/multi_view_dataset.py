# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 12:15
# @Author  : nieyuzhou
# @File    : multi_view_dataset.py
# @Software: PyCharm
import numpy as np
import torch

from scipy.io import savemat
from torch.utils.data import Dataset

from utils.extract_data import get_data, get_each_view_sample
from utils.metric import normalize


def get_each_view_sample_by_loss_rate(view_data, view_labels, args, loss_rate):
    np.random.seed(args.seed)
    num_view_labels = len(view_labels)
    index = np.random.choice(num_view_labels, num_view_labels, replace = False)
    last_index = int((1 - loss_rate) * num_view_labels)
    index = index[0: last_index]
    ldata = normalize(view_data[index], args.data)
    ldata = torch.tensor(ldata, dtype = torch.float32, device = args.device)
    llabel = torch.tensor(view_labels[index], device = args.device)
    return ldata, llabel


class Multiview_Dataset(Dataset):
    def __init__(self, args, test = 0):
        super(Multiview_Dataset, self).__init__()
        # full_data:list[ndarray] full_labels:ndarray(int64)
        if args.device != "cpu" and test == 0:
            args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            print(f"Device: {args.device}")
        full_data, full_labels = get_data(args)

        # get dataset info
        args.nums = len(full_labels)
        args.views = len(full_data)
        args.num_class = np.max(full_labels) - np.min(full_labels) + 1
        # alphas = [0.8, 0.9, 0.85, 0.82, 0.84, 0.85]
        args.alphas = [args.alpha for _ in range(args.views)]
        args.feature_dims = [full_data[v].shape[1] for v in range(args.views)]

        # get train/test data
        self.views = args.views
        if test == 1:
            args.test_nums = int(args.nums * args.test_rate)
            self.nums = args.test_nums
        else:
            args.train_nums = int(args.nums * (1 - args.test_rate))
            self.nums = args.train_nums
        self.data = {}
        self.labels = {}
        # process data based on alpha
        for i, alpha in enumerate(args.alphas):
            self.data[i], self.labels[i] = get_each_view_sample(full_data[i], full_labels, args, alpha, test, i)

    def __getitem__(self, idx):
        data = {}
        labels = {}
        for i in range(self.views):
            len_view = len(self.data[i])
            index_min = min(idx, len_view - 1)
            data[i] = self.data[i][index_min]
            labels[i] = self.labels[i][index_min]
        return data, labels

    def __len__(self):
        return self.nums


def save4mat(data, label):
    x = {"1": data[0], "2": data[4]}
    y = {"1": label, "2": label}
    savemat("x.mat", x)
    savemat("y.mat", y)
