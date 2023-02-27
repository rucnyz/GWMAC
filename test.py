# -*- coding: utf-8 -*-
# @Time    : 2022/5/9 11:26
# @Author  : nieyuzhou
# @File    : test.py
# @Software: PyCharm

import torch
from sklearn.metrics import normalized_mutual_info_score
from torch.utils.data import DataLoader

from dataset.multi_view_dataset import Multiview_Dataset
from utils.metric import cluster_accuracy
from utils.my_ot import gromov_wasserstein

if __name__ == '__main__':
    data_name = "hw"
    test_rate = 0
    alpha = 1.0
    Wlambda = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = torch.load(
        "./data/model/" + data_name + "_" + str(test_rate) + "_" + str(alpha) + "_" + str(Wlambda) + "_encoder.pt",
        map_location = device)
    weights, args = torch.load(
        "./data/model/" + data_name + "_" + str(test_rate) + "_" + str(alpha) + "_" + str(Wlambda) + "_weights_args.pt",
        map_location = device)
    C, E, p, p_e = torch.load(
        "./data/model/" + data_name + "_" + str(test_rate) + "_" + str(alpha) + "_" + str(Wlambda) + "_CEP.pt",
        map_location = device)
    torch.manual_seed(args.seed)
    if test_rate == 0:
        test_dataset = Multiview_Dataset(args, 0)
        args.test_nums = args.train_nums
        data_loader = DataLoader(test_dataset, args.batch_size, num_workers = args.num_workers, shuffle = True,
                                 pin_memory = True)
    else:
        test_dataset = Multiview_Dataset(args, 1)
        data_loader = DataLoader(test_dataset, args.batch_size, num_workers = args.num_workers, shuffle = False,
                                 pin_memory = True)
    accuracy = 0
    nmi = 0
    with torch.no_grad():
        for data, labels in data_loader:
            for i in range(args.views):
                data[i] = data[i].to(device)
                labels[i] = labels[i].to(device)
            Cs = encoder(data)
            ps = [torch.div(torch.ones([data[s].shape[0], 1], device = device), data[s].shape[0]) for s in
                  range(args.views)]
            T2 = gromov_wasserstein(C.detach(), E.detach(), p.detach(), p_e.detach())
            for i in range(args.views):
                T1 = gromov_wasserstein(Cs[i].detach(), C.detach(), ps[i].detach(), p.detach())
                predicted = torch.argmax(T1 @ T2, dim = 1)
                accuracy += (weights[i] * cluster_accuracy(predicted.cpu().numpy(),
                                                           labels[i].cpu().numpy()) * labels[i].shape[0]).item()
                nmi += (weights[i] * normalized_mutual_info_score(
                    labels[i].cpu().numpy(), predicted.cpu().numpy()) * labels[i].shape[0]).item()
        accuracy /= args.test_nums
        nmi /= args.test_nums
        print("accuracy:{:.4f}".format(accuracy))
        print("nmi:{:.4f}".format(nmi))
