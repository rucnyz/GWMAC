# -*- coding: utf-8 -*-
# @Time    : 2022/2/1 11:14
# @Author  : nieyuzhou
# @File    : main.py
# @Software: PyCharm
import time

import ot
from sklearn.metrics import normalized_mutual_info_score
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from dataset.multi_view_dataset import *
from models.ot_encoder import OtEncoder
from utils.args import compute_args
from utils.metric import cluster_accuracy
from utils.my_ot import gromov_wasserstein


def evaluate(data_loader, encoder, C, E, p, p_e, weights, args):
    accuracy = 0
    nmi = 0
    for data, labels in data_loader:
        for i in range(args.views):
            data[i] = data[i].to(args.device)
            labels[i] = labels[i].to(args.device)
        Cs = encoder(data)
        ps = [torch.div(torch.ones([data[s].shape[0], 1], device = args.device), data[s].shape[0]) for s in
              range(args.views)]
        T2 = gromov_wasserstein(C.detach(), E.detach(), p.detach(), p_e.detach())
        for i in range(args.views):
            T1 = gromov_wasserstein(Cs[i].detach(), C.detach(), ps[i].detach(), p.detach())
            predicted = torch.argmax(T1 @ T2, dim = 1)
            accuracy += (weights[i] * cluster_accuracy(predicted.cpu().numpy(), labels[i].cpu().numpy()) *
                         labels[i].shape[0]).item()
            nmi += (weights[i] * normalized_mutual_info_score(labels[i].cpu().numpy(), predicted.cpu().numpy()) *
                    labels[i].shape[0]).item()
    return accuracy, nmi


if __name__ == '__main__':
    # arguments
    args = compute_args()
    print(args)
    torch.manual_seed(args.seed)
    # train dataset
    test_loader = None
    train_dataset = Multiview_Dataset(args, 0)
    train_loader = DataLoader(train_dataset, args.batch_size, num_workers = args.num_workers, shuffle = True,
                              pin_memory = True)
    if args.test_rate != 0:
        test_dataset = Multiview_Dataset(args, 1)
        test_loader = DataLoader(test_dataset, args.batch_size, num_workers = args.num_workers, shuffle = False,
                                 pin_memory = True)
    # C = torch.randn([args.num_class, args.num_class], device = args.device)
    weights = torch.ones(args.views, requires_grad = True, device = args.device)
    weights.data = ot.utils.proj_simplex(weights)
    encoder = OtEncoder(args).to(args.device)
    optim = torch.optim.Adam(
        [{"params": encoder.parameters(), "lr": args.lr}, {"params": weights, "lr": args.lr_weight}])
    # model
    C = torch.eye(args.num_class, device = args.device)
    p = torch.div(torch.ones([args.num_class, 1], device = args.device), args.num_class)
    E = torch.eye(args.num_class, device = args.device)
    p_e = torch.div(torch.ones([args.num_class, 1], device = args.device), args.num_class)
    # training
    loss_items = []
    nmi_items = []
    acc_items = []
    noise_items = [weights[-1].item()]
    best_train_accuracy = best_train_nmi = 0
    best_loss = 100
    for epoch in range(args.epochs):
        start_time = time.time()
        # using minibatch
        loss_sum = 0
        for data, labels in train_loader:
            # compute weights
            weights_views = weights.softmax(dim = -1)
            for i in range(args.views):
                data[i] = data[i].to(args.device)
                labels[i] = labels[i].to(args.device)
            ps = [torch.div(torch.ones([data[s].shape[0], 1], device = args.device), data[s].shape[0]) for s in
                  range(args.views)]
            # ----------------compute GW distance----------------
            # transform origin data into similarity matrix
            Cs = encoder(data)
            # compute optimal transport and then barycenter
            for iteration in range(args.iters):
                B = 0
                if args.loss_fn == "KL":
                    for i in range(args.views):
                        T = gromov_wasserstein(Cs[i].detach(), C.detach(), ps[i].detach(), p.detach())
                        B += weights_views[i] * T.T @ torch.log(Cs[i]) @ T
                    C = torch.exp(torch.div(B, p @ p.T))
                else:  # L2
                    for i in range(args.views):
                        T = gromov_wasserstein(Cs[i].detach(), C.detach(), ps[i].detach(), p.detach())
                        B += weights_views[i] * T.T @ Cs[i] @ T
                    C = torch.div(B, p @ p.T)

            # ----------------compute loss----------------
            weight_entr = torch.dot(weights_views, torch.log(weights_views))
            # three loss functions
            # 1. GW
            # loss = ot.gromov_wasserstein2(C, E, p.squeeze(), p_e.squeeze())  + weight_entr * args.Wlambda
            # 2. MSE
            loss = mse_loss(C, E) + weight_entr * args.Wlambda
            # 3. KL_div
            # loss = kl_div(C.softmax(dim=-1).log(), E.softmax(dim=-1), reduction = 'sum',log_target=False) + args.Wlambda * weight_entr
            loss_sum += loss.item()
            optim.zero_grad()
            loss.backward()
            optim.step()
            # weights.data = ot.utils.proj_simplex(weights)
        print("[epoch: {}]  loss: {:.4f}".format(epoch, loss_sum))
        loss_items.append(loss_sum)
        # ----------------clustring and valid----------------
        if loss_sum < 1 or best_loss > loss_sum:
            weights_views = weights.softmax(dim = -1)
            best_loss = loss_sum
            with torch.no_grad():
                accuracy, nmi = evaluate(train_loader, encoder, C, E, p, p_e, weights_views, args)
                accuracy /= args.train_nums
                nmi /= args.train_nums
                nmi_items.append(nmi)
                noise_items.append(weights_views[-1].item())
                acc_items.append(accuracy)
                print("time:{:.4f}s".format(time.time() - start_time))
                print("accuracy:{:.4f}".format(accuracy))
                print("nmi:{:.4f}".format(nmi))
                print("weights:" + str(weights_views.data))
                if best_train_accuracy < accuracy:
                    best_train_epoch = epoch
                    best_train_accuracy = accuracy
                if best_train_nmi < nmi:
                    if args.save == 1:
                        torch.save(encoder, "./data/model/" + args.data + "_" + str(args.test_rate) + "_" + str(
                            args.alpha) + "_" + str(args.Wlambda) + "_encoder.pt")
                        torch.save([weights_views, args],
                                   "./data/model/" + args.data + "_" + str(args.test_rate) + "_" + str(
                                       args.alpha) + "_" + str(args.Wlambda) + "_weights_args.pt")
                        torch.save([C, E, p, p_e], "./data/model/" + args.data + "_" + str(args.test_rate) + "_" + str(
                            args.alpha) + "_" + str(args.Wlambda) + "_CEP.pt")
                    best_parameters = encoder.state_dict()
                    best_weights_views = weights_views
                    best_train_nmi = nmi
                    best_train_epoch = epoch
    print("--------------------------------")
    print("alphas:" + str(args.alphas))
    print("epoch:" + str(best_train_epoch))
    print("best accuracy:{:.4f}".format(best_train_accuracy))
    print("best nmi:{:.4f}".format(best_train_nmi))
    # teseting
    if test_loader is not None:
        with torch.no_grad():
            encoder.load_state_dict(best_parameters)
            accuracy, nmi = evaluate(test_loader, encoder, C, E, p, p_e, best_weights_views, args)
            accuracy /= args.test_nums
            nmi /= args.test_nums
            print("--------------------------------")
            print("testing")
            print("accuracy:{:.4f}".format(accuracy))
            print("nmi:{:.4f}".format(nmi))
