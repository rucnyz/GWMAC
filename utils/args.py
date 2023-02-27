# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 20:27
# @Author  : nieyuzhou
# @File    : args.py
# @Software: PyCharm
import argparse


def compute_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 350)
    parser.add_argument('--batch_size', type = int, default = 400)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--seed', type = int, default = 123)
    parser.add_argument('--lr', type = float, default = 2e-4)
    parser.add_argument('--lr_weight', type = float, default = 1)
    parser.add_argument('--loss_fn', type = str, default = "KL", choices = ["KL", "L2"])
    parser.add_argument('--iters', type = int, default = 1)
    parser.add_argument('--gamma', type = float, default = 1000)
    parser.add_argument('--alpha', type = float, default = 0.9)

    # weights
    parser.add_argument('--Wlambda', type = float, default = 0)

    parser.add_argument('--data', type = str, default = "hw",
                        choices = ["hw", "cal7", "cal20", "rt", "orl", "movie", "pro", "3s"])
    parser.add_argument('--device', type = str, default = "cuda",
                        choices = ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument('--ot_method', type = str, default = "ppa", choices = ["ppa", "b-admm"])
    parser.add_argument('--save', type = int, default = 0)

    # test
    parser.add_argument('--test_rate', type = float, default = 0)
    args = parser.parse_args()
    return args
