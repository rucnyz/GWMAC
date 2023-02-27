# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 18:34
# @Author  : nieyuzhou
# @File    : ot_encoder.py
# @Software: PyCharm
import torch
from torch import nn

def my_rbf_kernel(data, gamma = 20):
    gamma = gamma / data.shape[1]
    # out = -torch.cdist(data, data, p = 2)
    K = torch.cdist(data, data, p = 2) ** 2 * (-gamma)
    out = torch.exp(K)
    return out

def cal_polynomial_kernel(
            x1: torch.Tensor,
            x2: torch.Tensor,
            power,
            offset:float) -> torch.Tensor:
    p_dot = torch.matmul(x1, x2.transpose(-2, -1))
    p_kernel = (p_dot + offset).pow(power)
    return p_kernel


class OtEncoder(nn.Module):
    def __init__(self, args):
        super(OtEncoder, self).__init__()
        self.views = args.views
        self.gamma = args.gamma
        self.device = args.device
        self.encoders = nn.ModuleList([nn.Sequential(
            nn.Linear(args.feature_dims[i], args.feature_dims[i] * 2),
            nn.ReLU(),
            nn.Linear(args.feature_dims[i] * 2, args.feature_dims[i]),
            nn.ReLU(),
            nn.Linear(args.feature_dims[i], args.feature_dims[i]),
        ) for i in range(self.views)])

    def forward(self, x):
        output = {}
        for i in range(self.views):
            output[i] = my_rbf_kernel(self.encoders[i](x[i]), gamma = self.gamma)
        return output
