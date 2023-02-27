# -*- coding: utf-8 -*-
# @Time    : 2022/3/23 12:48
# @Author  : nieyuzhou
# @File    : my_ot.py
# @Software: PyCharm
import numpy as np
import torch
from ot.backend import get_backend
from ot.gromov import gwloss, gwggrad
from ot.lp import emd_c


def solve_1d_linesearch_quad(a, b, c):
    f0 = c
    df0 = b
    f1 = a + f0 + df0

    if a > 0:  # convex
        minimum = min(1, max(0, torch.divide(-b, 2.0 * a).item()))
        return minimum
    else:  # non convex
        if f0 > f1:
            return 1
        else:
            return 0


def solve_linesearch(nx, cost, G, deltaG, C1 = None, C2 = None, reg = None, constC = None, M = None,
                     alpha_min = None, alpha_max = None):
    dot = nx.dot(nx.dot(C1, deltaG), C2)
    a = -2 * reg * nx.sum(dot * deltaG)
    b = nx.sum((M + reg * constC) * deltaG) - 2 * reg * (nx.sum(dot * G) + nx.sum(nx.dot(nx.dot(C1, G), C2) * deltaG))
    c = cost(G)

    alpha = solve_1d_linesearch_quad(a, b, c)
    if alpha_min is not None or alpha_max is not None:
        alpha = np.clip(alpha, alpha_min, alpha_max)
    fc = None
    f_val = cost(G + alpha * deltaG)

    return alpha, fc, f_val


def init_matrix(C1, C2, p, q, nx, loss_fun = 'square_loss'):
    if loss_fun == 'square_loss':
        def f1(a):
            return a ** 2

        def f2(b):
            return b ** 2

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    else:  # 'kl_loss'
        def f1(a):
            return a * nx.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-15)
    constC1 = nx.dot(
        nx.dot(f1(C1), nx.reshape(p, (-1, 1))),
        nx.ones((1, len(q)), type_as = q)
    )
    constC2 = nx.dot(
        nx.ones((len(p), 1), type_as = p),
        nx.dot(nx.reshape(q, (1, -1)), f2(C2).T)
    )
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2


def emd(nx, a, b, M, numItermax = 100000, numThreads = 1):
    type_as = a
    # convert to numpy
    M, a, b = nx.to_numpy(M, a, b)
    # ensure float64
    a = np.asarray(a, dtype = np.float64)
    b = np.asarray(b, dtype = np.float64)
    M = np.asarray(M, dtype = np.float64, order = 'C')
    # if empty array given then use uniform distributions
    if len(a) == 0:
        a = np.ones((M.shape[0],), dtype = np.float64) / M.shape[0]
    if len(b) == 0:
        b = np.ones((M.shape[1],), dtype = np.float64) / M.shape[1]
    # ensure that same mass
    b = b * a.sum() / b.sum()
    G, cost, u, v, result_code = emd_c(a, b, M, numItermax, numThreads)
    return nx.from_numpy(G, type_as = type_as)


def cg(nx, a, b, M, reg, f, df, G0 = None, numItermax = 200, numItermaxEmd = 100000,
       stopThr = 1e-9, stopThr2 = 1e-9, **kwargs):
    loop = 1

    if G0 is None:
        G = nx.outer(a, b)
    else:
        G = G0

    def cost(_G):
        return nx.sum(M * _G) + reg * f(_G)

    f_val = cost(G)

    it = 0
    # return optimal_transport(C1, C2, p, q, device)
    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg * df(G)
        # set M positive
        Mi += nx.min(Mi)
        # Gc = ot.emd(a, b, Mi, numItermax = numItermaxEmd, numThreads = 1)
        Gc = emd(nx, a, b, Mi, numItermax = numItermaxEmd, numThreads = 1)
        deltaG = Gc - G

        # line search
        alpha, fc, f_val = solve_linesearch(
            nx, cost, G, deltaG, reg = reg, M = M, alpha_min = 0., alpha_max = 1., **kwargs
        )

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)
        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

    return G


def gromov_wasserstein(C1, C2, p, q, loss_fun = 'square_loss', **kwargs):
    nx = get_backend(p, q, C1, C2)

    p = p.squeeze()
    q = q.squeeze()
    constC, hC1, hC2 = init_matrix(C1, C2, p, q, nx, loss_fun)

    G0 = p[:, None] * q[None, :]

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

    return cg(nx, p, q, 0, 0.5, f, df, G0, C1 = C1, C2 = C2, constC = constC, **kwargs)


def cost_mat(cost_s: torch.Tensor,
             cost_t: torch.Tensor,
             p_s: torch.Tensor,
             p_t: torch.Tensor,
             tran: torch.Tensor,
             emb_s: torch.Tensor = None,
             emb_t: torch.Tensor = None) -> torch.Tensor:
    """
    Implement cost_mat for Gromov-Wasserstein discrepancy (GWD)

    Suppose the loss function in GWD is |a-b|^2 = a^2 - 2ab + b^2. We have:

    :math:`f_1(a) = a^2,`
    :math:`f_2(b) = b^2,`
    :math:`h_1(a) = a,`
    :math:`h_2(b) = 2b`

    When the loss function can be represented in the following format: loss(a, b) = f1(a) + f2(b) - h1(a)h2(b), we have
    cost_st = f1(cost_s)*mu_s*1_nt^T + 1_ns*mu_t^T*f2(cost_t)^T
    cost = cost_st - h1(cost_s)*trans*h2(cost_t)^T
    Args:
        cost_s: (ns, ns) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        cost_t: (nt, nt) matrix (torch tensor), representing distance matrix of samples or adjacency matrix of a graph
        p_s: (ns, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        p_t: (nt, 1) vector (torch tensor), representing the empirical distribution of samples or nodes
        tran: (ns, nt) matrix (torch tensor), representing the optimal transport from source to target domain.
        emb_s: (ns, d) matrix
        emb_t: (nt, d) matrix
    Returns:
        cost: (ns, nt) matrix (torch tensor), representing the cost matrix conditioned on current optimal transport
    """
    cost = ((cost_s ** 2) @ p_s).repeat(1, tran.size(1)) + \
           (torch.t(p_t) @ torch.t(cost_t ** 2)).repeat(tran.size(0), 1) - 2 * cost_s @ tran @ torch.t(cost_t)

    if emb_s is not None and emb_t is not None:
        tmp1 = emb_s @ torch.t(emb_t)
        tmp2 = torch.sqrt((emb_s ** 2) @ torch.ones(emb_s.size(1), 1))
        tmp3 = torch.sqrt((emb_t ** 2) @ torch.ones(emb_t.size(1), 1))
        cost += 0.5 * (1 - tmp1 / (tmp2 @ torch.t(tmp3)))
        # tmp1 = 2 * emb_s @ torch.t(emb_t)
        # tmp2 = ((emb_s ** 2) @ torch.ones(emb_s.size(1), 1)).repeat(1, tran.size(1))
        # tmp3 = ((emb_t ** 2) @ torch.ones(emb_t.size(1), 1)).repeat(1, tran.size(0))
        # tmp = 0.1 * (tmp2 + torch.t(tmp3) - tmp1) / (emb_s.size(1) ** 2)
        # cost += tmp
    return cost


def optimal_transport(cost_s: torch.Tensor, cost_t: torch.Tensor, p_s: torch.Tensor, p_t: torch.Tensor,
                      device, gamma: float = 0.05, ot_method: str = 'b-admm', num_layer: int = 6,
                      emb_s: torch.Tensor = None, emb_t: torch.Tensor = None):
    tran = p_s @ torch.t(p_t)
    if ot_method == 'ppa':
        dual = torch.ones_like(p_s) / p_s.size(0)
        for m in range(num_layer):
            kernel = torch.exp(-cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t) / gamma) * tran
            b = p_t / (torch.t(kernel) @ dual)
            for i in range(20):
                dual = p_s / (kernel @ b)
                b = p_t / (torch.t(kernel) @ dual)
            tran = (dual @ torch.t(b)) * kernel
    elif ot_method == 'b-admm':
        all1_s = torch.ones_like(p_s)
        all1_t = torch.ones_like(p_t)
        dual = torch.zeros(p_s.size(0), p_t.size(0), device = device)
        for m in range(num_layer):
            kernel_a = torch.exp((dual + 2 * torch.t(cost_s) @ tran @ cost_t) / gamma) * tran
            b = p_t / (torch.t(kernel_a) @ all1_s)
            aux = (all1_s @ torch.t(b)) * kernel_a

            dual = dual + gamma * (tran - aux)

            kernel_t = torch.exp(
                -torch.div((cost_mat(cost_s, cost_t, p_s, p_t, aux, emb_s, emb_t) + dual), gamma)) * aux
            a = p_s / (kernel_t @ all1_t)
            tran = (a @ torch.t(all1_t)) * kernel_t
    # d_gw = (cost_mat(cost_s, cost_t, p_s, p_t, tran, emb_s, emb_t) * tran).sum()
    return tran
