#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:29:18 2020

@author: Dumitru Mirauta
"""

import numpy as np
from NTF.core import Decomp, Factor, norm_mat


def shift_mat_down(M, i):
    Mt = np.roll(M, i, 0)
    Mt[:i, :] = 0
    return Mt


def shift_mat_up(M, i):
    Mt = np.roll(M, -i, 0)
    Mt[-i:, :] = 0
    return Mt


def cNMF_reconstruction(W, H, b=None):
    n, _ = W.shape
    nc, _, m = H.shape
    Va = np.zeros((n, m))

    for i in range(nc):
        Va += np.matmul(shift_mat_down(W, i), H[i])

    if b is not None:
        Va += np.tensordot(b, np.ones(m), 0)

    return Va


def DW(V, Wk, H):
    n, r = Wk.shape
    nc, _, m = H.shape
    DWm, DWp = np.zeros((n, r)), np.zeros((n, r))
    V_Va = np.divide(V, cNMF_reconstruction(Wk, H) + 1e-19)
    for i in range(nc):
        DWm += np.matmul(shift_mat_up(V_Va, i), H[i].T)
        DWp += np.ones((n, m)).dot(H[i].T)
    return DWm, DWp


def DH(V, W, Hk):
    n, r = W.shape
    nc, _, m = Hk.shape
    DHm, DHp = np.zeros((nc, r, m)), np.zeros((nc, r, m))
    V_Va = np.divide(V, cNMF_reconstruction(W, Hk) + 1e-19)
    for i in range(nc):
        sWT = shift_mat_down(W, i).T
        DHm[i] = np.matmul(sWT, V_Va)
        DHp[i] = np.matmul(sWT, np.ones((n, m)))
    return DHm, DHp


class oNMF_W(Factor):
    def __init__(self, decomp, **kwargs):
        kwargs["multiplicative"] = True
        super(oNMF_W, self).__init__(decomp, (decomp.n, decomp.r), **kwargs)

    def grad_neg_pos(self):
        return DW(self.decomp.V, self.value, self.decomp.factors["H"].value)

    def normalisation(self):
        self.value = self.value.dot(norm_mat(self.value))


class oNMF_H(Factor):
    def __init__(self, decomp, **kwargs):
        kwargs["multiplicative"] = True
        super(oNMF_H, self).__init__(decomp, (decomp.ms, decomp.r, decomp.m), **kwargs)

    def grad_neg_pos(self):
        return DH(self.decomp.V, self.decomp.factors["W"].value, self.value)


class oNMF(Decomp):
    def __init__(self, V, r, ms, **kwargs):
        self.n, self.m = V.shape
        self.r = r
        self.ms = ms  # max shift
        super(oNMF, self).__init__(V, **kwargs)

    def factor_initialisation(self):
        self.factors["W"] = oNMF_W(self, label="W")
        self.factors["H"] = oNMF_H(self, label="H")

    def reconstruct(self):
        return cNMF_reconstruction(self.factors["W"].value, self.factors["H"].value)
