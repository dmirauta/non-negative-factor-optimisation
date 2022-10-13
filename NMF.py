#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:29:18 2020

@author: Dumitru Mirauta

Two factor NMF classes, only KL multiplicative updates currently implemented here.

"""

import numpy as np
import time, os, sys

from NFO.core import Factor, Decomp, norm_mat

ε=1e-19

class KL_NMF_W(Factor):
    
    def __init__(self, decomp, **kwargs):
        kwargs['multiplicative'] = True
        self.ones = np.ones( decomp.V.shape)
        super(KL_NMF_W, self).__init__(decomp, (decomp.n, decomp.r), **kwargs)
        
    def grad_neg_pos(self):
        V, W, H = self.decomp.V, self.value, self.decomp.factors['H'].value
        return ( np.matmul( np.divide(V, np.matmul(W,H)+ε), H.T ),
                 np.matmul( self.ones, H.T) )
    
    def normalisation(self):
        self.value = np.matmul(self.value, norm_mat(self.value))

class KL_NMF_H(Factor):
    
    def __init__(self, decomp, **kwargs):
        kwargs['multiplicative'] = True
        self.ones = np.ones( decomp.V.shape)
        super(KL_NMF_H, self).__init__(decomp, (decomp.r, decomp.m), **kwargs)
        
    def grad_neg_pos(self):
        V, W, H = self.decomp.V, self.decomp.factors['W'].value, self.value
        return ( np.matmul(W.T, np.divide(V, np.matmul(W,H)+ε)),
                 np.matmul(W.T, self.ones) )
    
class KLNMF(Decomp):
    
    def __init__(self, V, r, **kwargs):
        self.n, self.m = V.shape
        self.r = r
        super(KLNMF, self).__init__(V, **kwargs)
        
    def factor_initialisation(self):
        self.factors['W'] = KL_NMF_W(self, label="W")
        self.factors['H'] = KL_NMF_H(self, label="H")
        
    def reconstruct(self):
        return np.matmul(self.factors['W'].value, self.factors['H'].value )

