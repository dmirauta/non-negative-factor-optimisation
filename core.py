#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:00:03 2020

@author: Dumitru Mirauta
"""

import numpy as np
import time

def norm_mat(M):
    return np.diag( 1/M.sum(0) )

class Factor:
    
    def __init__(self, decomp, shape, label="", multiplicative=False):
        
        self.decomp=decomp
        self.shape = shape
        self.nvars = np.prod(shape)
        self.label = label
        
        update_mask = decomp.update_masks.get(self.label)
        if update_mask is None:
            self.update_mask = np.ones(self.shape).astype("bool")
        else:
            self.update_mask = update_mask.astype("bool")

        self.n_frozen = (self.update_mask==0).sum()
        self.no_update = (self.n_frozen==self.nvars)

        self.sub_opt_kwargs=decomp.sub_opt_kwargs.get(self.label, dict())

        init_val=decomp.initial_factors.get(self.label)
        if not init_val is None:
            self.value = init_val.copy()
        else:
            self.value = np.random.uniform(0, 1, self.shape)

        if multiplicative:
            self.sub_opt_step = self.mult_sub_opt_step
            
        #self.old_value = self.value.copy()
        
        self.pull_up()
    
    def sub_opt_step(self):
        raise Exception("Did not override sub_opt_step")

    def float_format(self):
        if "debug_n_decimals" in self.sub_opt_kwargs.keys():
            return "{:." + self.sub_opt_kwargs["debug_n_decimals"] + "f}"
        else:
            return "{:.8f}"
    
    def normalisation(self):
        pass
    
    def mult_sub_opt_step(self):

        if not self.no_update:
            grad_negative, grad_positive = self.grad_neg_pos()
            mult  = grad_negative[self.update_mask] + self.sub_opt_kwargs['delta_minus']
            mult /= grad_positive[self.update_mask] + self.sub_opt_kwargs['delta_plus']
            self.value[self.update_mask] *= mult

        self.normalisation()
        self.pull_up()
    
    def sub_optimisation(self):
        
        t0 = time.time()
        
        self.old_value = self.value.copy()
        
        i=0; diff=2*self.sub_opt_kwargs["tol"]
        while i<self.sub_opt_kwargs["max_iter"] and diff>self.sub_opt_kwargs["tol"]:
            temp_val = self.value.copy()
            self.sub_opt_step()
            diff = np.abs( self.value - temp_val ).max()
            i+=1
    
        t1 = time.time()
        
        if self.decomp.debug:
            print("    max difference: " + self.float_format().format(diff) + ", iters: " + "{:d}".format(i) + ", time elapsed: " + self.float_format().format(t1-t0))
            
    def pull_up(self, eps=1e-30):
        self.value[ (self.value<eps) ] = eps

class Decomp:
    
    def __init__(self, V,
                 debug=False, detailed_err=False, silent=True,
                 initial_factors = {},
                 update_masks = {},
                 sub_opt_kwargs = {},
                 opt_tol=1e-9,
                 opt_max_iter=1e3):
        
        self.iters = 0
        
        self.V = V # data matrix
        
        self.opt_tol = opt_tol
        self.opt_max_iter = opt_max_iter
        
        self.initial_factors = initial_factors
        self.update_masks = update_masks
        self.sub_opt_kwargs = sub_opt_kwargs

        self.factors=dict()
        self.factor_initialisation()
        
        self.debug = debug
        self.detailed_err = detailed_err
        self.silent = silent

        self.logs = {"REFN":[]}
    
        self.Fnorm_V = np.linalg.norm(V, ord='fro')
    
    def factor_initialisation(self):
        raise Exception("Did not override factor_initialisation")

    def recostruct(self):
        raise Exception("Did not override recostruct")

    def eval_cost(self):
        return None
    
    def calc_error(self, print_=False, dt=None):

        Va = self.reconstruct()
        
        cost = self.eval_cost()
        
        EFN = np.linalg.norm(self.V-Va, ord='fro')
        
        REFN = EFN/self.Fnorm_V
        self.logs["REFN"].append(REFN)
        
        if print_:
            str_ = "|E_i| = {:.6f}, REFN={:.6f}".format(EFN, REFN)
            
            if not cost is None:
                str_ += ", cost={:.6f}".format(cost)
                
            if not dt is None:
                str_ += " ({:.2f} secs elapsed)".format(dt)
            
            str_ += "\n"
            
            print(str_)
        
    def stop_condition(self, max_iter_, REFN_diff):
        return self.iters<max_iter_ and REFN_diff>self.opt_tol
        
    def global_loop(self, max_iter=1, opt_order=None):

        if opt_order is None:
            opt_order=self.factors.keys()
        
        if self.iters>1:
            REFN_diff= abs(self.logs["REFN"][-1]-self.logs["REFN"][-2])
        else:
            self.calc_error(self.detailed_err)
            REFN_diff= 2*self.opt_tol 
        
        max_iter_ = max_iter + self.iters #add previous iter

        while self.stop_condition(max_iter_, REFN_diff):
            
            if self.debug:
                print( "-"*10 + " i = {:d} ".format(self.iters) + "-"*10 )
            
            # optimise for one factor while keeping the others fixed
            for fk in opt_order:
                if self.debug:
                    print("{:s} subproblem:".format(fk))
                
                t0=time.time()
                self.factors[fk].sub_optimisation()
                t1=time.time()
                
                if self.detailed_err:
                    self.calc_error(True, dt=t1-t0)
            
            if not self.detailed_err:
                self.calc_error(not self.silent)
                
            REFN_diff = abs(self.logs["REFN"][-1]-self.logs["REFN"][-2])
            
            self.iters+=1
            
        #cleanup()

