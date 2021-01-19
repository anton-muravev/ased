#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:46:18 2018

@author: muravev
"""
import numpy as np

def multiclass_matthews(confmat): 
    """Computes Matthews coefficient from the multiclass confusion matrix.
    
    Arguments
    ----------
    confmat: np.array
        A confusion matrix.
    
    Returns
    ----------
    A scalar with the Matthews coefficient value. If infinite, returns -2 
    (which is a soft bound on the possible values).
    """
    classes = confmat.shape[0]
    matthews_up = 0
    m1 = np.outer(np.diag(confmat), confmat.flatten()).sum()
    m2 = 0
    for k in range(classes):
        m2 += np.outer(confmat[k,:], confmat[:,k]).sum()
    matthews_up = m1-m2
    matthews_bleft = 0
    matthews_bright = 0
    for k in range(classes):
        ckl = np.sum(confmat[k,:])
        ckl_prime = np.sum(confmat)-ckl
        matthews_bleft += ckl*ckl_prime
        clk = np.sum(confmat[:,k])
        clk_prime = np.sum(confmat)-clk
        matthews_bright += clk*clk_prime
    matthews = matthews_up/( np.sqrt(matthews_bleft) * np.sqrt(matthews_bright) )
    if np.isfinite(matthews):
        return matthews
    else:
        return -2

def oneclass_matthews(confmat, refclass):
    """Computes Matthews coefficient from the one-class confusion matrix.
    
    Arguments
    ----------
    confmat: np.array
        A confusion matrix.
    
    Returns
    ----------
    A scalar with the Matthews coefficient value. If infinite, returns -2 
    (which is a soft bound on the possible values).
    """
    tp = confmat[refclass, refclass]
    tn = np.diag(confmat).sum()-tp
    fp = confmat[:,refclass].sum()-tp
    fn = confmat[refclass,:].sum()-tp
    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if denom==0:
        return -2
    else:
        return (tp*tn-fp*fn)/np.sqrt(denom)