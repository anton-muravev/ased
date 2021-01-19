#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:46:18 2018

@author: muravev
"""
import numpy as np
import py3nvml

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

def get_gpu_allocation(total_netcount, gpu_allowed):
    #REMOVE THIS WHEN MOVED
    """Obtains the allocation pattern of the networks between GPUs, assuming
    independent training.
    
    Arguments
    ----------
    total_netcount: int
        A total number of networks to be processed.
    gpu_allowed: list of int
        A list containing the identifiers of the GPUs that are allowed for 
        consideration.
    
    Returns
    ----------
    A dictionary, where the keys are the GPU identifiers and the values are 
    integers indicating how many networks that GPU should have allocated.
    """
    gpu_usable = gpu_allowed.copy()
    free_gpus = py3nvml.get_free_gpus()
    for i, k in enumerate(free_gpus):
        if not k and i in gpu_allowed:
            gpu_usable.remove(i)
    each = np.int(total_netcount/len(gpu_usable))
    output = {}
    for k in gpu_usable:
        output[k] = each
    return output