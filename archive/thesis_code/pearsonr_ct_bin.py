#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:14:27 2020

@author: bariskuru
"""

import numpy as np

def ct_a_bin(arr, bin_start_end):
    bin_start = bin_start_end[0]
    bin_end = bin_start_end[1]
    counts = np.empty((arr.shape[0], arr.shape[1]))
    for index, value in np.ndenumerate(arr):
        # print(index)
        counts[index] = ((value > bin_start) & (value< bin_end)).sum()
    return counts

def pearson_r(x,y):
    #corr mat is doubled in each axis since it is 2d*2d
    corr_mat = np.corrcoef(x, y, rowvar=False) 
    #slice out the 1 of 4 identical mat
    corr_mat = corr_mat[int(corr_mat.shape[0]/2):, :int(corr_mat.shape[0]/2)] 
    # indices in upper triangle
    iu =np.triu_indices(int(corr_mat.shape[0]), k=1)
    # corr arr is the values vectorized 
    diag_low = corr_mat[iu]
    diag = corr_mat.diagonal()
    return diag, diag_low
