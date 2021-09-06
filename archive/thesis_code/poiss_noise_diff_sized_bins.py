#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:36:14 2020

@author: bariskuru
"""

import seaborn as sns, numpy as np
import os
from grid_poiss_input_gen import inhom_poiss
from gridcell_traj_activ_pf import grid_population, draw_traj
import matplotlib.pyplot as plt

'''Poiss noise differences for variable sizes of time bins'''

savedir = os.getcwd()
input_scale = 1000

seed_1 = 100 #seed_1 for granule cell generation
seed_2_1 = 200 #seed_2 inh poisson input generation
seed_2_2 = 201
seed_2_3 = 202
seed_2_4 = 203
seed_3 = seed_1+50 #seed_3 for network generation & simulation
seed_2s=np.array([seed_2_1, seed_2_2, seed_2_3, seed_2_4])

#number of cells
n_grid = 200 
n_granule = 2000
n_mossy = 60
n_basket = 24
n_hipp = 24

# parameters for gc grid generator
par_trajs = np.array([75,74,73,71,67,59,43,11])
n_traj = par_trajs.shape[0]
max_rate = 20

np.random.seed(seed_1) #seed_1 for granule cell generation
all_grids = grid_population(n_grid, max_rate, seed_1)[0]
par_traj, dur_ms, dt_s = draw_traj(all_grids, n_grid, par_trajs)
dur_ms = 5000



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



counts_ms = np.array([100,200,400,500,1000,2000,3000,4000,5000])

# counts_ms = np.array([500, 600, 700, 800, 900, 1000])

bin_end = np.array([x+2000 if (x+2000)<5000 else 5000 for x in counts_ms])
bin_start = bin_end-counts_ms
bins_ms = np.concatenate((bin_start, bin_end), 0).reshape((2, bin_start.shape[0])).T


counts = np.empty((counts_ms.size, seed_2s.size), dtype=np.ndarray)
# generate temporal patterns out of grid cell act profiles as an input for pyDentate
inputs= []
for seed in seed_2s:
    inputs.append(inhom_poiss(par_traj, n_traj, dt_s=0.0001, seed=seed))

for idx, value in np.ndenumerate(counts):
    counts[idx] = ct_a_bin(inputs[idx[1]], bins_ms[idx[0]])
    

poiss_noise = np.empty(9, np.ndarray)

for idx, poiss in enumerate(poiss_noise):
    poiss_noise[idx] = np.concatenate((pearson_r(counts[idx,0], counts[idx,1])[0],
                                       pearson_r(counts[idx,0], counts[idx,2])[0],
                                       pearson_r(counts[idx,0], counts[idx,3])[0],
                                       pearson_r(counts[idx,1], counts[idx,2])[0],
                                       pearson_r(counts[idx,1], counts[idx,3])[0],
                                       pearson_r(counts[idx,2], counts[idx,3])[0]),  axis=None)
    

'Plotting'
sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)
plt.figure()

counts_ms = np.array([100,200,400,500,1000,2000,3000,4000,5000])
                      
                      
arr = np.array([3,4,5,8])
hist_bin = 0.02
for i in arr:
    sns.distplot(poiss_noise[i], np.arange(0, 1+hist_bin, hist_bin), rug=True)
plt.legend(counts_ms[arr])
        
plt.title('Distributions of Input Correlations in differently sized bins')
plt.xlabel('Rin')
plt.ylabel('Count')



# plt.legend(([100, 200, 400, 500, 1000, 2000, 3000, 4000, 5000]))  
# plt.legend(([500, 1000, 5000]))  

# plt.title('Distributions of Input Correlations in different size of Bins')
# plt.xlabel('Rin')
# plt.ylabel('Count')


# plt.figure()
# sns.distplot(poiss_noise_100ms, rug=True)
# sns.distplot(poiss_noise_200ms, rug=True)
# sns.distplot(poiss_noise_400ms, rug=True)
# sns.distplot(poiss_noise_500ms, rug=True)
# sns.distplot(poiss_noise_1000ms, rug=True)
# sns.distplot(poiss_noise_2000ms, rug=True)
# sns.distplot(poiss_noise_3000ms, rug=True)
# sns.distplot(poiss_noise_4000ms, rug=True)
# sns.distplot(poiss_noise_5000ms, rug=True)
# plt.legend(([100, 200, 400, 500, 1000, 2000, 3000, 4000, 5000]))  
# plt.legend(([500, 1000, 5000]))  



# '''poisson noise for variable max firing rate for grid cells'''



























