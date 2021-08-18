#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:36:14 2020

@author: bariskuru
"""

import seaborn as sns, numpy as np
import os
from grid_poiss_input_gen import inhom_poiss
from grid_trajs import grid_population, draw_traj
from pearsonr_ct_bin import ct_a_bin, pearson_r
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



counts_ms = np.array([200,500,1000,2000,5000])

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
    

poiss_noise = np.empty(5, np.ndarray)

for idx, poiss in enumerate(poiss_noise):
    poiss_noise[idx] = np.concatenate((pearson_r(counts[idx,0], counts[idx,1])[0],
                                       pearson_r(counts[idx,0], counts[idx,2])[0],
                                       pearson_r(counts[idx,0], counts[idx,3])[0],
                                       pearson_r(counts[idx,1], counts[idx,2])[0],
                                       pearson_r(counts[idx,1], counts[idx,3])[0],
                                       pearson_r(counts[idx,2], counts[idx,3])[0]),  axis=None)
    
import matplotlib.font_manager #for linux
'Plotting'
sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)
plt.figure()
arr = np.array([0,1,2,3,4])
hist_bin = 0.02
for i in arr:
    sns.distplot(poiss_noise[i], np.arange(0, 1+hist_bin, hist_bin), rug=True)
plt.legend(counts_ms[arr], title='Bin Size(ms)')
        
plt.title('Distributions of Input Correlations for Various Number of Grids')
parameters = ('number of grids = '+str(n_grid)+ '      parallel trajs = '+str(par_trajs)+' cm      max rate = '+str(max_rate)+
              '       seed1='+str(seed_1)+', seed2='+str(seed_2s)+', seed3='+str(seed_3))
plt.annotate(parameters, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)
plt.xlabel('Rin')
plt.ylabel('Count')







