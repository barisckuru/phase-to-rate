#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:35:32 2020

@author: bariskuru
"""

'''poisson noise for variable max firing rate for grid cells'''

import seaborn as sns, numpy as np
import os
from grid_poiss_input_gen import inhom_poiss
from grid_trajs import grid_population, draw_traj
import matplotlib.pyplot as plt
from pearsonr_ct_bin import ct_a_bin, pearson_r

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

max_rates = np.array([20,40,100,160,200])
par_trajs_pf = []



for rate in max_rates:
    np.random.seed(seed_1) #seed_1 for granule cell generation
    grids = grid_population(n_grid, rate, seed_1)[0]
    par_trajs_pf.append(draw_traj(grids, n_grid, par_trajs)[0])

counts = np.empty((max_rates.size, seed_2s.size), dtype=np.ndarray)
# generate temporal patterns out of grid cell act profiles as an input for pyDentate
for idx, val in np.ndenumerate(counts):
    print(idx)
    counts[idx] = ct_a_bin(inhom_poiss(par_trajs_pf[idx[0]], n_traj, dt_s=0.0001, seed=seed_2s[idx[1]]), [2000,2500])


poiss_noise = np.empty(5, np.ndarray)

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
arr = np.array([0,1,2,3,4])
hist_bin = 0.02
for i in arr:
    sns.distplot(poiss_noise[i], np.arange(0, 1+hist_bin, hist_bin), rug=True)
plt.legend(max_rates[arr], title='Max Rates (Hz)')

parameters = ('number of grids = '+str(n_grid)+'      parallel trajs= '+str(par_trajs)+' cm       bin size = 500 ms'+
              '       seed1='+str(seed_1)+', seed2='+str(seed_2s)+', seed3='+str(seed_3))
plt.annotate(parameters, (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)

plt.title('Distributions of Input Correlations for Various Max Rates')

plt.xlabel('Rin')
plt.ylabel('Count')












