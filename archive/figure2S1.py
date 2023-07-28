# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:38:23 2022

@author: Daniel
"""

from phase_to_rate.neural_coding import load_spikes, rate_n_phase, load_spikes_DMK
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import matplotlib as mpl
import pdb

dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')

trajectories = [75]

n_samples = 20
grid_seeds = np.arange(1,2,1)

tuning = 'full'

ns_grid_phases = np.empty(0)
ns_granule_phases = np.empty(0)
s_grid_phases = np.empty(0)
s_granule_phases = np.empty(0)

start_time_one = 600
end_time_one = 700
start_time_two = 1600
end_time_two = 1700

n_cells = 2000

data_bin_one = [[] for x in range(n_cells)]
data_bin_two = [[] for x in range(n_cells)]

for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    cycle_ms = end_time_one - start_time_one
    total_spikes_per_cell = np.zeros(2000)
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    if spike >= start_time_one and spike < end_time_one:
                        total_spikes_per_cell[idx] += 1
                        data_bin_one[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)
                    elif spike >= start_time_two and spike < end_time_two:
                        total_spikes_per_cell[idx] += 1
                        data_bin_two[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)

mean_phase_bin_one = np.array([np.nanmean(np.array(cell)) for cell in data_bin_one])
mean_phase_bin_two = np.array([np.nanmean(np.array(cell)) for cell in data_bin_two])
most_active = np.argpartition(total_spikes_per_cell, -50)[-50:]

mean_phase_bin_one_full = mean_phase_bin_one.copy()
mean_phase_bin_two_full = mean_phase_bin_two.copy()


"""NO FEEDBACK"""
tuning = 'no-feedback'
data_bin_one = [[] for x in range(n_cells)]
data_bin_two = [[] for x in range(n_cells)]

for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'adjusted_data', str(tuning),  'collective', 'grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_[75]_100-119_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes_DMK(ns_path, "granule", trajectories, n_samples)
    cycle_ms = end_time_one - start_time_one
    total_spikes_per_cell = np.zeros(2000)
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    if spike >= start_time_one and spike < end_time_one:
                        data_bin_one[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)
                    elif spike >= start_time_two and spike < end_time_two:
                        data_bin_two[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)

mean_phase_bin_one = np.array([np.nanmean(np.array(cell)) for cell in data_bin_one])
mean_phase_bin_two = np.array([np.nanmean(np.array(cell)) for cell in data_bin_two])

mean_phase_bin_one_nofb = mean_phase_bin_one.copy()
mean_phase_bin_two_nofb = mean_phase_bin_two.copy()

"""CALCULATE ROC CURVE"""
x = np.concatenate([mean_phase_bin_one_full, mean_phase_bin_two_full, mean_phase_bin_one_nofb, mean_phase_bin_two_nofb])
y = np.concatenate([np.zeros(mean_phase_bin_one_full.size + mean_phase_bin_two_full.size), np.ones(mean_phase_bin_one_nofb.size + mean_phase_bin_two_nofb.size)])

not_nan = np.argwhere(~np.isnan(x))[:,0]

x = x[not_nan]
y = y[not_nan]

parameters = np.arange(0, np.pi * 2, 0.01)

tpr_list = []
fpr_list = []
# sys.exit()
for param in parameters:
    classification = x > param
    tp = np.sum(np.logical_and(classification == 1, y == 1))
    fn = np.sum(np.logical_and(classification == 0, y == 1))
    fp = np.sum(np.logical_and(classification == 1, y == 0))
    tn = np.sum(np.logical_and(classification == 0, y == 0))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)

tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)

optimal_tradeoff = tpr_list - fpr_list

optimal_threshold = parameters[optimal_tradeoff.argmax()]
    
"""PLOTTING"""
plt.figure()
plt.scatter(mean_phase_bin_one_full[most_active], mean_phase_bin_two_full[most_active])
plt.xlim((0,2*np.pi))
plt.ylim((0,2*np.pi))
plt.hlines(optimal_threshold,xmin=0, xmax=2*np.pi)
plt.vlines(optimal_threshold,ymin=0, ymax=2*np.pi)
plt.xlabel("Mean Phase Bin # 7")
plt.ylabel("Mean Phase Bin # 17")


plt.figure()
plt.scatter(mean_phase_bin_one_nofb[most_active], mean_phase_bin_two_nofb[most_active])
plt.xlim((0,2*np.pi))
plt.ylim((0,2*np.pi))
plt.hlines(optimal_threshold,xmin=0, xmax=2*np.pi)
plt.vlines(optimal_threshold,ymin=0, ymax=2*np.pi)
plt.xlabel("Mean Phase Bin # 7")
plt.ylabel("Mean Phase Bin # 17")