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
import seaborn as sns

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
total_spikes_bin_one = np.zeros(2000)
total_spikes_bin_two = np.zeros(2000)

for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    granule_spikes_full = granule_spikes
    cycle_ms = end_time_one - start_time_one
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    if spike >= start_time_one and spike < end_time_one:
                        total_spikes_bin_one[idx] += 1
                        data_bin_one[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)
                    elif spike >= start_time_two and spike < end_time_two:
                        total_spikes_bin_two[idx] += 1
                        data_bin_two[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)

mean_phase_bin_one = np.array([np.nanmean(np.array(cell)) for cell in data_bin_one])
mean_phase_bin_two = np.array([np.nanmean(np.array(cell)) for cell in data_bin_two])
most_active = np.argpartition(total_spikes_bin_one + total_spikes_bin_two, -50)[-50:]

mean_phase_bin_one_full = mean_phase_bin_one.copy()
mean_phase_bin_two_full = mean_phase_bin_two.copy()
total_spikes_bin_one_full = total_spikes_bin_one.copy()
total_spikes_bin_two_full = total_spikes_bin_two.copy()
info_bin_one_full = total_spikes_bin_one_full / (total_spikes_bin_one_full + total_spikes_bin_two_full)
info_bin_two_full = total_spikes_bin_two_full / (total_spikes_bin_one_full + total_spikes_bin_two_full)


"""NO FEEDBACK"""
tuning = 'no-feedback'
data_bin_one = [[] for x in range(n_cells)]
data_bin_two = [[] for x in range(n_cells)]
total_spikes_bin_one = np.zeros(2000)
total_spikes_bin_two = np.zeros(2000)


for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    granule_spikes_nofb = granule_spikes
    cycle_ms = end_time_one - start_time_one
    total_spikes_per_cell = np.zeros(2000)
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    if spike >= start_time_one and spike < end_time_one:
                        total_spikes_bin_one[idx] += 1
                        data_bin_one[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)
                    elif spike >= start_time_two and spike < end_time_two:
                        total_spikes_bin_two[idx] += 1
                        data_bin_two[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)

mean_phase_bin_one = np.array([np.nanmean(np.array(cell)) for cell in data_bin_one])
mean_phase_bin_two = np.array([np.nanmean(np.array(cell)) for cell in data_bin_two])

mean_phase_bin_one_nofb = mean_phase_bin_one.copy()
mean_phase_bin_two_nofb = mean_phase_bin_two.copy()
total_spikes_bin_one_nofb = total_spikes_bin_one.copy()
total_spikes_bin_two_nofb = total_spikes_bin_two.copy()
info_bin_one_nofb = total_spikes_bin_one_nofb / (total_spikes_bin_one_nofb + total_spikes_bin_two_nofb)
info_bin_two_nofb = total_spikes_bin_two_nofb / (total_spikes_bin_one_nofb + total_spikes_bin_two_nofb)


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

fig, ax = plt.subplots(2, 2)
ax[0,0].scatter(info_bin_one_full, mean_phase_bin_one_full)
ax[0,1].scatter(info_bin_two_full, mean_phase_bin_two_full)
ax[1,0].scatter(info_bin_one_nofb, mean_phase_bin_one_nofb)
ax[1,1].scatter(info_bin_two_nofb, mean_phase_bin_two_nofb)

# ax[0,0].title("Full Network")
for row in ax:
    row[0].set_xlabel("# Spikes Bin 7 / (# Spikes Both Bins)")
    row[1].set_xlabel("# Spikes Bin 17 / (# Spikes Both Bins)")
    row[0].set_ylabel("Mean Phase Bin 7")
    row[1].set_ylabel("Mean Phase bin 17")
    for axis in row:
        axis.set_xlim((-0.1, 1.1))
        axis.set_ylim((0, 2*np.pi))


binarized_bins_full = np.zeros((2000, 20))
for idx, cell in enumerate(granule_spikes_full[75][0]):
    spiked_bins = np.array(cell / 100).astype(int)
    for s in spiked_bins:
        binarized_bins_full[idx, s] = 1

binarized_bins_nofb = np.zeros((2000, 20))
for idx, cell in enumerate(granule_spikes_nofb[75][0]):
    spiked_bins = np.array(cell / 100).astype(int)
    for s in spiked_bins:
        binarized_bins_nofb[idx, s] = 1

deleted_bins = (binarized_bins_full - binarized_bins_nofb) == -1


deleted_spikes = []
other_spikes = []
for idx, cell in enumerate(granule_spikes_nofb[75][0]):
    for spike in cell:
        # print(idx, cell, spike)
        #pdb.set_trace()
        if int(spike/ 100) in np.argwhere(deleted_bins[idx]):
            deleted_spikes.append(spike)
        else:
            other_spikes.append(spike)
            
deleted_spikes_phase = ((np.array(deleted_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi
other_spikes_phase = ((np.array(other_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi

all_spikes = np.concatenate((deleted_spikes_phase, other_spikes_phase))
all_spikes_label = ['deleted'] * deleted_spikes_phase.size + ['non-deleted'] * other_spikes_phase.size


df_spikes = pd.DataFrame({'spike_phase': all_spikes, 'spike_label': all_spikes_label})







