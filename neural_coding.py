#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:47:21 2021

@author: bariskuru
"""


import numpy as np
import copy



def _spike_counter(spike_times, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells = len(spike_times)
    counts = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, value in enumerate(spike_times):
            curr_ct = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            counts[idx, i] = curr_ct
    return counts



def _phase_definer(spike_times, nan_fill=True, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells = len(spike_times)
    phases = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, val in enumerate(spike_times):
            curr_train = val[((bin_size_ms*(i) < val) & (val < bin_size_ms*(i+1)))]
            if curr_train.size != 0:
                phases[idx, i] = np.mean(curr_train%(bin_size_ms)/(bin_size_ms)*2*np.pi)
    if nan_fill is True:
        mean_phases = np.mean(phases[phases != 0])
        phases[phases == 0] = mean_phases
    return phases
    


def rate_n_phase(spike_times, poiss_seeds, trajs, nan_fill=False, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms/bin_size_ms)
    n_traj = len(trajs)
    n_poiss = len(poiss_seeds)
    n_cell = len(spike_times[poiss_seeds[0]][0])
    counts = np.empty((n_cell, n_bins, n_traj, n_poiss))
    phases = np.empty((n_cell, n_bins, n_traj, n_poiss))
    rate_code = np.empty((2*n_cell*n_bins, n_traj, n_poiss))
    phase_code = np.empty((2*n_cell*n_bins, n_traj, n_poiss))
    polar_code = np.empty((2*n_cell*n_bins, n_traj, n_poiss))
    for seed_idx, seed in enumerate(poiss_seeds):
        spike_times_single_seed = spike_times[seed]
        for traj_idx in range(n_traj):
            spike_times_single_traj = spike_times_single_seed[traj_idx]
            single_count = _spike_counter(spike_times_single_traj, bin_size_ms=bin_size_ms, dur_ms=dur_ms)
            single_phase = _phase_definer(spike_times_single_traj, bin_size_ms=bin_size_ms, dur_ms=dur_ms)
            counts[:, :, traj_idx, seed_idx] = single_count
            phases[:, :, traj_idx, seed_idx] = single_phase
            single_rate_code, single_phase_code, single_polar_code = code_maker(single_count, single_phase, n_cell, n_bins)
            rate_code[:, traj_idx, seed_idx] = single_rate_code
            phase_code[:, traj_idx, seed_idx] = single_phase_code
            polar_code[:, traj_idx, seed_idx] = single_polar_code
    return counts, phases, rate_code, phase_code, polar_code


def code_maker(single_count, single_phase, phase_of_rate_code=np.pi/4, rate_in_phase=1):
    single_count = single_count.flatten('C')
    single_phase = single_phase.flatten('C')

    cts_for_phase = copy.deepcopy(single_count)
    # change rate code to mean of non zeros where it is nonzero
    # cts_for_phase[cts_for_phase!=0]=np.mean(cts_for_phase[cts_for_phase!=0])
    cts_for_phase[cts_for_phase != 0] = rate_in_phase

    # rate code with constant 45 deg phase
    rate_y = single_count*np.sin(phase_of_rate_code)
    rate_x = single_count*np.cos(phase_of_rate_code)
    rate_code = np.concatenate((rate_x, rate_y), axis=None)

    # phase code with phase and mean rate
    phase_y = cts_for_phase*np.sin(single_phase)
    phase_x = cts_for_phase*np.cos(single_phase)
    phase_code = np.concatenate((phase_x, phase_y), axis=None)

    # polar code with rate and phase
    polar_y = single_count*np.sin(single_phase)
    polar_x = single_count*np.cos(single_phase)
    polar_code = np.concatenate((polar_x, polar_y), axis=None)

    return rate_code, phase_code, polar_code




poiss_seeds = np.array([100, 201, 302])#

trajs = [75, 74, 65]


poiss_seeds = np.array([150, 250, 350])
trajs =  [75, 74, 73, 70]

counts, phases, rate_code, phase_code, polar_code = rate_n_phase(test_grids, poiss_seeds, trajs)

# counts1, phases1, rate_code1, phase_code1, polar_code1 = rate_n_phase(test_grids, poiss_seeds, trajs, nan_fill=False)

from scipy.stats import pearsonr,  spearmanr


pearson1 = pearsonr(rate_code[:,0,0], rate_code[:,1,0])# 75 vs 74 same poiss
pearson2 = pearsonr(rate_code[:,0,0], rate_code[:,0,1])# 75 vs 75 diff poiss
pearson3 = pearsonr(rate_code[:,0,0], rate_code[:,1,1])# 75 vs 74 diff poiss
pearson31 = pearsonr(rate_code[:,0,0], rate_code[:,2,0])# 75 vs 73 same poiss
pearson32 = pearsonr(rate_code[:,0,0], rate_code[:,2,1])# 75 vs 73 diff poiss
pearson33 = pearsonr(rate_code[:,0,0], rate_code[:,3,0])# 75 vs 70 same poiss
pearson34 = pearsonr(rate_code[:,0,0], rate_code[:,3,1])# 75 vs 70 diff poiss

pearson10 = pearsonr(rate_code[:,1,0], rate_code[:,2,0])# 74 vs 73 same poiss
pearson20 = pearsonr(rate_code[:,1,0], rate_code[:,1,1])# 74 vs 74 diff poiss
pearson30 = pearsonr(rate_code[:,1,0], rate_code[:,2,1])# 74 vs 73 diff poiss
pearson310 = pearsonr(rate_code[:,1,0], rate_code[:,3,0])# 74 vs 70 same poiss
pearson320 = pearsonr(rate_code[:,1,0], rate_code[:,3,1])# 74 vs 70 diff poiss

print(pearson1)
print(pearson2)
print(pearson3)
print(pearson31)
print(pearson32)
print(pearson33)
print(pearson34)
print('                ')
print(pearson10)
print(pearson20)
print(pearson30)
print(pearson310)
print(pearson320)





spearman1 = spearmanr(rate_code[:,0,0], rate_code[:,1,0])# 75 vs 74 same poiss
spearman2 = spearmanr(rate_code[:,0,0], rate_code[:,0,1])# 75 vs 75 diff poiss
spearman3 = spearmanr(rate_code[:,0,0], rate_code[:,1,1])# 75 vs 74 diff poiss
spearman31 = spearmanr(rate_code[:,0,0], rate_code[:,2,0])# 75 vs 73 same poiss
spearman32 = spearmanr(rate_code[:,0,0], rate_code[:,2,1])# 75 vs 73 diff poiss
spearman33 = spearmanr(rate_code[:,0,0], rate_code[:,3,0])# 75 vs 70 same poiss
spearman34 = spearmanr(rate_code[:,0,0], rate_code[:,3,1])# 75 vs 70 diff poiss

spearman10 = spearmanr(rate_code[:,1,0], rate_code[:,2,0])# 74 vs 73 same poiss
spearman20 = spearmanr(rate_code[:,1,0], rate_code[:,1,1])# 74 vs 74 diff poiss
spearman30 = spearmanr(rate_code[:,1,0], rate_code[:,2,1])# 74 vs 73 diff poiss
spearman310 = spearmanr(rate_code[:,1,0], rate_code[:,3,0])# 74 vs 70 same poiss
spearman320 = spearmanr(rate_code[:,1,0], rate_code[:,3,1])# 74 vs 70 diff poiss

print(spearman1)
print(spearman2)
print(spearman3)
print(spearman31)
print(spearman32)
print(spearman33)
print(spearman34)
print('                ')
print(spearman10)
print(spearman20)
print(spearman30)
print(spearman310)
print(spearman320)






pearson1 = pearsonr(phase_code[:,0,0], phase_code[:,1,0])# 75 vs 74 same poiss
pearson2 = pearsonr(phase_code[:,0,0], phase_code[:,0,1])# 75 vs 75 diff poiss
pearson3 = pearsonr(phase_code[:,0,0], phase_code[:,1,1])# 75 vs 74 diff poiss
pearson31 = pearsonr(phase_code[:,0,0], phase_code[:,2,0])# 75 vs 73 same poiss
pearson32 = pearsonr(phase_code[:,0,0], phase_code[:,2,1])# 75 vs 73 diff poiss
pearson33 = pearsonr(phase_code[:,0,0], phase_code[:,3,0])# 75 vs 70 same poiss
pearson34 = pearsonr(phase_code[:,0,0], phase_code[:,3,1])# 75 vs 70 diff poiss

pearson10 = pearsonr(phase_code[:,1,0], phase_code[:,2,0])# 74 vs 73 same poiss
pearson20 = pearsonr(phase_code[:,1,0], phase_code[:,1,1])# 74 vs 74 diff poiss
pearson30 = pearsonr(phase_code[:,1,0], phase_code[:,2,1])# 74 vs 73 diff poiss
pearson310 = pearsonr(phase_code[:,1,0], phase_code[:,3,0])# 74 vs 70 same poiss
pearson320 = pearsonr(phase_code[:,1,0], phase_code[:,3,1])# 74 vs 70 diff poiss

print(pearson1)
print(pearson2)
print(pearson3)
print(pearson31)
print(pearson32)
print(pearson33)
print(pearson34)
print('                ')
print(pearson10)
print(pearson20)
print(pearson30)
print(pearson310)
print(pearson320)



pearson1 = pearsonr(polar_code[:,0,0], polar_code[:,1,0]) # 75 vs 74 same poiss
pearson2 = pearsonr(polar_code[:,0,0], polar_code[:,0,1]) # 75 vs 75 diff poiss
pearson3 = pearsonr(polar_code[:,0,0], polar_code[:,1,2]) # 75 vs 74 diff poiss
pearson31 = pearsonr(polar_code[:,0,0], polar_code[:,2,0]) # 75 vs 65 same poiss
pearson32 = pearsonr(polar_code[:,0,0], polar_code[:,2,1]) # 75 vs 65 diff poiss

print(pearson1)
print(pearson2)
print(pearson3)
print(pearson31)
print(pearson32)



pearson1 = pearsonr(phase_code[:,0,0], phase_code[:,1,0])
pearson2 = pearsonr(phase_code[:,0,0], phase_code[:,0,1])
pearson3 = pearsonr(phase_code[:,0,0], phase_code[:,1,1])
pearson31 = pearsonr(phase_code[:,0,0], phase_code[:,2,0])
pearson32 = pearsonr(phase_code[:,0,0], phase_code[:,2,1])

pearson4 = pearsonr(counts[:,:,0,0].flatten(), counts[:,:,0,1].flatten())
pearson5 = pearsonr(counts[:,:,0,0].flatten(), counts[:,:,1,0].flatten())
pearson6 = pearsonr(counts[:,:,0,0].flatten(), counts[:,:,1,1].flatten())


pearson4 = spearmanr(counts[:,:,0,0].flatten(), counts[:,:,0,1].flatten())
pearson5 = spearmanr(counts[:,:,0,0].flatten(), counts[:,:,1,0].flatten())
pearson6 = spearmanr(counts[:,:,0,0].flatten(), counts[:,:,1,1].flatten())

pearson4 = spearmanr(counts[:,:,0,1].flatten(), counts[:,:,1,1].flatten())
pearson5 = spearmanr(counts[:,:,0,1].flatten(), counts[:,:,0,2].flatten())
pearson6 = spearmanr(counts[:,:,0,1].flatten(), counts[:,:,1,2].flatten())

# code maker is fine but different poisson seeds doesnt really funtion
# they are different but correlation values look weird
# maybe counter has a problem

import matplotlib.pyplot as plt

plt.imshow(counts[:,:,0,0], aspect='auto')
plt.figure()
plt.imshow(counts[:,:,0,1], aspect='auto')
plt.figure()
plt.imshow(counts[:,:,1,1], aspect='auto')
