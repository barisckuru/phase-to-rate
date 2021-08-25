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
    


def rate_n_phase(spike_times, poiss_seeds, trajs, nan_fill=True, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms/bin_size_ms)
    n_traj = len(trajs)
    n_poiss = len(poiss_seeds)
    n_cell = len(spike_times[poiss_seeds[0]][0])
    counts = np.empty((n_cell, n_bins, n_traj, n_poiss))
    phases = np.empty((n_cell, n_bins, n_traj, n_poiss))
    for seed_idx, seed in enumerate(poiss_seeds):
        spike_times_single_seed = spike_times[seed]
        for traj_idx in range(n_traj):
            spike_times_sigle_traj = spike_times_single_seed[traj_idx]
            single_count = _spike_counter(spike_times_sigle_traj, bin_size_ms=bin_size_ms, dur_ms=dur_ms)
            single_phase = _phase_definer(spike_times_sigle_traj, bin_size_ms=bin_size_ms, dur_ms=dur_ms)
            counts[:,:,traj_idx,seed_idx] = single_count
            phases[:,:,traj_idx,seed_idx] = single_phase
    return counts, phases
    

# phases = _phase_definer(test_grids[100][0])     
    
# counts = _spike_counter(test_grids[100][0])     

# poiss_seeds = np.array([100, 101, 102])
# trajs = [75, 73]
    
# counts, phases = rate_n_phase(test_grids, poiss_seeds, trajs)


    
    

'code_maker will be updated!'
def code_maker(spike_cts, phases):
    
    #change rate code to mean of non zeros where it is nonzero
    cts_for_phase = copy.deepcopy(spike_cts)
    # cts_for_phase[cts_for_phase!=0]=np.mean(cts_for_phase[cts_for_phase!=0]) #was 1
    cts_for_phase[cts_for_phase!=0]=1

    #rate code with constant 45 deg phase
    phase_of_rate_code = np.pi/4
    rate_y = spike_cts*np.sin(phase_of_rate_code)
    rate_x = spike_cts*np.cos(phase_of_rate_code)
    rate_code =  np.concatenate((rate_y, rate_x), axis=1)

    #phase code with phase and mean rate 
    phase_y = cts_for_phase*np.sin(phases)
    phase_x = cts_for_phase*np.cos(phases)
    phase_code =  np.concatenate((phase_y, phase_x), axis=1)

    #complex code with rate and phase
    polar_y = spike_cts*np.sin(phases)
    polar_x = spike_cts*np.cos(phases)
    polar_code = np.concatenate((polar_y, polar_x), axis=1)

    return rate_code, phase_code, polar_code








