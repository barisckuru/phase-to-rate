#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:52:52 2021

@author: baris
"""


import numpy as np
import copy
import shelve


def _spike_counter(spike_times, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells = len(spike_times)
    counts = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, value in enumerate(spike_times):
            curr_ct = ((bin_size_ms*(i) < value) &
                       (value < bin_size_ms*(i+1))).sum()
            counts[idx, i] = curr_ct
    return counts


def _phase_definer(spike_times, nan_fill=True, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells = len(spike_times)
    phases = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, val in enumerate(spike_times):
            curr_train = val[((bin_size_ms*(i) < val) &
                              (val < bin_size_ms*(i+1)))]
            if curr_train.size != 0:
                phases[idx, i] = np.mean(curr_train % (bin_size_ms) /
                                         (bin_size_ms)*2*np.pi)
    if nan_fill is True:
        mean_phases = np.mean(phases[phases != 0])
        phases[phases == 0] = mean_phases
    return phases


def _code_maker(single_count, single_phase,
                phase_of_rate_code=np.pi/4, rate_in_phase=1):
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


def rate_n_phase(spike_times, poiss_seeds, trajs, nan_fill=False, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms/bin_size_ms)
    n_poiss = len(poiss_seeds)
    print(type(spike_times))
    n_cell = len(spike_times)

    spike_times_single_traj = spike_times
    single_count = _spike_counter(spike_times_single_traj, bin_size_ms=bin_size_ms, dur_ms=dur_ms)
    single_phase = _phase_definer(spike_times_single_traj, bin_size_ms=bin_size_ms, dur_ms=dur_ms)
    single_rate_code, single_phase_code, single_polar_code = _code_maker(single_count, single_phase, n_cell, n_bins)

    return single_count, single_phase, single_rate_code, single_phase_code, single_polar_code



def load_spikes(path, cell_type, trajectories, n_samples):
    storage = shelve.open(path)
    spikes = {}
    for traj in trajectories:
        requested_spikes = []
        traj_key = str(traj)
        poisson_seeds = storage[traj_key]['parameters']['poisson_seeds']
        if n_samples > len(poisson_seeds):
            raise Exception('Too much samples requested!')
        elif n_samples < 1:
            raise Exception('n_samples should be larger than 0!')
        else:
            poisson_seeds = poisson_seeds[0:n_samples]
        
        if cell_type == 'grid':
            all_spikes = storage[traj_key]['grid_spikes']
        elif cell_type == 'granule':
            all_spikes = storage[traj_key]['granule_spikes']
        else:
            raise Exception('Cell type does not exist!')
        for poisson in poisson_seeds:
            requested_spikes.append(all_spikes[poisson])
        spikes[traj] = requested_spikes
    return spikes







# poiss_seeds = np.array([100, 201, 302])

# trajs = [75, 74, 65]


# poisson_seeds = np.array([100])



# # test_grids = storage['grid_spikes']["non-shuffled"]

# # test_gras = storage['granule_spikes']["shuffled"]


# test_gras_75 = storage['75']['grid_spikes'][100]
# counts_75, phases_75, rate_code_75, phase_code_75, polar_code_75 = rate_n_phase(test_gras_75, [100], [75], dur_ms=dur_ms)

# test_gras_75_2 = storage['75']['grid_spikes'][101]
# counts_75_2, phases_75_2, rate_code_75_2, phase_code_75_2, polar_code_75_2 = rate_n_phase(test_gras_75_2, [100], [75], dur_ms=dur_ms)



# seed =  storage['74']['parameters']['poisson_seeds'][0]
# test_gras_74 = storage['74']['grid_spikes'][seed]
# counts_74, phases_74, rate_code_74, phase_code_74, polar_code_74 = rate_n_phase(test_gras_74, [100], [74], dur_ms=dur_ms)


# seed_2 =  storage['74']['parameters']['poisson_seeds'][1]
# test_gras_74_2 = storage['74']['grid_spikes'][seed]
# counts_74_2, phases_74_2, rate_code_74_2, phase_code_74_2, polar_code_74_2 = rate_n_phase(test_gras_74_2, [100], [74], dur_ms=dur_ms)



# test_gras1 = granule_spikes[75][101]
# trajs1 =  [75]
# poisson_seed1 = [101]
# counts1, phases1, rate_code1, phase_code1, polar_code1 = rate_n_phase(test_gras1, poisson_seed1, trajs1, dur_ms=dur_ms)


# from scipy.stats import pearsonr,  spearmanr
# pearson = pearsonr(counts_75[:,0], counts_74[:,0])
# pearson1 = pearsonr(rate_code_75, rate_code_74)
# pearson2 = pearsonr(phase_code_75, phase_code_74)
# pearson3 = pearsonr(polar_code_75, polar_code_74)


# pearson = pearsonr(counts_75[:,0], counts_75_2[:,0])
# pearson1 = pearsonr(rate_code_75, rate_code_75_2)
# pearson2 = pearsonr(phase_code_75, phase_code_75_2)
# pearson3 = pearsonr(polar_code_75, polar_code_75_2)



