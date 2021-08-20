#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:47:21 2021

@author: bariskuru
"""


import numpy as np
import copy



def mean_phase(spikes, bin_size_ms, n_phase_bins, dur_ms):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells =spikes.shape[0] 
    n_traj = spikes.shape[1]
    rad = n_phase_bins/360*2*np.pi
    phases = np.zeros((n_bins, n_cells, n_traj))

    for i in range(n_bins):
        for idx, val in np.ndenumerate(spikes):
            curr_train = val[((bin_size_ms*(i) < val) & (val < bin_size_ms*(i+1)))]
            if curr_train.size != 0:
                phases[i][idx] = np.mean(curr_train%(bin_size_ms)/(bin_size_ms)*rad)
    return phases

def mean_filler(phases):
    mean_phases = np.mean(phases[phases!=0])
    phases[phases==0] = mean_phases
    return phases

#Count the number of spikes in bins 

def binned_ct(arr, bin_size_ms, dt_ms=25, time_ms=5000):
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    n_traj = arr.shape[1]
    counts = np.zeros((n_bins, n_cells, n_traj))
    for i in range(n_bins):
        for index, value in np.ndenumerate(arr):
            curr_ct = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            counts[i][index] = curr_ct
            #search and count the number of spikes in the each bin range
    return counts



def spikes_n_phases(grid_spikes, gra_spikes, dur_ms, grid_seed, poiss_seeds, tune, shuffle, f=10, shift_deg=180):
    T = 1/f
    bin_size_ms = T*1000
    n_phase_bins = 360
    n_time_bins = int(dur_ms/bin_size_ms)
    n_traj = grid_spikes[0][1]
    n_grid = grid_spikes[0][0]
    n_gra = gra_spikes[0][0]
    grid_phases = np.zeros((len(poiss_seeds), n_traj, n_time_bins*n_grid))
    gra_phases = np.zeros((len(poiss_seeds), n_traj, n_time_bins*n_gra))
    
    for i, poiss_seed in enumerate(poiss_seeds):
        #grid phases
        curr_grid_spikes = grid_spikes[i]
        curr_grid_phases = mean_phase(curr_grid_spikes, bin_size_ms, n_phase_bins, dur_ms)
        curr_grid_phases = mean_filler(curr_grid_phases)
        n_bins, n_cells, n_traj = curr_grid_phases.shape
        grid_phases[i, :, :] = np.moveaxis(curr_grid_phases, 2, 0).reshape(n_traj, n_bins*n_cells)
        #granule phases
        curr_gra_spikes = gra_spikes[i]
        curr_gra_phases = mean_phase(curr_gra_spikes, bin_size_ms, n_phase_bins, dur_ms)
        gra_phases[i, :, :] = np.moveaxis(curr_gra_phases, 2, 0).reshape(n_traj, n_bins*n_cells)
        grid_spikes[i] = copy.deepcopy(curr_grid_spikes)
        gra_spikes[i] = copy.deepcopy(curr_gra_spikes)
    return grid_phases, gra_phases




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


'will be deprecated'
# def overall_spike_ct(grid_spikes, gra_spikes, dur_ms, poiss_seeds, n_traj=2):
#     n_bin = int(dur_ms/bin_size)
#     dur_s = dur_ms/1000
#     counts_grid_1 = np.zeros((len(poiss_seeds), n_bin*n_grid))
#     counts_grid_2 = np.zeros((len(poiss_seeds), n_bin*n_grid))
#     counts_gra_1 = np.zeros((len(poiss_seeds), n_bin*n_granule))
#     counts_gra_2 = np.zeros((len(poiss_seeds), n_bin*n_granule))

#     for idx, poiss_seed in enumerate(poiss_seeds):
#         counts_grid_1[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
#         counts_grid_2[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
#         counts_gra_1[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
#         counts_gra_2[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
#     counts_grid = np.vstack((counts_grid_1, counts_grid_2))
#     counts_granule = np.vstack((counts_gra_1, counts_gra_2))
#     return counts_grid, counts_granule   






