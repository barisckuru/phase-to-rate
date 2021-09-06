#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:19:38 2020

@author: baris
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from scipy import interpolate
from skimage.measure import profile_line
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
import os
import copy
from scipy import ndimage

from neuron import h, gui  # gui necessary for some parameters to h namespace
import net_tunedrev
import scipy.stats as stats
import random

savedir = os.getcwd()
n_grid = 200 
n_granule = 2000
max_rate = 1
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
bin_size = 100
speed_cm = 20
field_size_cm = 100
def_dt_s = 0.025
new_dt_s = 0.002
dt_s = new_dt_s


plt.close('all')



# def rand_spike(n_spikes, norm_n=norm_n,):
#     curr_train = stg.inhomogeneous_poisson_process(asig, as_array=True, refractory_period=(dt_s*pq.s))
#     spikes_ms = random.choices(curr_train*1000, k=n_spikes)
#     return spikes_ms


norm_n = np.load('/home/baris/results/grid_mixed_input/norm_grid_phase_dist.npz')['grid_norm_dist']
n_spikes = 5
dt_s = 0.001
dur_s = 0.1
total_spikes_bin = 25
phase_prof = (total_spikes_bin/np.sum(norm_n))*norm_n/(dt_s)
asig = AnalogSignal(phase_prof,
                        units=1*pq.Hz,
                        t_start=0*pq.s,
                        t_stop=dur_s*pq.s,
                        sampling_period=dt_s*pq.s,
                        sampling_interval=dt_s*pq.s)

# def randomize_grid_spikes(arr, bin_size_ms, asig=asig, dt_ms=25, time_ms=2000):
#     randomized_grid = np.empty((arr.shape[0], arr.shape[1]), dtype=np.ndarray)
#     n_bins = int(time_ms/bin_size_ms)
#     for i in range(n_bins):
#         for index, value in np.ndenumerate(arr):
#             # print(index)
#             curr_ct = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
#             curr_train = stg.inhomogeneous_poisson_process(asig, as_array=True)
#             spikes_ms = random.choices(curr_train, k=curr_ct)
#             randomized_grid[index] = np.array(spikes_ms)
#     return randomized_grid




def rate2dist(grids, spacings, max_rate):
    grid_dist = np.zeros((grids.shape[0], grids.shape[1], grids.shape[2]))
    for i in range(grids.shape[2]):
        grid = grids[:,:,i]
        spacing = spacings[i]
        trans_dist_2d = (np.arccos(((grid*3/(2*max_rate))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)
        grid_dist[:,:,i] = (trans_dist_2d/(spacing/2))/2
        # grid_dist[:,:,i][grid_dist[:,:,i]>0.5] = 0.5
    return grid_dist

def _direction(dist_trajs, n_traj, n_grid):
    #infer the direction out of rate of change in the location
    direction = np.diff(dist_trajs, axis=1)
    #last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:,-1:,:]), axis=1)
    for trj in range(n_traj):
        for cell in range(n_grid):
            cur_dir = direction[cell, :, trj]
            if max(cur_dir) != 0:
                cur_dir = -cur_dir*(1/max(cur_dir))
            cur_dir[np.logical_and(cur_dir < 0.1, cur_dir > -0.1)] = 1
            cur_dir = ndimage.gaussian_filter(cur_dir, sigma=2)
            direction[cell, :, trj] = cur_dir
    return direction
    
    

#interpolation
def interp(arr, dur_s, def_dt_s, new_dt_s):
    arr_len = arr.shape[1]
    t_arr = np.linspace(0, dur_s, arr_len)
    if new_dt_s != def_dt_s: #if dt given is different than default_dt_s(0.025), then interpolate
        new_len = int(dur_s/new_dt_s)
        new_t_arr = np.linspace(0, dur_s, new_len)
        f = interpolate.interp1d(t_arr, arr, axis=1)
        interp_arr = f(new_t_arr)
    return interp_arr, new_t_arr


def randomize_grid_spikes(arr, bin_size_ms, asig=asig, time_ms=2000):
    randomized_grid = np.empty(0)
    n_bins = int(time_ms/bin_size_ms)
    for i in range(n_bins):
        curr_ct = ((bin_size_ms*(i) < arr) & (arr < bin_size_ms*(i+1))).sum()
        curr_train = stg.inhomogeneous_poisson_process(asig, refractory_period=0.001*pq.s, as_array=True)*1000
        rand_spikes = np.array(random.sample(list(curr_train), k=curr_ct))
        spikes_ms = np.ones(rand_spikes.shape[0])*(bin_size_ms*i)+rand_spikes
        randomized_grid = np.append(randomized_grid, np.array(spikes_ms))
    return np.sort(randomized_grid)


def inhom_poiss(arr, n_traj, dur_s, shuffle=False, poiss_seed=0, dt_s=0.025, dur_ms=2000):
    #length of the trajectory that mouse went
    np.random.seed(poiss_seed)
    n_cells = arr.shape[0]
    spi_arr = np.zeros((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            np.random.seed(poiss_seed+grid_idc)
            rate_profile = arr[grid_idc,:,i]
            asig = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=dur_s*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)
            curr_train = stg.inhomogeneous_poisson_process(asig, refractory_period=0.001*pq.s, as_array=True)*1000
            if shuffle==True:
                # print(curr_train.shape)
                curr_train = randomize_grid_spikes(curr_train, 100, time_ms=dur_ms)
                # print(curr_train)
            spi_arr[grid_idc, i] = np.array(curr_train) #time conv to ms
    return spi_arr

def inhom_poiss_d(arr, n_traj, dur_s, shuffle=False, poiss_seed=0, dt_s=0.025, dur_ms=2000):
    #length of the trajectory that mouse went
    # np.random.seed(poiss_seed)
    n_cells = arr.shape[0]
    spi_arr = np.zeros((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            np.random.seed(poiss_seed+grid_idc+(5*i))
            rate_profile = arr[grid_idc,:,i]
            asig = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=dur_s*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)
            curr_train = stg.inhomogeneous_poisson_process(asig, refractory_period=0.001*pq.s, as_array=True)*1000
            if shuffle==True:
                # print(curr_train.shape)
                curr_train = randomize_grid_spikes(curr_train, 100, time_ms=dur_ms)
                # print(curr_train)
            spi_arr[grid_idc, i] = np.array(curr_train) #time conv to ms
    return spi_arr



def inhom_poiss_diff(arr, n_traj, dur_s, poiss_seed=0, dt_s=0.025):
    #length of the trajectory that mouse went
    # np.random.seed(poiss_seed)
    n_cells = arr.shape[0]
    spi_arr = np.zeros((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            np.random.seed(poiss_seed+grid_idc+(5*i))
            rate_profile = arr[grid_idc,:,i]
            asig = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=dur_s*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)
            curr_train = stg.inhomogeneous_poisson_process(asig)
            spi_arr[grid_idc, i] = np.array(curr_train.times*1000) #time conv to ms
    return spi_arr



def mean_phase(spikes, bin_size_ms, n_phase_bins, dur_ms):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells =spikes.shape[0] 
    n_traj = spikes.shape[1]
    rad = n_phase_bins/360*2*np.pi
    # spikes=np.array(spikes)

    # phases = rad*np.ones((n_bins, n_cells, n_traj))
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



def rate_only_spike_ct(trajs_pf, dur_ms, poiss_seeds, n_traj=2):
    n_bin = int(dur_ms/bin_size)
    dur_s = dur_ms/1000
    grid_spikes = []
    granule_spikes = []
    counts_grid_1 = np.zeros((len(poiss_seeds), n_bin*n_grid))
    counts_grid_2 = np.zeros((len(poiss_seeds), n_bin*n_grid))
    counts_gra_1 = np.zeros((len(poiss_seeds), n_bin*n_granule))
    counts_gra_2 = np.zeros((len(poiss_seeds), n_bin*n_granule))

    for idx, poiss_seed in enumerate(poiss_seeds):
        curr_spikes = inhom_poiss(trajs_pf, n_traj, dur_s, poiss_seed=poiss_seed, dt_s=dt_s)
        #granule output from pydentate
        granule_out = pyDentate(curr_spikes, poiss_seed, n_traj, dur_ms, pp_weight)[0]
        grid_spikes.append(curr_spikes)
        granule_spikes.append(granule_out)
        counts_grid_1[idx, :] = binned_ct(curr_spikes, bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_grid_2[idx,:] = binned_ct(curr_spikes, bin_size, time_ms=dur_ms)[:,:,1].flatten()
        counts_gra_1[idx, :] = binned_ct(granule_out, bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_gra_2[idx,:] = binned_ct(granule_out, bin_size, time_ms=dur_ms)[:,:,1].flatten()
    counts_grid = np.vstack((counts_grid_1, counts_grid_2))
    counts_granule = np.vstack((counts_gra_1, counts_gra_2))
    return counts_grid, counts_granule, grid_spikes, granule_spikes

def overall_spike_ct(grid_spikes, gra_spikes, dur_ms, poiss_seeds, n_traj=2):
    n_bin = int(dur_ms/bin_size)
    dur_s = dur_ms/1000
    counts_grid_1 = np.zeros((len(poiss_seeds), n_bin*n_grid))
    counts_grid_2 = np.zeros((len(poiss_seeds), n_bin*n_grid))
    counts_gra_1 = np.zeros((len(poiss_seeds), n_bin*n_granule))
    counts_gra_2 = np.zeros((len(poiss_seeds), n_bin*n_granule))

    for idx, poiss_seed in enumerate(poiss_seeds):
        counts_grid_1[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_grid_2[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
        counts_gra_1[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_gra_2[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
    counts_grid = np.vstack((counts_grid_1, counts_grid_2))
    counts_granule = np.vstack((counts_gra_1, counts_gra_2))
    return counts_grid, counts_granule   

def phase_code(trajs, dur_ms, grid_seed, poiss_seeds, pp_weight, f=10, shift_deg=180):
    dur_s = dur_ms/1000
    T = 1/f
    bin_size_ms = T*1000
    n_phase_bins = 360
    n_time_bins = int(dur_ms/bin_size_ms)
    bins_size_deg = 1
    n_traj = trajs.shape[0]
    dt_s = new_dt_s
    
    grids, spacings = grid_population(n_grid, max_rate, seed=grid_seed, arr_size=arr_size)
    grid_dist = rate2dist(grids, spacings, max_rate)
    # grid_dist = rate2dist(grids, spacings, max_rate)
    dist_trajs = draw_traj(grid_dist, n_grid, trajs, dur_ms=dur_ms)
    # dist_trajs, dist_t_arr = interp(dist_trajs, dur_s, def_dt_s, new_dt_s)
    rate_trajs = draw_traj(grids, n_grid, trajs, dur_ms=dur_ms)
    rate_trajs, rate_t_arr = interp(rate_trajs, dur_s, def_dt_s, new_dt_s)
    
    
    # #infer the direction out of rate of change in the location
    direction = np.diff(dist_trajs, axis=1)
    # #last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:,-1:,:]), axis=1)
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    direction = -direction
    # direction[np.logical_and(direction < 0.1, direction > -0.1)] = 1
    # direction = ndimage.gaussian_filter(direction, sigma=2)

    # direction = _direction(dist_trajs, n_traj, n_grid)
    traj_dist_dir = dist_trajs*direction
    traj_dist_dir = ndimage.gaussian_filter1d(traj_dist_dir, sigma=1, axis=1)
    traj_dist_dir, dist_t_arr = interp(traj_dist_dir, dur_s, def_dt_s, new_dt_s)
    factor = shift_deg/360 #change the phase shift from 360 degrees to 240 degrees
    one_theta_phase = (2*np.pi*(dist_t_arr%T)/T)%(2*np.pi)
    theta_phase = np.repeat(one_theta_phase[np.newaxis,:], 200, axis=0)
    theta_phase =  np.repeat(theta_phase[:,:,np.newaxis], n_traj, axis=2)
    firing_phase_dir = 2*np.pi*(traj_dist_dir+0.5)*factor
    phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-theta_phase))
    scaling_factor = 5 
    constant_mv = 0.16
    overall_dir = phase_code_dir*rate_trajs*speed_cm*constant_mv*scaling_factor
    
    grid_phases_1 = np.zeros((len(poiss_seeds), n_time_bins*n_grid))
    grid_phases_2 = np.zeros((len(poiss_seeds), n_time_bins*n_grid))
    gra_phases_1 = np.zeros((len(poiss_seeds), n_time_bins*n_granule))
    gra_phases_2 = np.zeros((len(poiss_seeds), n_time_bins*n_granule))
    ori_grid_spikes = np.zeros(len(poiss_seeds), dtype = np.ndarray)
    grid_spikes = np.zeros(len(poiss_seeds), dtype = np.ndarray)
    gra_spikes = np.zeros(len(poiss_seeds), dtype = np.ndarray)
    for idx, poiss_seed in enumerate(poiss_seeds):
        shuff_curr_grid_spikes = inhom_poiss(overall_dir, n_traj, dur_s, shuffle=True, dt_s=dt_s, poiss_seed=poiss_seed)
        ori_curr_grid_spikes = inhom_poiss(overall_dir, n_traj, dur_s, shuffle=False, dt_s=dt_s, poiss_seed=poiss_seed)
        # print(curr_grid_spikes.shape)
        curr_gra_spikes = pyDentate(ori_curr_grid_spikes, trajs, grid_seed, poiss_seed, n_traj, dur_ms, pp_weight)[0]
        #grid phases
        curr_grid_phases = mean_phase(curr_grid_spikes, bin_size_ms, n_phase_bins, dur_ms)
        curr_grid_phases = mean_filler(curr_grid_phases)
        grid_phases_1[idx, :] = curr_grid_phases[:,:,0].flatten()
        grid_phases_2[idx, :] = curr_grid_phases[:,:,1].flatten()

        #granule phases
        curr_gra_phases = mean_phase(curr_gra_spikes, bin_size_ms, n_phase_bins, dur_ms)
        # curr_gra_phases = mean_filler(curr_gra_phases)
        #this separation is necessary bcs 
        #when there are multiple seeds, same types of inputs are grouped together, 
        #then all are stacked in one array at the end
        gra_phases_1[idx, :] = curr_gra_phases[:,:,0].flatten()
        gra_phases_2[idx, :] = curr_gra_phases[:,:,1].flatten()
        grid_spikes[idx] = copy.deepcopy(shuff_curr_grid_spikes)
        ori_grid_spikes[idx] = copy.deepcopy(ori_curr_grid_spikes)
        gra_spikes[idx] = copy.deepcopy(curr_gra_spikes)
    grid_phases = np.vstack((grid_phases_1, grid_phases_2))
    gra_phases = np.vstack((gra_phases_1, gra_phases_2))
    return grid_phases, gra_phases, grid_spikes, ori_grid_spikes, gra_spikes, rate_trajs, dt_s, theta_phase, phase_code_dir, overall_dir, traj_dist_dir, dist_trajs, direction, spacings, grids


def phase_code_diff_poiss(trajs, dur_ms, grid_seed, poiss_seeds, pp_weight, tune, shuffle, f=10, shift_deg=180):
    dur_s = dur_ms/1000
    T = 1/f
    bin_size_ms = T*1000
    n_phase_bins = 360
    n_time_bins = int(dur_ms/bin_size_ms)
    bins_size_deg = 1
    n_traj = trajs.shape[0]
    dt_s = new_dt_s
    
    grids, spacings = grid_population(n_grid, max_rate, seed=grid_seed, arr_size=arr_size)
    grid_dist = rate2dist(grids, spacings, max_rate)
    # grid_dist = rate2dist(grids, spacings, max_rate)
    dist_trajs = draw_traj(grid_dist, n_grid, trajs, dur_ms=dur_ms)
    # dist_trajs, dist_t_arr = interp(dist_trajs, dur_s, def_dt_s, new_dt_s)
    rate_trajs = draw_traj(grids, n_grid, trajs, dur_ms=dur_ms)
    rate_trajs, rate_t_arr = interp(rate_trajs, dur_s, def_dt_s, new_dt_s)
    
    
    # #infer the direction out of rate of change in the location
    direction = np.diff(dist_trajs, axis=1)
    # #last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:,-1:,:]), axis=1)
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    direction = -direction
    # direction[np.logical_and(direction < 0.1, direction > -0.1)] = 1
    # direction = ndimage.gaussian_filter(direction, sigma=2)

    # direction = _direction(dist_trajs, n_traj, n_grid)
    traj_dist_dir = dist_trajs*direction
    traj_dist_dir = ndimage.gaussian_filter1d(traj_dist_dir, sigma=1, axis=1)
    traj_dist_dir, dist_t_arr = interp(traj_dist_dir, dur_s, def_dt_s, new_dt_s)
    factor = shift_deg/360 #change the phase shift from 360 degrees to 240 degrees
    one_theta_phase = (2*np.pi*(dist_t_arr%T)/T)%(2*np.pi)
    theta_phase = np.repeat(one_theta_phase[np.newaxis,:], 200, axis=0)
    theta_phase =  np.repeat(theta_phase[:,:,np.newaxis], n_traj, axis=2)
    firing_phase_dir = 2*np.pi*(traj_dist_dir+0.5)*factor
    phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-theta_phase))
    scaling_factor = 5
    constant_mv = 0.16
    overall_dir = phase_code_dir*rate_trajs*speed_cm*constant_mv*scaling_factor
    
    grid_phases_1 = np.zeros((len(poiss_seeds), n_time_bins*n_grid))
    grid_phases_2 = np.zeros((len(poiss_seeds), n_time_bins*n_grid))
    gra_phases_1 = np.zeros((len(poiss_seeds), n_time_bins*n_granule))
    gra_phases_2 = np.zeros((len(poiss_seeds), n_time_bins*n_granule))
    grid_spikes = np.zeros(len(poiss_seeds), dtype = np.ndarray)
    gra_spikes = np.zeros(len(poiss_seeds), dtype = np.ndarray)
    for idx, poiss_seed in enumerate(poiss_seeds):
        curr_grid_spikes = inhom_poiss_d(overall_dir, n_traj, dur_s, shuffle=shuffle, dt_s=dt_s, poiss_seed=poiss_seed)
        curr_gra_spikes = pyDentate(curr_grid_spikes, trajs, grid_seed, poiss_seed, n_traj, dur_ms, pp_weight, tune)[0]
        #grid phases
        curr_grid_phases = mean_phase(curr_grid_spikes, bin_size_ms, n_phase_bins, dur_ms)
        curr_grid_phases = mean_filler(curr_grid_phases)
        grid_phases_1[idx, :] = curr_grid_phases[:,:,0].flatten()
        grid_phases_2[idx, :] = curr_grid_phases[:,:,1].flatten()

        #granule phases
        curr_gra_phases = mean_phase(curr_gra_spikes, bin_size_ms, n_phase_bins, dur_ms)
        # curr_gra_phases = mean_filler(curr_gra_phases)
        #this separation is necessary bcs 
        #when there are multiple seeds, same types of inputs are grouped together, 
        #then all are stacked in one array at the end
        gra_phases_1[idx, :] = curr_gra_phases[:,:,0].flatten()
        gra_phases_2[idx, :] = curr_gra_phases[:,:,1].flatten()
        grid_spikes[idx] = copy.deepcopy(curr_grid_spikes)
        gra_spikes[idx] = copy.deepcopy(curr_gra_spikes)
    grid_phases = np.vstack((grid_phases_1, grid_phases_2))
    gra_phases = np.vstack((gra_phases_1, gra_phases_2))
    return grid_phases, gra_phases, grid_spikes, gra_spikes, rate_trajs, dt_s, theta_phase, phase_code_dir, overall_dir, traj_dist_dir, dist_trajs, direction, spacings, grids



def pyDentate(input_grid_out, trajs, grid_seed, poiss_seed, n_traj, dur_ms, pp_weight, tune):
    savedir = os.getcwd()
    input_scale = 1000
    dent_seed = grid_seed+150 #dent_seed for network generation & simulation
    
    #number of cells
    n_grid = 200 
    n_granule = 2000
    n_mossy = 60
    n_basket = 24
    n_hipp = 24
    
    np.random.seed(dent_seed) # dent_seed for connections in the network
    
    # Randomly choose target cells for the GridCell lines
    gauss_gc = stats.norm(loc=1000, scale=input_scale)
    gauss_bc = stats.norm(loc=12, scale=(input_scale/float(n_granule))*n_basket)
    pdf_gc = gauss_gc.pdf(np.arange(n_granule))
    pdf_gc = pdf_gc/pdf_gc.sum()
    pdf_bc = gauss_bc.pdf(np.arange(n_basket))
    pdf_bc = pdf_bc/pdf_bc.sum()
    GC_indices = np.arange(n_granule)
    start_idc = np.random.randint(0, n_granule-1, size=n_grid)
    
    PP_to_GCs = []
    for x in start_idc:
        curr_idc = np.concatenate((GC_indices[x:n_granule], GC_indices[0:x]))
        PP_to_GCs.append(np.random.choice(curr_idc, size=100, replace=False,
                                          p=pdf_gc))
    
    PP_to_GCs = np.array(PP_to_GCs)
    
    BC_indices = np.arange(n_basket)
    start_idc = np.array(((start_idc/float(n_granule))*24), dtype=int)
    
    PP_to_BCs = []
    for x in start_idc:
        curr_idc = np.concatenate((BC_indices[x:24], BC_indices[0:x]))
        PP_to_BCs.append(np.random.choice(curr_idc, size=1, replace=False,
                                          p=pdf_bc))
    PP_to_BCs = np.array(PP_to_BCs)
    
    # generate temporal patterns out of grid cell act profiles as an input for pyDentate
    # input_grid_out = inhom_poiss(par_traj, n_traj, dt_s=0.0001, seed=poiss_seed)
    
    np.random.seed(dent_seed) #dent_seed again for network generation & simulation
    
    granule_output = np.zeros((n_granule, n_traj), dtype = np.ndarray)
    mossy_output = np.zeros((n_mossy, n_traj), dtype = np.ndarray)
    basket_output = np.zeros((n_basket, n_traj), dtype = np.ndarray)
    hipp_output = np.zeros((n_hipp, n_traj), dtype = np.ndarray)
    
    for trj in range(n_traj):
        nw = net_tunedrev.TunedNetwork(dent_seed, input_grid_out[:,trj], PP_to_GCs, PP_to_BCs, pp_weight=pp_weight)
        # Attach voltage recordings to all cells
        nw.populations[0].voltage_recording(range(n_granule))
        nw.populations[1].voltage_recording(range(n_mossy))
        nw.populations[2].voltage_recording(range(n_basket))
        nw.populations[3].voltage_recording(range(n_hipp))
        # Run the model
        
        """Initialization for -2000 to -100"""
        h.cvode.active(0)
        dt = 0.1
        h.steps_per_ms = 1.0/dt
        h.finitialize(-60)
        h.t = -2000
        h.secondorder = 0
        h.dt = 10
        while h.t < -100:
            h.fadvance()
            
        h.secondorder = 2
        h.t = 0
        h.dt = 0.1
        
        """Setup run control for -100 to 1500"""
        h.frecord_init()  # Necessary after changing t to restart the vectors
        while h.t < dur_ms:
            h.fadvance()
        print("Done Running")
        
        granule_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[0].ap_counters], dtype=object))
        # mossy_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[1].ap_counters], dtype=object))
        # basket_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[2].ap_counters], dtype=object))
        # hipp_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[3].ap_counters], dtype=object))
        
        fig = nw.plot_aps(time=dur_ms)
        tuned_fig_file_name = (str(nw) + "spike_plot_rate_"+tune+"_n_phase_gra_out_for_perceptron_seed3_dur_weight_"+
                    str(dent_seed)+ '_' + str(dur_ms) +'_' +str(pp_weight)+'_'+ str(trj))
        nw.save_ap_fig(fig, savedir, tuned_fig_file_name)
    
    path = '/home/baris/results/pyDentate/'
    fname = 'pyDentate_'+tune+'_out_traj_'+str(trajs[0])+'-'+str(trajs[1])+'_dur_'+str(dur_ms)+'_weight_'+str(pp_weight)+'_seed1_'+str(grid_seed)+'_seed2_'+str(poiss_seed)
    np.savez(path+fname, 
    granule_output = granule_output,
    mossy_output = mossy_output,
    basket_output = basket_output,
    hipp_output = hipp_output, allow_pickle=True)

    # tuned_save_file_name = (str(nw) + "data_rate_n_phase_gra_out_for_perceptron_seed3_dur_" +
    #                 '-'+str(dent_seed)+ '_' +
    #                 str(dur_ms))
    # nw.shelve_network(savedir, tuned_save_file_name)


    return granule_output, mossy_output, basket_output, hipp_output
