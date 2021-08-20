#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:50:07 2021

@author: baris
"""


import numpy as np
import random
from scipy import ndimage
from scipy.stats import skewnorm
from skimage.measure import profile_line
from scipy import interpolate
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation as stg
import copy


def _grid_maker(spacing, orientation, pos_peak, arr_size, sizexy, max_rate):
    #define the params from input here, scale the resulting array for maxrate and sperate the xy for size and shift
    arr_size = arr_size
    x, y = pos_peak
    pos_peak = np.array([x,y])
    max_rate = max_rate
    lambda_spacing = spacing*(arr_size/100) #100 required for conversion, they have probably used 100*100 matrix in 
    k = (4*np.pi)/(lambda_spacing*np.sqrt(3))
    degrees = orientation
    theta = np.pi*(degrees/180)
    meterx, metery = sizexy
    arrx = meterx*arr_size # *arr_size for defining the 2d array size
    arry = metery*arr_size
    dims = np.array([arrx,arry])
    rate = np.ones(dims)
    # dist = np.ones(dims)
    #implementation of grid function
    # 3 ks for 3 cos gratings with different angles
    k1 = ((k/np.sqrt(2))*np.array((np.cos(theta+(np.pi)/12) + np.sin(theta+(np.pi)/12),
          np.cos(theta+(np.pi)/12) - np.sin(theta+(np.pi)/12)))).reshape(2,)
    k2 = ((k/np.sqrt(2))*np.array((np.cos(theta+(5*np.pi)/12) + np.sin(theta+(5*np.pi)/12),
          np.cos(theta+(5*np.pi)/12) - np.sin(theta+(5*np.pi)/12)))).reshape(2,)
    k3 = ((k/np.sqrt(2))*np.array((np.cos(theta+(9*np.pi)/12) + np.sin(theta+(9*np.pi)/12),
          np.cos(theta+(9*np.pi)/12) - np.sin(theta+(9*np.pi)/12)))).reshape(2,)
    

    #.reshape is only need when function is in the loop(shape somehow becomes (2,1) otherwise normal shape is already (2,)
    for i in range(dims[0]):
        for j in range(dims[1]):
            curr_dist = np.array([i,j]-pos_peak)
            # dist[i,j] = (np.arccos(np.cos(np.dot(k1, curr_dist)))+
            #              np.arccos(np.cos(np.dot(k2, curr_dist)))+np.arccos(np.cos(np.dot(k3, curr_dist))))/3
            rate[i,j] = (np.cos(np.dot(k1, curr_dist))+
               np.cos(np.dot(k2, curr_dist))+ np.cos(np.dot(k3, curr_dist)))/3
    rate = max_rate*2/3*(rate+1/2)   # arr is the resulting 2d grid out of 3 gratings
    # dist = (dist/np.pi)*(3/4)
    return rate

def _grid_population(n_grid, max_rate, seed, arr_size=200):
    # skewed normal distribution for grid_spc
    np.random.seed(seed)
    median_spc = 43
    spc_max = 100
    skewness = 6  #Negative values are left skewed, positive values are right skewed.
    grid_spc = skewnorm.rvs(a = skewness,loc=spc_max, size=n_grid)  #Skewnorm function
    grid_spc = grid_spc - min(grid_spc)      #Shift the set so the minimum value is equal to zero.
    grid_spc = grid_spc / max(grid_spc)      #Standadize all the vlues between 0 and 1. 
    grid_spc = grid_spc * spc_max         #Multiply the standardized values by the maximum value.
    grid_spc = grid_spc + (median_spc - np.median(grid_spc))
    
    grid_ori = np.random.randint(0, high=60, size=[n_grid,1]) #uniform dist btw 0-60 degrees
    grid_phase = np.random.randint(0, high=(arr_size-1), size=[n_grid,2]) #uniform dist grid phase
    
    # create a 3d array with grids for n_grid
    rate_grids = np.zeros((arr_size, arr_size, n_grid))#empty array
    for i in range(n_grid):
        x = grid_phase[i][0]
        y = grid_phase[i][1]
        rate = _grid_maker(grid_spc[i], grid_ori[i], [x, y], arr_size, [1,1], max_rate)
        rate_grids[:, :, i] = rate
    
    return rate_grids, grid_spc

def _draw_traj(all_grids, n_grid, par_trajs, arr_size=200, field_size_cm = 100, dur_ms=2000, speed_cm=20):
    "Trajectory (A mouse walking on a straight line) simulated" 
    all_grids = all_grids
    size2cm = int(arr_size/field_size_cm)
    dur_s = dur_ms/1000
    traj_len_cm = int(dur_s*speed_cm)
    traj_len_dp = traj_len_cm*size2cm
    par_idc_cm = par_trajs
    par_idc = par_idc_cm*size2cm-1
    n_traj = par_idc.shape[0]
    #empty arrays
    traj = np.empty((n_grid,traj_len_dp))
    trajs = np.empty((n_grid,traj_len_dp,n_traj))
    
    #draw the trajectories
    for j in range(n_traj):
        idc = par_idc[j]
        for i in range(n_grid):
            traj[i,:] = profile_line(all_grids[:,:,i], (idc,0), (idc,traj_len_dp-1), mode='constant')
            trajs[:,:,j] = traj
    
    return trajs


def _rate2dist(grids, spacings, max_rate):
    grid_dist = np.zeros((grids.shape[0], grids.shape[1], grids.shape[2]))
    for i in range(grids.shape[2]):
        grid = grids[:,:,i]
        spacing = spacings[i]
        trans_dist_2d = (np.arccos(((grid*3/(2*max_rate))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)
        grid_dist[:,:,i] = (trans_dist_2d/(spacing/2))/2
    return grid_dist

    
#interpolation
def _interp(arr, dur_s, def_dt_s = 0.025, new_dt_s = 0.002):
    arr_len = arr.shape[1]
    t_arr = np.linspace(0, dur_s, arr_len)
    if new_dt_s != def_dt_s: #if dt given is different than default_dt_s(0.025), then interpolate
        new_len = int(dur_s/new_dt_s)
        new_t_arr = np.linspace(0, dur_s, new_len)
        f = interpolate.interp1d(t_arr, arr, axis=1)
        interp_arr = f(new_t_arr)
    return interp_arr, new_t_arr



'Randomize grid spikes'

def _import_phase_dist(path='/home/baris/phase_coding/norm_grid_phase_dist.npz'):
    norm_n = np.load(path)['grid_norm_dist']
    dt_s = 0.001
    dur_s = 0.1
    total_spikes_bin = 25
    phase_prof = (total_spikes_bin/np.sum(norm_n))*norm_n/(dt_s)
    def_phase_asig = AnalogSignal(phase_prof,
                            units=1*pq.Hz,
                            t_start=0*pq.s,
                            t_stop=dur_s*pq.s,
                            sampling_period=dt_s*pq.s,
                            sampling_interval=dt_s*pq.s)
    return def_phase_asig
    
def _randomize_grid_spikes(arr, bin_size_ms, time_ms=2000):
    def_phase_asig = _import_phase_dist()
    randomized_grid = np.empty(0)
    n_bins = int(time_ms/bin_size_ms)
    for i in range(n_bins):
        curr_ct = ((bin_size_ms*(i) < arr) & (arr < bin_size_ms*(i+1))).sum()
        curr_train = stg.inhomogeneous_poisson_process(def_phase_asig, refractory_period=0.001*pq.s, as_array=True)*1000
        rand_spikes = np.array(random.sample(list(curr_train), k=curr_ct))
        spikes_ms = np.ones(rand_spikes.shape[0])*(bin_size_ms*i)+rand_spikes
        randomized_grid = np.append(randomized_grid, np.array(spikes_ms))
    return np.sort(randomized_grid)



'Generate shuffled/nonshuffled spikes with inh poisson funciton'

def _inhom_poiss(arr, n_traj, dur_s, shuffle=False, diff_seed=True, poiss_seed=0, dt_s=0.025, dur_ms=2000):
    #length of the trajectory that mouse went
    np.random.seed(poiss_seed)
    n_cells = arr.shape[0]
    spi_arr = np.zeros((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            if diff_seed==True:
                np.random.seed(poiss_seed+grid_idc+(5*i))
            elif diff_seed==False:
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
                curr_train = _randomize_grid_spikes(curr_train, 100, time_ms=dur_ms)
            spi_arr[grid_idc, i] = np.array(curr_train) #time conv to ms
    return spi_arr


def _overall_with_dir(dist_trajs, rate_trajs, shift_deg, T, n_traj, speed_cm, dur_s=2):
    #infer the direction out of rate of change in the location
    direction = np.diff(dist_trajs, axis=1)
    #last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:,-1:,:]), axis=1)
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    direction = -direction
    traj_dist_dir = dist_trajs*direction
    traj_dist_dir = ndimage.gaussian_filter1d(traj_dist_dir, sigma=1, axis=1)
    traj_dist_dir, dist_t_arr = _interp(traj_dist_dir, dur_s)
    factor = shift_deg/360 #change the phase shift from 360 degrees to 240 degrees
    one_theta_phase = (2*np.pi*(dist_t_arr%T)/T)%(2*np.pi)
    theta_phase = np.repeat(one_theta_phase[np.newaxis,:], 200, axis=0)
    theta_phase =  np.repeat(theta_phase[:,:,np.newaxis], n_traj, axis=2)
    firing_phase_dir = 2*np.pi*(traj_dist_dir+0.5)*factor
    phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-theta_phase))
    scaling_factor = 5
    constant_mv = 0.16
    overall_dir = phase_code_dir*rate_trajs*speed_cm*constant_mv*scaling_factor
    return overall_dir


def grid_simulate(trajs, dur_ms, grid_seed, poiss_seeds, tune, shuffle, n_grid=200, speed_cm=20, max_rate=1, arr_size=200, f=10, shift_deg=180, dt_s = 0.002):
    dur_s = dur_ms/1000
    T = 1/f
    n_traj = trajs.shape[0]
    grid_spikes = np.zeros(len(poiss_seeds), dtype = np.ndarray)
    
    grids, spacings = _grid_population(n_grid, max_rate, seed=grid_seed, arr_size=arr_size)
    grid_dist = _rate2dist(grids, spacings, max_rate)
    dist_trajs = _draw_traj(grid_dist, n_grid, trajs, dur_ms=dur_ms)
    rate_trajs = _draw_traj(grids, n_grid, trajs, dur_ms=dur_ms)
    rate_trajs, rate_t_arr = _interp(rate_trajs, dur_s)
    overall_dir = _overall_with_dir(dist_trajs, rate_trajs, shift_deg, T, n_traj, speed_cm)
      
    for idx, poiss_seed in enumerate(poiss_seeds):
        curr_grid_spikes = _inhom_poiss(overall_dir, n_traj, dur_s, shuffle=shuffle, dt_s=dt_s, poiss_seed=poiss_seed)
        grid_spikes[idx] = copy.deepcopy(curr_grid_spikes)
    return grid_spikes


# grid_spikes = grid_simulate(np.array([75, 74.5, 74, 70]), 2000, 400, np.arange(420,425,1), 'full', True)
