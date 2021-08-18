#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:56:04 2020

@author: bariskuru
"""

import numpy as np
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
from scipy import interpolate

def inhom_poiss(arr, n_traj, dt_s=0.0001, speed_cm = 20, field_size_cm = 100, seed=10000):
    #length of the trajectory that mouse went
    np.random.seed(seed)
    dt_ms = dt_s*1000
    t_sec = field_size_cm/speed_cm
    arr_len = arr.shape[1]
    t_arr = np.linspace(0, t_sec, arr_len)
    default_dt_s = t_sec / arr_len
    new_len = int(t_sec/dt_s)
    new_t_arr = np.linspace(0, t_sec, new_len)
        # arr=signal.resample(arr, new_len, axis=1)
    if dt_s != default_dt_s: #if dt given is different than default_dt(0.025), then interpolate
        new_len = int(t_sec/dt_s)
        new_t_arr = np.linspace(0, t_sec, new_len)
        f = interpolate.interp1d(t_arr, arr, axis=1)
        arr = f(new_t_arr)
    n_cells = arr.shape[0]
    spi_arr = np.empty((n_cells, n_traj), dtype = np.ndarray)
    #go through each rate profile or each cell for variable trajectory
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            np.random.seed(seed+grid_idc)
            rate_profile = arr[grid_idc,:,i]
            #produce analig signal out of rate profiles
            asig = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=t_sec*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)
            #generate the spike train out of analog signal
            curr_train = stg.inhomogeneous_poisson_process(asig)
            spi_arr[grid_idc, i] = np.array(curr_train.times*1000) #time conv to ms
    return spi_arr


def time_stamps_to_signal(time_stamps, dt_signal, t_start, t_stop):
    """Convert an array of timestamps to a signal where 0 is absence and 1 is
    presence of spikes
    """
    # Construct a zero array with size corresponding to desired output signal
    sig = np.zeros((np.shape(time_stamps)[0],int((t_stop-t_start)/dt_signal)))
    # Find the indices where spikes occured according to time_stamps
    time_idc = []
    for x in time_stamps:
        curr_idc = []
        curr_idc.append((x-t_start)/ dt_signal)
        time_idc.append(curr_idc)
    
    # Set the spike indices to 1
    for sig_idx, idc in enumerate(time_idc):
        sig[sig_idx,np.array(idc,dtype=np.int)] = 1

    return sig

