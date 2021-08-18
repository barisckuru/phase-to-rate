#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:31:17 2020

@author: bariskuru
"""

import numpy as np
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
from scipy import signal, interpolate

loaded_traj = np.load('/Users/bariskuru/Desktop/MasterThesis/trajectories/trajectories_100_cells_200_arrsize.npz', allow_pickle=True)
dev_deg=loaded_traj['dev_deg']


def inhom_poiss_2(arr, dt=0.025, speed_cm = 20, field_size_cm = 100):
    #length of the trajectory that mouse went
    t_sec = field_size_cm/speed_cm
    arr_len = arr.shape[1]
    t_arr = np.linspace(0, t_sec, arr_len)
    default_dt = t_sec / arr_len
    new_len = int(t_sec/dt)
    new_t_arr = np.linspace(0, t_sec, new_len)
        # arr=signal.resample(arr, new_len, axis=1)
    if dt != default_dt:
        new_len = int(t_sec/dt)
        new_t_arr = np.linspace(0, t_sec, new_len)
        f = interpolate.interp1d(t_arr, arr, axis=1)
        arr = f(new_t_arr)
    n_traj = dev_deg.shape[0] #8
    n_cells = arr.shape[0]
    spi_arr = np.empty((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(dev_deg.shape[0]):
            rate_profile = arr[grid_idc,:,i]
            asig = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=t_sec*pq.s,
                                    sampling_period=dt*pq.s,
                                    sampling_interval=dt*pq.s)
            curr_train = stg.inhomogeneous_poisson_process(asig)
            spi_arr[grid_idc, i] = np.around(curr_train.times*1000, decimals=1)
    return spi_arr


