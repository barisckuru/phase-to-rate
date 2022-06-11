#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:27:20 2022

@author: baris
"""

import os 
import shelve

#variables to change
shuffling = "non-shuffled"
network_type = "no-feedback" # netowrk type for the filename
path = "/home/baris/results/olli_data/"
folder = "no-feedback_adjusted" # folder name

#variables not to change
trajectories = 75
grid_seed = 11
poisson_seeds = range(100,120)
dur_ms = 2000

save_dir = os.path.join(path, f'{folder}')
file_name_signature = "grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning"
file_name_data = f"{grid_seed}_[{trajectories}]_{poisson_seeds[0]}-{poisson_seeds[-1]}_{dur_ms}_{shuffling}_{network_type}"
file_name = f"{file_name_signature}_{file_name_data}"
shelve_loc = os.path.join(save_dir, file_name)

with shelve.open(shelve_loc) as storage:
    print(list(storage.keys()))
    grid_spikes = storage['grid_spikes'][trajectories][poisson_seeds[0]]
    granule_spikes = storage['granule_spikes'][trajectories][poisson_seeds[0]]
    parameters = storage['parameters']
