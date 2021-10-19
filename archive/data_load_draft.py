#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:37:58 2021

@author: baris
"""


grid_seed = 1
output_path = (
    f"{grid_seed}_{trajectories}_{poisson_seeds[0]}-{poisson_seeds[-1]}_{dur_ms}"
)
storage_old = shelve.open(
    "/home/baris/results/poisson_seeds_seperate/grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_"
    + output_path
    + "_"
    + shuffling
    + "_"
    + network_type,
    writeback=True,
)


output_name = f"{grid_seed}_{dur_ms}"
storage_path = "/home/baris/results/grid-seed_duration_shuffling_tuning_"
storage_name = storage_path + output_name + "_" + shuffling + "_" + network_type
storage = shelve.open(storage_name, writeback=True)
traj_key = str(traj)
storage[traj_key] = {}
storage[traj_key]["grid_spikes"] = copy.deepcopy(storage_old["grid_spikes"][traj])
storage[traj_key]["granule_spikes"] = copy.deepcopy(storage_old["granule_spikes"][traj])
storage[traj_key]["parameters"] = copy.deepcopy(storage_old["parameters"])

storage.close()
storage_old.close()
