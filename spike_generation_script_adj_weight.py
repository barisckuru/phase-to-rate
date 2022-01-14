#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 16:49:14 2021

@author: baris
"""


from pydentate import neuron_tools
from grid_model import grid_simulate
import shelve
import copy
import time
import numpy as np
from pydentate_integrate import granule_simulate
import os

start = time.time()

"""Setup"""
neuron_tools.load_compiled_mechanisms(
    path="/home/baris/pydentate/mechs/x86_64/libnrnmech.so"
)


"""Parameters"""
# grid_seeds = [1,2,3,4,5,6,7,8,9,10]
# grid_seeds = [1,2]
# grid_seeds = [3,4]
# grid_seeds = [5,6]
# grid_seeds = [7,8]
grid_seeds = [9,10]

# shuffling = "non-shuffled"
shuffling = "shuffled"

pp_weight = 0.00059
# network_type = "full"
# network_type = "no-feedback" 
network_type = "disinhibited"
# network_type = "no-feedforward"

# grid_seeds = [11]

trajectories, p = [75], 100  # In cm
# trajectories, p = [74.5], 200
# trajectories, p = [74], 300
# trajectories, p = [73.5], 400
# trajectories, p = [73], 500
# trajectories, p = [72.5], 600
# trajectories, p = [72], 700
# trajectories, p = [71], 800
# trajectories, p = [70], 900
# trajectories, p = [69], 1000
# trajectories, p = [68], 1100
# trajectories, p = [67], 1200
# trajectories, p = [66], 1300
# trajectories, p = [65], 1400
# trajectories, p = [60], 1500
# trajectories, p = [30], 1600
# trajectories, p = [15], 1700

poisson_seeds = np.arange(p, p + 20, 1)
# poisson_seeds = np.arange(p, p + 1, 1)
poisson_seeds = list(poisson_seeds)


# weights_full = [0.00105, 0.00145, 0.0020, 0.0022]
# weights_noff = [0.00083, 0.0015, 0.0021, 0.00235]
# weights_nofb = [0.00074, 0.00085, 0.00095, 0.00105]
# weights_disinhibited = [0.00065, 0.00073, 0.00082, 0.00086]



# poisson_reseeding = True  # Controls seeding between trajectories
speed_cm = 20
dur_ms = 2000
rate_scale = 5
n_grid = 200


print(network_type)
print(pp_weight)
parameters = {}
parameters = {
    "dur_ms": dur_ms,
    "pp_weight": pp_weight,
    "speed": speed_cm,
    "n_grid": n_grid,
    "rate_scale": rate_scale,
    "poisson_seeds": poisson_seeds,
}

print("grid", grid_seeds, "poiss", poisson_seeds,
      "traj", trajectories, "dur", dur_ms)



# for grid_seed in grid_seeds:
#     save_dir = (f'/home/baris/results/different_weight/{network_type}/seperate/seed_'
#                 + str(grid_seed) + '/')
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)
#     save_dir = save_dir+str(pp_weight)+'/'
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)
#     output_path = (
#         f"{grid_seed}_{trajectories}_{poisson_seeds[0]}-{poisson_seeds[-1]}_{dur_ms}"
#     )


for grid_seed in grid_seeds:
    save_dir = (f'/home/baris/results/different_weight/{network_type}/seperate/seed_'
                    + str(grid_seed) + '/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_dir = save_dir+str(pp_weight)+'/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    output_path = (
        f"{grid_seed}_{trajectories}_{poisson_seeds[0]}-{poisson_seeds[-1]}_{dur_ms}"
    )

    storage = shelve.open(
        save_dir
        + 'grid-seed_trajectory_poisson-seeds_'
        +'duration_shuffling_tuning_'
        + output_path
        + "_"
        + shuffling
        + "_"
        + network_type,
        writeback=True,
    )

    grid_spikes, _ = grid_simulate(
        trajs=trajectories,
        dur_ms=dur_ms,
        grid_seed=grid_seed,
        poiss_seeds=poisson_seeds,
        shuffle=shuffling,
        n_grid=n_grid,
        speed_cm=speed_cm,
        rate_scale=rate_scale,
    )

    granule_spikes = {}
    for traj in trajectories:
        granule_spikes[traj] = {}
        for poisson_seed in poisson_seeds:
            granule_spikes[traj][poisson_seed] = {}
            granule_spikes_poiss = granule_simulate(
                grid_spikes[traj][poisson_seed],
                dur_ms=dur_ms,
                network_type=network_type,
                grid_seed=grid_seed,
                pp_weight=pp_weight,
            )
            granule_spikes[traj][poisson_seed] = granule_spikes_poiss

    storage["grid_spikes"] = copy.deepcopy(grid_spikes)
    storage["granule_spikes"] = copy.deepcopy(granule_spikes)
    storage["parameters"] = copy.deepcopy(parameters)
    print("seed " + str(grid_seed) + " completed")
    storage.close()




# counts = [len(cell) for cell in granule_spikes[75][100]]

# np.mean(counts)/2

# stop = time.time()
# time_sec = stop - start
# time_min = time_sec / 60
# time_hour = time_min / 60
# print("time, ", time_sec, " sec, ", time_min, " min, ", time_hour, "hour  ")


# import matplotlib.pyplot as plt
# full_weights = [0.0009, 0.001, 0.0012, 0.0015, 0.0017, 0.002, 0.0025, .003]
# full_means = [0.2523, 0.3123, 0.4288, 0.6055, 0.7353, 0.9483, 1.413, 1.937]


# # 0.25, 0.41, 0.76, 1.387
# # 0.0009, 0.0012, 0.0017, 0.0025

# # 

# noff_weights = [0.0007, 0.0008, 0.0009, 0.001, 0.0015, 0.002, 0.0025]
# noff_means = [0.203, 0.317, 0.41, 0.529, 0.861, 1.247, 1.67]

# # 0.25, 0.41, 0.76, 1.387
# # 0.00075, 0.0009, 0.00097, 0.00175

# nofb_weights = [0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0015]
# nofb_means = [0.13, 0.288, 0.489, 0.76, 0.993, 2.45]

# # 0.25, 0.41, 0.76, 1.387
# # 0.0007, 0.0078, 0.0009,  0.0011

# disinh_weights = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
# disinh_means = [0.111, 0.217, 0.532, 0.945, 1.387]

# # 0.25, 0.41, 0.76, 1.387
# # 0.00061, 0.00068, 0.00077, 0.0009

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharey=True)
# ax1.plot(full_weights, full_means)
# ax1.set_ylabel('mean rate')
# ax1.set_xlabel('pp weight')
# ax1.tick_params(labelrotation=45)
# ax1.set_title('full')
# ax2.plot(noff_weights, noff_means)
# ax2.set_xlabel('pp weight')
# ax2.set_title('noff')
# ax2.tick_params(labelrotation=45)
# ax3.plot(nofb_weights, nofb_means)
# ax3.set_xlabel('pp weight')
# ax3.set_title('nofb')
# ax3.tick_params(labelrotation=45)
# ax4.plot(disinh_weights, disinh_means)
# ax4.set_xlabel('pp weight')
# ax4.set_title('disinhibited')
# ax4.tick_params(labelrotation=45)
# plt.tight_layout()