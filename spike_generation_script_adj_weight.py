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
import glob

start = time.time()

"""Setup"""
neuron_tools.load_compiled_mechanisms(
    path="/home/baris/pydentate/mechs/x86_64/libnrnmech.so"
)


"""Parameters"""
# grid_seeds = [1,2,3,4,5,6,7,8,9,10]

# grid_seeds = [11,12,13,14,15,16,17,18,19,20]

grid_seeds = [21,22,23,24,25,26,27,28,29,30]

# shuffling = "non-shuffled"
shuffling = "shuffled"

pp_weight = 0.00067
# network_type = "full"
network_type = "no-feedback" 
# network_type = "disinhibited"
# network_type = "no-feedforward"

# grid_seeds = [11]

# trajectories, p = [75], 100  # In cm
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
trajectories, p = [60], 1500
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
print(shuffling)
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
    save_dir = (f'/home/baris/results/tempotron/adjusted_weight/'
                +f'{network_type}/seperate/seed_'
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



def _collect_spikes(
    grid_seed,
    shuffling,
    dur_ms,
    trajectories,
    network_type,
    weight,
    path="/home/baris/results/tempotron/adjusted_weight/"
):

    collective_path = (path + str(network_type) 
                       + "/collective/")
    separate_path = (path + str(network_type) + "/seperate/" +
                     "seed_" + str(grid_seed) + "/"+str(weight)+"/")
    print(separate_path)

    for traj in trajectories:
        print(traj)
        file = glob.glob(os.path.join(
            separate_path, "*%s*_%s*.dat" % (traj, shuffling)))[0][0:-4]
        fname = f"{separate_path}*{traj}]*_{shuffling}*.dat"
        file = glob.glob(fname)[0][0:-4]
        print(file)
        storage_old = shelve.open(file)
        output_name = f"{grid_seed}"
        collective_storage = []
        collective_storage = (collective_path +
                              "grid-seed_duration_shuffling_tuning_trajs_")
        collective_storage = (collective_storage + output_name + "_" +
                              str(dur_ms) + "_" + shuffling + "_" 
                              + network_type + '-adjusted_75-60')
        storage = shelve.open(collective_storage, writeback=True)
        traj_key = str(traj)
        storage[traj_key] = {}
        storage[traj_key]["grid_spikes"] = copy.deepcopy(
            storage_old["grid_spikes"][traj]
        )
        storage[traj_key]["granule_spikes"] = copy.deepcopy(
            storage_old["granule_spikes"][traj]
        )
        storage[traj_key]["parameters"] = copy.deepcopy(
            storage_old["parameters"])
        storage.close()
        storage_old.close()

# grid-seed_duration_shuffling_tuning_trajs_20_2000_shuffled_no-feedback_75-60
# trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
#                 71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
# grid_seeds = np.arange(1,11,1)

# disinh 0.0006
# noff 0.00071
weight = 0.00071
trajectories = [75, 60]
grid_seeds = np.arange(11,21,1)
tuning = 'no-feedforward'

for i in grid_seeds:
    print(i)
    _collect_spikes(i, 'non-shuffled', 2000, trajectories, tuning, weight)
    
for i in grid_seeds:
    print(i)
    _collect_spikes(i, 'shuffled', 2000, trajectories, tuning, weight)