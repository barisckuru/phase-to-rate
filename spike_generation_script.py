#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:16:37 2021

@author: baris
"""

from pydentate import net_tunedrev, neuron_tools
from grid_model import grid_simulate
import shelve
import copy
import time
import numpy as np
import scipy.stats as stats
from pydentate_integrate import granule_simulate

start = time.time()

"""Setup"""
neuron_tools.load_compiled_mechanisms(path='/home/baris/pydentate/mechs/x86_64/libnrnmech.so')


"""Parameters"""
grid_seeds = [1]
poisson_seeds = [100]
trajectories = [75]  # In cm
shuffled = ["shuffled", "non-shuffled"]
poisson_reseeding = True  # Controls seeding between trajectories
speed_cm = 20
dur_ms = 100
rate_scale = 5
n_grid = 200
pp_weight = 9e-4
network_type = 'tuned'

print('grid', grid_seeds, 'poiss', poisson_seeds, 'traj', trajectories, 'dur', dur_ms)



for grid_seed in grid_seeds:
    output_path = f'{grid_seed}_{poisson_seeds}_{poisson_reseeding}'
    # save_storage TODO shelve
    storage = shelve.open('grid_seed_poisson_seed_reseeding_'+output_path, writeback=True)
    grid_dict = {'grid_spikes': {}}
    granule_dict ={'granule_spikes': {}}
    for shuffling in shuffled:
        grid_spikes, _ = grid_simulate(trajs=trajectories,
                                    dur_ms=dur_ms,
                                    grid_seed=grid_seed,
                                    poiss_seeds=poisson_seeds,
                                    shuffle=shuffling,
                                    diff_seed=poisson_reseeding,
                                    n_grid=n_grid,
                                    speed_cm=speed_cm,
                                    rate_scale=rate_scale)
        granule_spikes = granule_simulate(grid_spikes, 
                                          poisson_seeds=poisson_seeds, 
                                          network_type=network_type, 
                                          grid_seed=grid_seed, 
                                          pp_weight=pp_weight)

        grid_dict['grid_spikes'][shuffling] = grid_spikes
        granule_dict['granule_spikes'][shuffling] = granule_spikes

        storage['grid_spikes'] = copy.deepcopy(grid_dict)
        storage['granule_spikes'] = copy.deepcopy(granule_dict)

# "grid_seed_poisson_seed_reseeding-True"


stop = time.time()
time_sec = stop-start
time_min = time_sec/60
time_hour = time_min/60
print('time, ', time_sec, ' sec, ', time_min, ' min, ', time_hour, 'hour  ')