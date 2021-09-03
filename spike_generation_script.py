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
dur_ms = 200
rate_scale = 5
n_grid = 200
pp_weight = 9e-4
network_type = 'tuned'

print('grid', grid_seeds, 'poiss', poisson_seeds, 'traj', trajectories, 'dur', dur_ms)



for grid_seed in grid_seeds:
    output_path = f'{grid_seed}_{poisson_seeds}_{poisson_reseeding}'
    # save_storage TODO shelve
    storage = shelve.open('/home/baris/results/grid_seed_poisson_seed_reseeding_'+output_path, writeback=True)
    storage = {'grid_spikes': {},
               'granule_spikes': {}}
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
                                          dur_ms=dur_ms, 
                                          poisson_seeds=poisson_seeds, 
                                          network_type=network_type, 
                                          grid_seed=grid_seed, 
                                          pp_weight=pp_weight)

        storage['grid_spikes'][shuffling] = copy.deepcopy(grid_spikes)
        storage['granule_spikes'][shuffling] = copy.deepcopy(granule_spikes)


# "grid_seed_poisson_seed_reseeding-True"


stop = time.time()
time_sec = stop-start
time_min = time_sec/60
time_hour = time_min/60
print('time, ', time_sec, ' sec, ', time_min, ' min, ', time_hour, 'hour  ')