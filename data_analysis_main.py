#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:19:20 2021

@author: baris
"""

from neural_coding import load_spikes, rate_n_phase
from perceptron import run_perceptron
import numpy as np

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72, 71, 70, 69, 68, 67, 66, 65, 60]
n_samples = 20

path = "/home/baris/results/grid-seed_duration_shuffling_tuning_2_2000_shuffled_tuned"


grid_spikes = load_spikes(path, "grid", trajectories, n_samples)
granule_spikes = load_spikes(path, "granule", trajectories, n_samples)

(
    grid_counts,
    grid_phases,
    grid_rate_code,
    grid_phase_code,
    grid_polar_code,
) = rate_n_phase(grid_spikes, trajectories, n_samples)

(
    granule_counts,
    granule_phases,
    granule_rate_code,
    granule_phase_code,
    granule_polar_code,
) = rate_n_phase(granule_spikes, trajectories, n_samples)


def encode_trajs(code, trajectories)
for traj_idx, traj in enumerate(trajectories):
    perceptron_input = np.hstack((granule_rate_code[:,:,0], granule_rate_code[:,:,traj_idx+1]))
    run_perceptron
    


stacked = np.hstack((granule_rate_code[:,:,0], granule_rate_code[:,:,1]))










spikes.keys()

type(spikes[75])

len(grid_spikes[75][0])

len(spikes[70])


granule_75 = granule_spikes[75][19]

grid_75 = grid_spikes[75][19]

import matplotlib.pyplot as plt

plt.figure()
plt.eventplot(granule_75)
plt.figure()
plt.eventplot(grid_75)
