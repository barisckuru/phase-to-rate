#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:19:20 2021

@author: baris
"""

from neural_coding_new import load_spikes, rate_n_phase

trajectories = [75, 70]
n_samples = 20

path = "/home/baris/results/grid-seed_duration_shuffling_tuning_3_2000_shuffled_tuned"


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
