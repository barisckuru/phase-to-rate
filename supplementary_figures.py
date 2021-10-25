#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:55:29 2021

@author: baris
"""

from neural_coding import load_spikes, rate_n_phase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



# =============================================================================
# =============================================================================
# # Phase Distibutions
# =============================================================================
# =============================================================================

grid_seeds = np.array([5])
trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

n_samples = 20
tuning = 'full'
n_iter = 2000

all_codes = {}
for grid_seed in grid_seeds:
    path = ("/home/baris/results/"+str(tuning)+
            "/collective/grid-seed_duration_shuffling_tuning_")
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
    s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
    s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
    

    
#Grid and granule phase distributions form normal network
bin_size_ms = 100
n_phase_bins=360
dur_ms = 2000


trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

trajectories = [70]

#spike to phase converter
def spike2phase(spikes, trajectories, bin_size_ms = 100,
                n_phase_bins=360, dur_ms = 2000):
    spike_phases=np.empty(0)
    n_bins = int(dur_ms/bin_size_ms)
    rad = n_phase_bins/360*2*np.pi    
    for traj in trajectories:
        spike = spikes[traj]
        for i in range(n_bins):
            for idx, val in np.ndenumerate(spike):
                curr_train = val[((bin_size_ms*(i) < val)
                                  & (val < bin_size_ms*(i+1)))]
                if curr_train.size != 0:
                    spike_phases = np.concatenate((spike_phases,
                                                   curr_train%(bin_size_ms)/
                                                   (bin_size_ms)*rad))
    return spike_phases


grid_phases = spike2phase(grid_spikes, trajectories)
granule_phases = spike2phase(granule_spikes, trajectories)



plt.close('all')
phase_dist= np.append(grid_phases, granule_phases)
cell = ['grid']*(grid_phases.shape[0]) + ['granule']*(granule_phases.shape[0])
phase_df = pd.DataFrame({'phase distribution ($\pi$)': phase_dist,
                   'cell': pd.Categorical(cell)})

fig_inh = sns.histplot(data=phase_df, x='phase distribution ($\pi$)',
                       kde=True, hue='cell', binwidth=(2*np.pi/180))
plt.title('Full Network (Non-Shuffled)')
sns.set(context='paper', style='darkgrid', palette='deep',
        font='Arial', font_scale=1.5, color_codes=True,
        rc={'figure.figsize': (8, 4)})
fig_inh.set(ylim=(0, 100000))

plt.tight_layout()

plt.figure()
n, bins, pathces = plt.hist(phases/np.pi, bins = np.arange(0, 2.001, 2/100))


# =============================================================================
# =============================================================================
# # Phase Distibutions
# =============================================================================
# =============================================================================