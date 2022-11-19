# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 13:01:21 2022

@author: Daniel
"""


from phase_to_rate.neural_coding import load_spikes, rate_n_phase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import matplotlib as mpl
from mpl_toolkits import mplot3d


dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')

trajectories = [75]

# trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
#                 71, 70, 69, 68, 67, 66, 65, 60]

n_samples = 20
grid_seeds = np.arange(1,2,1)

# grid_seeds = np.array([1])

tuning = 'full'

all_codes = {}
for grid_seed in grid_seeds:
    # path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')

    # non-shuffled
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    
    print('ns path ok')
    
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

    # shuffled
    s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
    s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
    s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
    
    print('shuffled path ok')
    
    (
        s_grid_counts,
        s_grid_phases,
        s_grid_rate_code,
        s_grid_phase_code,
        s_grid_polar_code,
    ) = rate_n_phase(s_grid_spikes, trajectories, n_samples)
    
    (
        s_granule_counts,
        s_granule_phases,
        s_granule_rate_code,
        s_granule_phase_code,
        s_granule_polar_code,
    ) = rate_n_phase(s_granule_spikes, trajectories, n_samples)


    all_codes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
    all_codes[grid_seed]["shuffled"] = {"grid": {}, "granule": {}}
    all_codes[grid_seed]["non-shuffled"] = {"grid": {}, "granule": {}}

    all_codes[grid_seed]['non-shuffled']['grid'] = {
                      'rate': grid_rate_code,
                      'phase': grid_phase_code,
                      'polar': grid_polar_code}
    all_codes[grid_seed]['shuffled']['grid'] = {
                      'rate': s_grid_rate_code,
                      'phase': s_grid_phase_code,
                      'polar': s_grid_polar_code}
    all_codes[grid_seed]['non-shuffled']['granule'] = {
                      'rate': granule_rate_code,
                      'phase': granule_phase_code,
                      'polar': granule_polar_code}
    all_codes[grid_seed]['shuffled']['granule'] = {
                      'rate': s_granule_rate_code,
                      'phase': s_granule_phase_code,
                      'polar': s_granule_polar_code}
    
single_episode_grid = all_codes[1]["non-shuffled"]["grid"]["polar"][:,0,0]

grid = single_episode_grid.reshape((4000, 2), order="F")
grid = grid.reshape((200,20,2), order="C")
grid_mean = grid.mean(axis=0)

grid_bins_flat = grid.reshape((grid.shape[0] * grid.shape[1], 2))
grid_bins_silent_zeroth_trajectory = grid_bins_flat.sum(axis=1) == 0

single_episode_granule = all_codes[1]["non-shuffled"]["granule"]["polar"][:,0,0]

granule = single_episode_granule.reshape((40000, 2), order="F")
granule = granule.reshape((2000,20,2), order="C")
granule_mean = granule.mean(axis=0)

granule_bins_flat = granule.reshape((granule.shape[0] * granule.shape[1], 2))
granule_bins_silent_zeroth_trajectory = granule_bins_flat.sum(axis=1) == 0

grid_cell_silent_zeroth_trajectory = (grid[:,0,0] == 0) & (grid[:, 0, 0] == 0)


plt.figure()
plt.scatter(grid[:,0,0], grid[:,0,1])
plt.scatter(granule[:,0,0], granule[:,0,1])
plt.xlabel("x")
plt.ylabel("y")
plt.legend(("Grid Cells", "Granule Cells"))

plt.figure()
plt.scatter(grid[:,:,0], grid[:,:,1], alpha=0.3)
plt.xlim((-8, 8))
plt.ylim((-8, 8))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Grid Cells")

plt.figure()
plt.scatter(granule[:,:,0], granule[:,:,1], alpha=0.3)
plt.xlim((-8, 8))
plt.ylim((-8, 8))
plt.xlabel("x")
plt.ylabel("y")
plt.title("Granule Cells")

"""PLOTTING"""
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({'font.size': 32})
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
ax3 = fig.add_subplot(gs[1, 0], aspect='equal')
ax4 = fig.add_subplot(gs[1, 1], aspect='equal')

ax1.scatter(x=grid_bins_flat[:, 0], y=grid_bins_flat[:, 1], alpha=0.3)
ax2.scatter(x=granule_bins_flat[:, 0], y=granule_bins_flat[:, 1], alpha=0.3)

for cax in [ax1, ax2]:
    cax.set_xlim((-8, 8))
    cax.set_ylim((-8, 8))

hist_bin_width = 0.4
hist_grid, xbins_grid, ybins_grid = np.histogram2d(x=grid_bins_flat[~grid_bins_silent_zeroth_trajectory,0], y=grid_bins_flat[~grid_bins_silent_zeroth_trajectory,1], bins=(np.arange(-8, 8, hist_bin_width), np.arange(-8, 8, hist_bin_width)))
grid_image = ax3.imshow(hist_grid.T, cmap='jet',interpolation=None,origin='lower', extent=[-8, 8, -8, 8])
plt.colorbar(mappable=grid_image, ax=ax3)

hist_granule, xbins_granule, ybins_grid = np.histogram2d(x=granule_bins_flat[~granule_bins_silent_zeroth_trajectory,0], y=granule_bins_flat[~granule_bins_silent_zeroth_trajectory,1], bins=(np.arange(-8, 8, hist_bin_width), np.arange(-8, 8, hist_bin_width)))
granule_image = ax4.imshow(hist_granule.T, cmap='jet', interpolation=None, origin='lower', extent=[-8, 8, -8, 8])

plt.colorbar(mappable=granule_image, ax=ax4)

for cax in [ax1, ax2, ax3, ax4]:
    cax.set_xlabel("x")
    cax.set_ylabel("y")
    
"""PLOTTING"""
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({'font.size': 32})
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0], aspect='equal')
ax2 = fig.add_subplot(gs[0, 1], aspect='equal')
ax3 = fig.add_subplot(gs[1, 0], aspect='equal')
ax4 = fig.add_subplot(gs[1, 1], aspect='equal')

ax1.scatter(x=grid_bins_flat[:, 0], y=grid_bins_flat[:, 1], alpha=0.3)
ax2.scatter(x=granule_bins_flat[:, 0], y=granule_bins_flat[:, 1], alpha=0.3)

for cax in [ax1, ax2]:
    cax.set_xlim((-8, 8))
    cax.set_ylim((-8, 8))

hist_grid, xbins_grid, ybins_grid = np.histogram2d(x=grid_bins_flat[:,0], y=grid_bins_flat[:,1], bins=(np.arange(-8, 8, hist_bin_width), np.arange(-8, 8, hist_bin_width)))
grid_image = ax3.imshow(hist_grid.T, cmap='jet',interpolation=None,origin='lower', extent=[-8, 8, -8, 8])
plt.colorbar(mappable=grid_image, ax=ax3)

hist_granule, xbins_granule, ybins_grid = np.histogram2d(x=granule_bins_flat[:,0], y=granule_bins_flat[:,1], bins=(np.arange(-8, 8, hist_bin_width), np.arange(-8, 8, hist_bin_width)))
granule_image = ax4.imshow(hist_granule.T, cmap='jet', interpolation=None, origin='lower', extent=[-8, 8, -8, 8])

plt.colorbar(mappable=granule_image, ax=ax4)

for cax in [ax1, ax2, ax3, ax4]:
    cax.set_xlabel("x")
    cax.set_ylabel("y")
