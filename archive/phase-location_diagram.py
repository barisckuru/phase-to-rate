#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:39:39 2021

@author: baris
"""

import matplotlib.pyplot as plt
import numpy as np
import grid_model as grid
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation as stg
from scipy import ndimage
import seaborn as sns



# =============================================================================
# =============================================================================
# # Phase-Location Diagram
# =============================================================================
# =============================================================================



def precession_spikes(overall, dur_s=5, n_sim=1000, T=0.1,
                      dt_s=0.002, bins_size_deg=7.2, shuffle=False):
    dur_ms = dur_s*1000
    asig = AnalogSignal(overall,
                        units=1*pq.Hz,
                        t_start=0*pq.s,
                        t_stop=dur_s*pq.s,
                        sampling_period=dt_s*pq.s,
                        sampling_interval=dt_s*pq.s)
    
    times = np.arange(0, 5+T, T)
    n_time_bins = int(dur_s/T)
    phase_norm_fact = 360/bins_size_deg
    n_phase_bins = int(720/bins_size_deg)
    phases = [[] for _ in range(n_time_bins)]
    n_sim = 1000
    for i in range(n_sim):
        train = stg.inhomogeneous_poisson_process(asig,
                                                  refractory_period = (0.001*pq.s),
                                                  as_array=True)*1000
        if shuffle is True:
            train = grid._randomize_grid_spikes(train, 100, time_ms=dur_ms)/1000
        else:
            train = train/1000
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = train[np.logical_and(train > time,
                                              train < times[j+1])]
            if curr_train.size > 0:
                phases[j] += list(curr_train % (T)/(T)*360)
                phases[j] += list(curr_train % (T)/(T)*360+360)
    counts = np.empty((n_phase_bins, n_time_bins))
    for i in range(n_phase_bins):
        for j, phases_in_time in enumerate(phases):
            phases_in_time = np.array(phases_in_time)
            counts[i][j] = ((bins_size_deg*(i) < phases_in_time) &
                            (phases_in_time < bins_size_deg*(i+1))).sum()
    f = int(1/T)
    spike_phases = counts*phase_norm_fact*f/n_sim
    spike_phases = ndimage.gaussian_filter(spike_phases, sigma=[1, 1])
    return spike_phases


spacing = 40
pos_peak = [100, 100]
orientation = 30
dur = 5

grid_rate = grid._grid_maker(spacing,
                             orientation, pos_peak).reshape(200, 200, 1)

grid_rates = np.append(grid_rate, grid_rate, axis=2)
spacings = [spacing, spacing]
grid_dist = grid._rate2dist(grid_rates, spacings)[:, :, 0].reshape(200, 200, 1)
trajs = np.array([50])
dist_trajs = grid._draw_traj(grid_dist, 1, trajs, dur_ms=5000)
rate_trajs = grid._draw_traj(grid_rate, 1, trajs, dur_ms=5000)
rate_trajs, rate_t_arr = grid._interp(rate_trajs, 5, new_dt_s=0.002)

grid_overall = grid._overall(dist_trajs,
                             rate_trajs, 180, 0.1, 1, 1, 5, 20, 5)[0, :, 0]

spike_phases = precession_spikes(grid_overall, shuffle=True)



# plotting
rate = grid_rate.reshape(200, 200)
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
cmap2 = sns.color_palette('RdYlBu_r', as_cmap=True)
f2, (ax1, ax2) = plt.subplots(2, 1, sharex=False,
                              gridspec_kw={'height_ratios': [1, 2]},
                              figsize=(7, 9))
# f2.tight_layout(pad=0.1)
im2 = ax2.imshow(spike_phases, aspect=1/7, cmap=cmap, extent=[0, 100, 720, 0])
ax2.set_ylim((0, 720))
im1 = ax1.imshow(rate[50:150, :], cmap=cmap2)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title(f'\u0394= {spacing} cm    ' +
              f'[$x_{0}$ $y_{0}$]={(100-np.array(pos_peak))}     ' +
              f'\u03A8={orientation} degree', loc='center')
ax2.set_xlabel('Position (cm)')
ax2.set_ylabel('Theta phase (deg)')
f2.subplots_adjust(right=0.8)
# cax = f2.add_axes([0.60,0.16,0.015,0.35])
cax = f2.add_axes([0.80, 0.16, 0.025, 0.35])
cbar = f2.colorbar(im2, cax=cax)
cbar.set_label('Hz', labelpad=15, rotation=270)
# =============================================================================
# =============================================================================
# # Phase-Location Diagram
# =============================================================================
# =============================================================================


