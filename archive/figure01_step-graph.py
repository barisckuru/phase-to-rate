#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 12:31:03 2022

@author: baris
"""


import matplotlib.pyplot as plt
import numpy as np
import grid_model
import matplotlib.gridspec as gridspec
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation as stg
from scipy import ndimage
import seaborn as sns
import pandas as pd




def precession_spikes(overall, dur_s=5, n_sim=1000, T=0.1,
                      dt_s=0.002, bins_size_deg=7.2, shuffle=False):
    dur_ms = dur_s*1000
    asig = AnalogSignal(overall,
                        units=1*pq.Hz,
                        t_start=0*pq.s,
                        t_stop=dur_s*pq.s,
                        sampling_period=dt_s*pq.s,
                        sampling_interval=dt_s*pq.s)
    
    times = np.arange(0, dur_s+T, T)
    n_time_bins = int(dur_s/T)
    phase_norm_fact = 360/bins_size_deg
    n_phase_bins = int(720/bins_size_deg)
    phases = [[] for _ in range(n_time_bins)]
    trains = []
    for i in range(n_sim):
        train = stg.inhomogeneous_poisson_process(asig,
                                                  refractory_period = (0.001*pq.s),
                                                  as_array=True)
        if shuffle is True:
            train = grid_model._randomize_grid_spikes(train, 100, time_ms=dur_ms)
        else:
            train = train
        trains.append(train)
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = train[np.logical_and(train > time,
                                              train < times[j+1])]
            if curr_train.size > 0:
                phases[j] += list(curr_train % (T)/(T)*360)
                # phases[j] += list(curr_train % (T)/(T)*360+360)
    counts = np.empty((n_phase_bins, n_time_bins))
    for i in range(n_phase_bins):
        for j, phases_in_time in enumerate(phases):
            phases_in_time = np.array(phases_in_time)
            counts[i][j] = ((bins_size_deg*(i) < phases_in_time) &
                            (phases_in_time < bins_size_deg*(i+1))).sum()
    f = int(1/T)
    spike_phases = counts*phase_norm_fact*f/n_sim
    spike_phases = ndimage.gaussian_filter(spike_phases, sigma=[1, 1])
    return trains, phases



spacing = 50
pos_peak = [100, 50]
orientation = 30
dur = 5
shuffle = False

grid_rate = grid_model._grid_maker(spacing,
                             orientation, pos_peak).reshape(200, 200, 1)

grid_rates = np.append(grid_rate, grid_rate, axis=2)
spacings = [spacing, spacing]
grid_dist = grid_model._rate2dist(grid_rates, spacings)[:, :, 0].reshape(200, 200, 1)
trajs = np.array([50])
dist_trajs = grid_model._draw_traj(grid_dist, 1, trajs, dur_ms=5000)
rate_trajs = grid_model._draw_traj(grid_rate, 1, trajs, dur_ms=5000)
rate_trajs, rate_t_arr = grid_model._interp(rate_trajs, 5, new_dt_s=0.002)

grid_overall = grid_model._overall(dist_trajs,
                             rate_trajs, 240, 0.1, 1, 1, 5, 20, 5)[0, :, 0]
spike_trains, spike_phases = precession_spikes(grid_overall, shuffle=shuffle, n_sim = 50)

means = [np.mean(i) for i in spike_phases]
repeated = np.repeat(means, 100)
plt.plot(repeated)
plt.plot(means)




'Trajectory -Overall - Spikes'
###################################
spacing = 20
pos_peak = [100,100]
orientation = 30
dur_s = 2
dur_ms = dur_s*1000
# overall_dir, rate, time_hr = overall(spacing, center, orientation, dur)
n_sim = 20
# poiss_seeds = np.arange(105,110,1)
# spikes = inhom_poiss(overall_dir, dur, poiss_seeds)



grid_rate = grid_model._grid_maker(spacing,
                             orientation, pos_peak).reshape(200, 200, 1)

grid_rates = np.append(grid_rate, grid_rate, axis=2)
spacings = [spacing, spacing]
grid_dist = grid_model._rate2dist(grid_rates, spacings)[:, :, 0].reshape(200, 200, 1)
trajs = np.array([50])
dist_trajs = grid_model._draw_traj(grid_dist, 1, trajs, dur_ms=dur_ms)
rate_trajs = grid_model._draw_traj(grid_rate, 1, trajs, dur_ms=dur_ms)
rate_trajs, rate_t_arr = grid_model._interp(rate_trajs, dur_s, new_dt_s=0.002)

grid_overall = grid_model._overall(dist_trajs,
                             rate_trajs, 240, 0.1, 1, 1, 5, 20, dur_s)[0, :, 0]

spike_trains, spike_phases = precession_spikes(grid_overall, dur_s = 2,  shuffle=shuffle, n_sim = 20)

means = [np.mean(i) for i in spike_phases]
repeated = np.repeat(means, 100)
# plt.plot(repeated)


plt.close('all')
sns.reset_orig()
sns.set(style='dark', palette='deep', font='Arial',font_scale=1,color_codes=True)
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
f1, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True, gridspec_kw={'height_ratios':[1.8,1,1,1]}, figsize=(6,8))

# f2, ax = plt.subplots(figsize=(11,5))
# plt.imshow(grid_rate[80:120,:80],cmap=cmap, extent=[0,40,60,40], aspect='auto')
# plt.imshow(grid_rate)
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Trajectory')


ax1.imshow(grid_rate[80:120,:80],cmap=cmap, extent=[0,2,1,0], aspect='auto')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_ylabel('Grid field')
# ax1.set_title('Trajectory')


ax2.plot(rate_t_arr, grid_overall)
# ax2.set_title('Overall Rate Profile')
ax2.set_ylabel('Frequency (Hz)')
    
ax3.eventplot(np.array(spike_trains[:5]), linewidth=0.7, linelengths=0.5)
# ax3.set_title('Spikes')
ax3.set_yticklabels([])
# ax3.set_yticks(np.arange(5))
# ax3.set_yticklabels(['seed1', 'seed2', 'seed3', 'seed4', 'seed5'])
ax3.set_ylabel('Spike trains')

ax4.plot(np.arange(0, 2, 2/2000), repeated/180)
# ax4.set_title('Phases')
ax4.set_ylim([0, 2])
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Phases (\u03C0)')

xpos = np.arange(0, dur_s+0.1, 0.1)
for xc in xpos:
    ax2.axvline(xc, color='0.5', linestyle='-.', linewidth=0.5)
    ax3.axvline(xc, color="0.5", linestyle='-.', linewidth=0.5)

f1.tight_layout()


save_dir = '/home/baris/paper/figures/'
f1.savefig(save_dir+'figure01_D.eps', dpi=200)
f1.savefig(save_dir+'figure01_D.png', dpi=200)

##############################################