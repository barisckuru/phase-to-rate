#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:15:39 2021

@author: baris
"""


from neural_coding import load_spikes, rate_n_phase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

n_samples = 20
grid_seeds = np.arange(1,11,1)

tuning = 'no-feedback'

ns_grid_phases = np.empty(0)
ns_granule_phases = np.empty(0)
s_grid_phases = np.empty(0)
s_granule_phases = np.empty(0)

for grid_seed in grid_seeds:
    path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
    
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
        shuf_grid_phases,
        s_grid_rate_code,
        s_grid_phase_code,
        s_grid_polar_code,
    ) = rate_n_phase(s_grid_spikes, trajectories, n_samples)
    
    (
        s_granule_counts,
        shuf_granule_phases,
        s_granule_rate_code,
        s_granule_phase_code,
        s_granule_polar_code,
    ) = rate_n_phase(s_granule_spikes, trajectories, n_samples)
    
    curr_ns_grid_phases = grid_phases.flatten()
    ns_grid_phases = np.append(ns_grid_phases,
                               curr_ns_grid_phases[curr_ns_grid_phases!=0])
    curr_ns_granule_phases = granule_phases.flatten()
    ns_granule_phases = np.append(ns_granule_phases, 
                                  curr_ns_granule_phases[curr_ns_granule_phases!=0])
    
    
    curr_s_grid_phases = shuf_grid_phases.flatten()
    s_grid_phases = np.append(s_grid_phases,
                              curr_s_grid_phases[curr_s_grid_phases!=0])
    curr_s_granule_phases = shuf_granule_phases.flatten()
    s_granule_phases = np.append(s_granule_phases,
                                 curr_s_granule_phases[curr_s_granule_phases!=0])


with open(f'phase-distribution_{tuning}.pkl', 'wb') as f:
    all_codes = pickle.load(f)




plt.close('all')
phase_dist= np.concatenate((ns_grid_phases, ns_granule_phases, s_grid_phases, s_granule_phases))
phase_dist = phase_dist/np.pi
cell = (['grid']*(ns_grid_phases.shape[0]) +
        ['granule']*(ns_granule_phases.shape[0]) +
        ['grid']*(s_grid_phases.shape[0]) +
        ['granule']*(s_granule_phases.shape[0]))
shuffling = ((len(ns_grid_phases)+len(ns_granule_phases))*['non-shuffled'] +
             (len(s_grid_phases)+len(s_granule_phases))*['shuffled'])
phase_df = pd.DataFrame({'phase (pi)': phase_dist,
                         'cell': pd.Categorical(cell),
                         'shuffling': pd.Categorical(shuffling)})
phase_df.to_pickle(f'fig2_phase-dist_{tuning}.pkl')

fig_inh = sns.histplot(data=phase_df, x='phase distribution ($\pi$)',
                       kde=True, hue='cell', binwidth=(2*np.pi/180))
plt.title(f'{tuning}                non-shuffled')
fig_inh.set(ylim=(0, 1200000))

plt.tight_layout()


phase_dist= np.append(s_grid_phases, s_granule_phases)
phase_dist = phase_dist/np.pi
cell = ['grid']*(s_grid_phases.shape[0]) + ['granule']*(s_granule_phases.shape[0])
phase_df = pd.DataFrame({'phase distribution ($\pi$)': phase_dist,
                   'cell': pd.Categorical(cell)})
plt.figure()
fig_shuf = sns.histplot(data=phase_df, x='phase distribution ($\pi$)',
                       kde=True, hue='cell', binwidth=(2*np.pi/180))
plt.title(f'{tuning}               shuffled')

fig_shuf.set(ylim=(0, 1200000))

plt.tight_layout()