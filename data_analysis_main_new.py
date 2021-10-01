#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:19:20 2021

@author: baris
"""

from neural_coding import load_spikes, rate_n_phase
from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
n_samples = 20
grid_seeds = np.arange(1,11,1)

rate_learning_rate = 1e-4
phase_learning_rate = 1e-3
n_iter = 2000

all_codes = {}
for grid_seed in grid_seeds:
    path = "/home/baris/results/full/collective/grid-seed_duration_shuffling_tuning_"
    
    # non-shuffled
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_full.dir")
    grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    
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
    s_path = (path + str(grid_seed) + "_2000_shuffled_full.dir")
    s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
    s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
    
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

    all_codes[grid_seed]['non-shuffled']['grid'] = {'rate': grid_rate_code,
                      'phase': grid_phase_code}
    all_codes[grid_seed]['shuffled']['grid'] = {'rate': s_grid_rate_code,
                      'phase': s_grid_phase_code}
    all_codes[grid_seed]['non-shuffled']['granule'] = {'rate': granule_rate_code,
                      'phase': granule_phase_code}
    all_codes[grid_seed]['shuffled']['granule'] = {'rate': s_granule_rate_code,
                      'phase': s_granule_phase_code}

    # all_codes[grid_seed]['non-shuffled']['grid'] = {'rate': grid_rate_code,
    #                   'phase': grid_phase_code,
    #                   'polar': grid_polar_code}
    # all_codes[grid_seed]['shuffled']['grid'] = {'rate': s_grid_rate_code,
    #                   'phase': s_grid_phase_code,
    #                   'polar': s_grid_polar_code}
    # all_codes[grid_seed]['non-shuffled']['granule'] = {'rate': granule_rate_code,
    #                   'phase': granule_phase_code,
    #                   'polar': granule_polar_code}
    # all_codes[grid_seed]['shuffled']['granule'] = {'rate': s_granule_rate_code,
    #                   'phase': s_granule_phase_code,
    #                   'polar': s_granule_polar_code}


all_codes[grid_seed]['non-shuffled']['granule']['rate'].shape

import copy
data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for traj_idx in range(len(trajectories)-1):
                    if code == 'phase':
                        learning_rate = phase_learning_rate
                    elif code == 'rate':
                        learning_rate = rate_learning_rate
                        if cell == 'granule':
                            n_iter = 10000
                    input_code = all_codes[grid_seed][shuffling][cell][code]
                    perceptron_input = np.hstack((input_code[:, :, 0], input_code[:, :, traj_idx + 1]))
                    th_cross, train_loss = run_perceptron(
                        perceptron_input,
                        grid_seed,
                        learning_rate=learning_rate,
                        n_iter=n_iter)
                    traj = trajectories[traj_idx+1]
                    idx = 75 - traj
                    comp_trajectories = str(75)+'_'+str(traj)
                    data_sing = [idx, 1/th_cross, th_cross, comp_trajectories, grid_seed, shuffling, cell, code, learning_rate]
                    data.append(copy.deepcopy(data_sing))
                    print(traj)
                    print(grid_seed)
                    print(shuffling)
                    print(cell)
                    print(code)

df = pd.DataFrame(data,
                  columns=['distance', 'speed', 'threshold_crossing',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'learning_rate'])


grid_rate = df.groupby(['cell_type', 'code_type']).get_group(('grid', 'rate'))
grid_phase = df.groupby(['cell_type', 'code_type']).get_group(('grid', 'phase'))
granule_rate = df.groupby(['cell_type', 'code_type']).get_group(('granule', 'rate'))
granule_phase = df.groupby(['cell_type', 'code_type']).get_group(('granule', 'phase'))




#here!



import pickle 
file_name = "75-15_full_perceptron_speed.pkl"

open_file = open(file_name, "wb")
pickle.dump(data, open_file)
open_file.close()

import copy
data1 = copy.deepcopy(data)


fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col")

fig1.suptitle(
    "Perceptron learning speed \n  "
    + str(n_samples)
    + " samples per trajectory, "
    + str(len(grid_seeds))
    + " grid seeds \n\n rate code learning rate = "
    + format(rate_learning_rate, ".1E")
    + "\n phase code learning rate = "
    + format(phase_learning_rate, ".1E")
)

# for ax in fig1.axes:
#     ax.set_ylabel("Speed (1/N)")

ax1.set_title("grid rate code")
ax2.set_title("grid phase code")
ax3.set_title("granule rate code")
ax4.set_title("granule phase code")


sns.lineplot(ax=ax1,
             data=grid_rate, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd'
             )
sns.lineplot(ax=ax2,
             data=grid_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )
sns.lineplot(ax=ax3,
             data=granule_rate, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd'
             )
sns.lineplot(ax=ax4,
             data=granule_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )


plt.tight_layout()






'pearson r'


from scipy.stats import pearsonr,  spearmanr

r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for traj_idx in range(len(trajectories)):
                    r_input_code = all_codes[grid_seed][shuffling][cell][code]
                    f = int(r_input_code.shape[0]/8000)
                    baseline_traj = np.concatenate((
                        r_input_code[1000*f:1200*f, 0, 0], 
                        r_input_code[5000*f:5200*f, 0, 0]))
                    for poisson in range(r_input_code.shape[2]):
                        compared_traj = r_input_code[:, poisson, traj_idx]
                        compared_traj = np.concatenate((
                            compared_traj[1000*f:1200*f],
                            compared_traj[5000*f:5200*f]))
                        pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                        traj = trajectories[traj_idx]
                        idx = 75 - traj
                        comp_trajectories = str(75)+'_'+str(traj)
                        r_data_sing = [idx, pearson_r, poisson,
                                     comp_trajectories, grid_seed, shuffling,
                                     cell, code]
                        r_data.append(copy.deepcopy(r_data_sing))

df = pd.DataFrame(r_data,
                  columns=['distance', 'pearson_r', 'poisson_seed',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')

new_df = df.pivot(columns='cell_type')['pearson_r','distance','code_type', 'shuffling'].reset

a = df.pivot(columns='cell_type')


grouped = df.groupby(['shuffling', 'code_type', 'cell_type'])
shuffled_grid_rate = grouped.get_group(('shuffled', 'rate', 'grid'))
shuffled_grid_phase = grouped.get_group(('shuffled', 'phase', 'grid'))
shuffled_granule_rate = grouped.get_group(('shuffled', 'rate', 'granule'))
shuffled_granule_phase = grouped.get_group(('shuffled', 'phase', 'granule'))


nonshuffled_grid_rate = grouped.get_group(('non-shuffled', 'rate', 'grid'))
nonshuffled_grid_phase = grouped.get_group(('non-shuffled', 'phase', 'grid'))
nonshuffled_granule_rate = grouped.get_group(('non-shuffled', 'rate', 'granule'))
nonshuffled_granule_phase = grouped.get_group(('non-shuffled', 'phase', 'granule'))


sns.scatterplot(x=list(shuffled_grid_rate['pearson_r']), y =list(shuffled_granule_rate['pearson_r'] ))


sns.scatterplot(x=list(shuffled_grid_rate['pearson_r']), y =list(shuffled_granule_rate['pearson_r'] ))
plt.figure()
sns.scatterplot(x=list(nonshuffled_grid_rate['pearson_r']), y =list(nonshuffled_granule_rate['pearson_r'] ))


sns.scatterplot(x=list(shuffled_grid_phase['pearson_r']), y =list(shuffled_granule_phase['pearson_r'] ))
plt.figure()
sns.scatterplot(x=list(nonshuffled_grid_phase['pearson_r']), y =list(nonshuffled_granule_phase['pearson_r'] ))





type(list(shuffled_grid_rate['pearson_r']))
sns.pairplot(df)


fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col")

fig2.suptitle(
    "Perceptron learning speed \n  "
    + str(n_samples)
    + " samples per trajectory, "
    + str(len(grid_seeds))
    + " grid seeds \n\n rate code learning rate = "
    + format(rate_learning_rate, ".1E")
    + "\n phase code learning rate = "
    + format(phase_learning_rate, ".1E")
)

ax1.set_title("grid rate code")
ax2.set_title("grid phase code")
ax3.set_title("granule rate code")
ax4.set_title("granule phase code")


sns.scatterplot(ax=ax1,
             data=shuffled_rate, x="grid", y="granule",
             hue="distance")
sns.scatterplot(ax=ax2,
             data=shuffled_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )
sns.scatterplot(ax=ax3,
             data=nonshuffled_rate, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd'
             )
sns.scatterplot(ax=ax4,
             data=nonshuffled_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )


plt.tight_layout()





