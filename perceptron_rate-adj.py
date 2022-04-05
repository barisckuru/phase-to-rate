#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:52:41 2022

@author: baris
"""
from neural_coding import load_spikes, rate_n_phase
from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import copy

save_dir = '/home/baris/results/pickled/'

rate_learning_rate = 1e-4
phase_learning_rate = 1e-4
n_iter = 10000

trajectories = [75, 60]

n_samples = 20

tuning = 'adjusted_noff'
shuffle = ['non-shuffled', 'shuffled']
all_codes = {"shuffled": {}, "non-shuffled": {}}
grid_seeds = np.arange(11,21,1)
for shuffling in shuffle:
    for grid_seed in grid_seeds:
        adj_path = '/home/baris/results/tempotron/adjusted_weight/no-feedforward/collective/'
        adj_fname= f'grid-seed_duration_shuffling_tuning_trajs_{grid_seed}_2000_{shuffling}_no-feedforward-adjusted_75-60'
        full_path = '/home/baris/results/tempotron/full/collective/'
        full_fname= f'grid-seed_duration_shuffling_tuning_trajs_{grid_seed}_2000_{shuffling}_full_75-60'
        
        if tuning == 'full':
            path = full_path
            fname = full_fname
        elif tuning == 'adjusted_noff':
            path = adj_path
            fname = adj_fname
        directory = path + fname
        granule_spikes = load_spikes(directory, "granule", trajectories, n_samples) 
        (
            granule_counts,
            granule_phases,
            granule_rate_code,
            granule_phase_code,
            granule_polar_code,
        ) = rate_n_phase(granule_spikes, trajectories, n_samples)
        all_codes[shuffling][grid_seed] = {'rate': granule_rate_code,
                                           'phase': granule_phase_code}
        print(f'{grid_seed} done')


fname = f'{tuning}_codes_grid-seeds_11-30.pkl'
with open(save_dir+fname, 'wb') as f:
    pickle.dump(all_codes, f)
    
data = []
for shuffling in all_codes:
    for grid_seed in all_codes[shuffling]:
            for code in all_codes[shuffling][grid_seed]:
                if code == 'phase':
                    learning_rate = phase_learning_rate
                elif code == 'rate':
                    learning_rate = rate_learning_rate
                input_code = all_codes[shuffling][grid_seed][code]
                perceptron_input = np.hstack((input_code[:, :, 0], input_code[:, :, 1]))
                th_cross, train_loss = run_perceptron(
                    perceptron_input,
                    grid_seed,
                    learning_rate=learning_rate,
                    n_iter=n_iter)
                data_sing = [grid_seed, 1/th_cross, th_cross, shuffling, code, learning_rate]
                data.append(copy.deepcopy(data_sing))
                print(grid_seed)
                print(shuffling)
                print(code)

df = pd.DataFrame(data,
                  columns=['grid_seed', 'speed', 'threshold_crossing',
                           'shuffling', 'code_type', 'learning_rate'])

fname2 = f'{tuning}_adjusted_perceptron.pkl'
with open(save_dir+fname2, 'wb') as f:
    pickle.dump(df, f)
    
sns.reset_orig()
rate_df = df.groupby(['code_type']).get_group(('rate'))
phase_df = df.groupby(['code_type']).get_group(('phase'))

import copy
data1 = copy.deepcopy(data)

sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)
fig1, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

fig1.suptitle(
    "Perceptron learning speed "+ str(tuning)+" network\n  "
    + str(n_samples)
    + " samples per trajectory, "
    + str(len(grid_seeds))
    + " grid seeds \n\n rate code learning rate = "
    + format(rate_learning_rate, ".1E")
    + "\n phase code learning rate = "
    + format(phase_learning_rate, ".1E")
)

ax1.set_title("granule rate code")
ax2.set_title("granule phase code")


sns.barplot(ax=ax1,
            data=rate_df, x="shuffling", y="speed",
            hue="shuffling")
sns.barplot(ax=ax2,
            data=phase_df, x="shuffling", y="speed",
            hue="shuffling")
plt.tight_layout()   