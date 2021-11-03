#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:21:07 2021

@author: baris
"""

from neural_coding import load_spikes, rate_n_phase
from perceptron_new import run_perceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
import copy



# =============================================================================
# =============================================================================
# # Load the data
# =============================================================================
# =============================================================================


trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

# trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
#                 71, 70, 69, 68, 67, 66, 65, 60]

n_samples = 20
grid_seeds = np.arange(1,11,1)

# grid_seeds = np.array([1])

tuning = 'full'

all_codes = {}
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
    



# =============================================================================
# Perceptron 
# =============================================================================

rate_learning_rate = 1e-4
polar_learning_rate = 1e-4
phase_learning_rate = 1e-4
n_iter = 2000
optimizer='Adam'

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
                    elif code == 'rate':
                        learning_rate = polar_learning_rate
                        if cell == 'granule':
                            n_iter = 10000
                    input_code = all_codes[grid_seed][shuffling][cell][code]
                    perceptron_input = np.hstack((input_code[:, :, 0], input_code[:, :, traj_idx + 1]))
                    th_cross, train_loss = run_perceptron(
                        perceptron_input,
                        grid_seed,
                        optimizer=optimizer,
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



 # load pickled perceptron results

# data = pickle.load( open( "75-15_disinhibited_perceptron_speed.pkl", "rb" ) )

# drop rows for 45 cm and 30 cm for perceptron figures



    
# for data_sing in add_data:
#     data_sing[4] = 1
#     data_sing[5] = 'shuffled'
#     data_sing[6] = 'grid'
#     data_sing[7] = 'phase'
    
# data11 = copy.deepcopy(data)

# data[16:32] = add_data


'pickle perceptron results'
file_name = "75-15_"+str(tuning)+"_perceptron_speed_SparseAdam.pkl"
open_file = open(file_name, "wb")
pickle.dump(data, open_file)
open_file.close()

df = pd.DataFrame(data,
                  columns=['distance', 'speed', 'threshold_crossing',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'learning_rate'])

i = df[(df.distance==45) | (df.distance==60)].index
df = df.drop(i)

grid_rate = df.groupby(['cell_type', 'code_type']).get_group(('grid', 'rate'))
grid_phase = df.groupby(['cell_type', 'code_type']).get_group(('grid', 'phase'))
grid_polar = df.groupby(['cell_type', 'code_type']).get_group(('grid', 'polar'))
granule_rate = df.groupby(['cell_type', 'code_type']).get_group(('granule', 'rate'))
granule_phase = df.groupby(['cell_type', 'code_type']).get_group(('granule', 'phase'))
granule_polar = df.groupby(['cell_type', 'code_type']).get_group(('granule', 'polar'))


import copy
data1 = copy.deepcopy(data)


fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex="col")

fig1.suptitle(
    "Perceptron learning speed "+ str(tuning)+" network\n  "
    + str(n_samples)
    + " samples per trajectory, "
    + str(len(grid_seeds))
    + " grid seeds \n\n rate code learning rate = "
    + format(rate_learning_rate, ".1E")
    + "\n phase code learning rate = "
    + format(phase_learning_rate, ".1E")
    + "\n polar code learning rate = "
    + format(polar_learning_rate, ".1E")
)

# for ax in fig1.axes:
#     ax.set_ylabel("Speed (1/N)")

ax1.set_title("grid rate code")
ax2.set_title("grid phase code")
ax3.set_title('grid polar code')
ax4.set_title("granule rate code")
ax5.set_title("granule phase code")
ax6.set_title('granule polar code')

sns.lineplot(ax=ax1,
             data=grid_rate, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd'
             )
sns.lineplot(ax=ax2,
             data=grid_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )
sns.lineplot(ax=ax3,
             data=grid_polar, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )
sns.lineplot(ax=ax4,
             data=granule_rate, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd'
             )
sns.lineplot(ax=ax5,
             data=granule_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )
sns.lineplot(ax=ax6,
             data=granule_polar, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )


plt.tight_layout()
   
    