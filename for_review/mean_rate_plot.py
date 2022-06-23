#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:46:21 2021

@author: baris
"""

from neural_coding import load_spikes, rate_n_phase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


#load data, rates


trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

n_samples = 20
grid_seeds = np.arange(1,11,1)
n_grid= len(grid_seeds)

tunes = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
means = []
for tuning in tunes:
    for grid_seed in grid_seeds:
        path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
        
        # non-shuffled
        ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
        grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
        granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
        grid_counts,_ ,_ ,_ ,_ = rate_n_phase(grid_spikes, trajectories, n_samples)
        granule_counts,_ ,_ ,_ ,_ = rate_n_phase(granule_spikes, trajectories, n_samples)
        
        
        s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
        s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
        s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
        s_grid_counts,_ ,_ ,_ ,_ = rate_n_phase(s_grid_spikes, trajectories, n_samples)
        s_granule_counts,_ ,_ ,_ ,_ = rate_n_phase(s_granule_spikes, trajectories, n_samples)
        print(f'{grid_seed} loaded')
    
    
        ns_grid_mean = [grid_seed, grid_counts.mean()*10, 'non-shuffled', 'grid']
        s_grid_mean = [grid_seed, s_grid_counts.mean()*10, 'shuffled', 'grid']
        ns_granule_mean = [grid_seed, granule_counts.mean()*10, 'non-shuffled', 'granule']
        s_granule_mean = [grid_seed, s_granule_counts.mean()*10, 'shuffled', 'granule']
        means.append(ns_grid_mean)
        means.append(s_grid_mean)
        means.append(ns_granule_mean)
        means.append(s_granule_mean)


all_means = []
ct=0
for mean in means:
    if ct <40 :
        all_means.append(mean+['full'])
    elif ct <80 :
        all_means.append(mean+['no-feedforward'])
    elif ct <120 :
        all_means.append(mean+['no-feedback'])
    elif ct <160 :
        all_means.append(mean+['disinhibited'])
    ct+=1

# grids = 4*(n_grid*[1]+4*[2]+4*[3]+4*[4]+4*[5]+4*[6]+4*[7]+4*[8]+4*[9]+4*[10])
grids=[]
for grid in grid_seeds:
    grids+=16*[grid]
shuffling = 8*n_grid*(['non-shuffled'] + ['shuffled'])
cell = 4*n_grid*(2*['grid']+2*['granule'])
tuning= []
for tune in tunes:
    tuning+=40*[tune]

all_means = np.stack((means,tuning, grids, shuffling, cell))

all_means = list(zip(means,tuning, grids, shuffling, cell))

means_df = pd.DataFrame(all_means,
                        columns=['mean_rate', 'tuning', 'grid_seeds',
                                 'shuffling', 'cell'])


means_df = pd.DataFrame(all_means,
                        columns=['grid_seeds', 'mean_rate',
                                 'shuffling', 'cell', 'tuning'])

means_df['mean_rate'] = means_df['mean_rate'].astype('float')
means_df['grid_seeds'] = means_df['grid_seeds'].astype('float')


sns.barplot(data=means_df.loc[means_df['cell']=='granule'], x= 'tuning', y='mean_rate', hue='shuffling',
            ci='sd', capsize=0.2, errwidth=(2))

granule_data = means_df.loc[means_df['cell']=='granule']

ax = sns.catplot(data=granule_data, kind='bar', x='tuning', y='mean_rate',
                 hue='shuffling', ci='sd', capsize=0.2, errwidth=2)

ax = sns.swarmplot(data=granule_data, x='tuning', y='mean_rate',
                   hue='shuffling', color='black', dodge=True)

ax.get_legend().set_visible(False)

plt.title('Granule cells mean firing rate')
plt.tight_layout()
# mean rates

sns.set()
ns_grid_code = np.array(((all_codes[grid_seed]['non-shuffled']['grid']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))
s_grid_code = np.array(((all_codes[grid_seed]['shuffled']['grid']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))
ns_granule_code = np.array(((all_codes[grid_seed]['non-shuffled']['granule']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))
s_granule_code = np.array(((all_codes[grid_seed]['shuffled']['granule']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))


means = np.concatenate((ns_grid_code, s_grid_code,
                       ns_granule_code, s_granule_code))
shuffling =  2*(340*['non-shuffled']+340*['shuffled'])
cell =  680*['grid']+680*['granule']

avarage_rate = np.stack((means, shuffling, cell), axis=1)

means_df = pd.DataFrame(avarage_rate,
                        columns=['mean_rate', 'shuffling', 'cell'])

means_df['mean_rate'] = means_df['mean_rate'].astype('float')

sns.barplot(data=means_df, x= 'cell', y='mean_rate', hue='shuffling',
            ci='sd', capsize=0.2, errwidth=(2))




import copy
merged_df = copy.deepcopy(means_df)
merged_df['cell & tuning'] = merged_df['cell'] +' '+ merged_df['tuning']


merged_df.drop(merged_df[merged_df['cell & tuning']=='grid no-feedback'].index, inplace=True)
merged_df.drop(merged_df[merged_df['cell & tuning']=='grid no-feedforward'].index, inplace=True)
merged_df.drop(merged_df[merged_df['cell & tuning']=='grid disinhibited'].index, inplace=True)


ax = sns.catplot(data=merged_df, kind='bar', x='cell & tuning', y='mean_rate',
                 hue='shuffling', ci='sd', capsize=0.2, errwidth=2)

ax = sns.swarmplot(data=merged_df, x='cell & tuning', y='mean_rate',
                   hue='shuffling', color='black', dodge=True)

ax.get_legend().set_visible(False)

plt.title('Mean firing rate')
plt.tight_layout()



with pd.ExcelWriter('mean_firing_rates.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='Mean Firing Rates (Hz)')


# =============================================================================
# =============================================================================
# # Mean Rates
# =============================================================================
# =============================================================================

