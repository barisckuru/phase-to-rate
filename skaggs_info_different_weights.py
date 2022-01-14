#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:21:33 2021

@author: baris
"""

import shelve
import numpy as np
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_spikes(grid_seed, network_type, shuffling, pp_weight):
    trajectories, p = [75], 100  # In cm
    poisson_seeds = np.arange(p, p + 20, 1)
    poisson_seeds = list(poisson_seeds)
    dur_ms = 2000
    save_dir = (f'/home/baris/results/different_weight/{network_type}/seperate/seed_'
                + str(grid_seed) + '/'+str(pp_weight)+'/')
    fname = glob.glob(os.path.join(save_dir, f'*_{shuffling}*.dat'))[0][0:-4]
    print(f'{network_type}, {shuffling}, {pp_weight}')
    with shelve.open(fname) as storage:
        grid_spikes = storage['grid_spikes'][75]
        granule_spikes = storage['granule_spikes'][75]
    return grid_spikes, granule_spikes

def aggr_n_filter (all_spikes, cell, threshold, trajectory=75):
    poisson_seeds = range(100,120)
    if cell == 'grid_spikes':
        n_cell = 200
    elif cell == 'granule_spikes':
        n_cell = 2000
    agg_spikes = [[] for _ in range(n_cell)]
    for poiss in poisson_seeds:
        for c in range(n_cell):
            agg_spikes[c]+= list(all_spikes[poiss][c])
            agg_spikes[c].sort()
    filtered_cells = []
    for cell in range(n_cell):
        if len(agg_spikes[cell])>threshold:
            filtered_cells.append(agg_spikes[cell])
    return filtered_cells, agg_spikes


def skaggs_information(spike_times, dur_ms, time_bin_size,
                        phase_bin_size=360, theta_bin_size=100):

    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    time_bin_s = time_bin_size/1000
    n_time_bins = int(dur_ms/time_bin_size)
    theta_bin_size_s = theta_bin_size/1000
    skaggs_all = np.zeros(n_cell)
    if phase_bin_size == 360:
        rates = np.zeros((n_cell, n_time_bins))
        for cell in range(n_cell):
            spikes = np.array(spike_times[cell])
            phases = [[] for _ in range(n_time_bins)]
            skaggs = np.zeros((n_time_bins))
            times = np.arange(0, dur_ms+time_bin_size, time_bin_size)
            for j, time in enumerate(times):
                if j == times.shape[0]-1:
                    break
                count = np.logical_and(spikes > time, spikes < times[j+1]).sum()
                rates[cell, j] = count/time_bin_s
            mean_rate = np.mean(rates[cell, :])
    
            for j, time in enumerate(times):
                if j == times.shape[0]-1:
                    break
                rate = rates[cell, j]
                info = (rate/mean_rate)*(np.log2(rate/mean_rate))
                if info == info: 
                    skaggs[j] = info
            skaggs_all[cell] = (1/(n_time_bins))*np.sum(skaggs)
        skaggs_info = np.mean(skaggs_all)
    else:
        n_phase_bins = int(360/phase_bin_size)
        rates = np.zeros((n_cell, n_phase_bins, n_time_bins))
        for cell in range(n_cell):
            spikes = np.array(spike_times[cell])
            phases = [[] for _ in range(n_time_bins)]
            skaggs = np.zeros((n_phase_bins, n_time_bins))
            times = np.arange(0, dur_ms+time_bin_size, time_bin_size)
            for j, time in enumerate(times):
                if j == times.shape[0]-1:
                    break
                curr_train = spikes[np.logical_and(spikes > time,
                                                  spikes < times[j+1])]
                if curr_train.size > 0:
                    phases[j] = list(curr_train % (theta_bin_size) / (theta_bin_size)*360)
            for i in range(n_phase_bins):
                for j, phases_in_time in enumerate(phases):
                    phases_in_time = np.array(phases_in_time)
                    count = ((phase_bin_size*(i) < phases_in_time) &
                                    (phases_in_time < phase_bin_size*(i+1))).sum()
                    rate = count*((1/theta_bin_size_s)*n_phase_bins)
                    rates[cell, i, j] = rate
            for j, phases_in_time in enumerate(phases):
                mean_rate = np.mean(rates[cell, :, j])
                for i in range(n_phase_bins):
                    rate = rates[cell, i, j]
                    info = (rate/mean_rate)*(np.log2(rate/mean_rate))
                    if info == info: 
                        skaggs[i, j] = info
            skaggs_all[cell] = (1/(n_phase_bins*n_time_bins))*np.sum(skaggs)
        # skaggs_info = np.sum(skaggs_all)
        skaggs_info = np.mean(skaggs_all)
    return skaggs_info

# weights_full = [0.00105, 0.00145, 0.0020, 0.0022]
# weights_noff = [0.00083, 0.0015, 0.0021, 0.00235]
# weights_nofb = [0.00074, 0.00085, 0.00095, 0.00105]
# weights_dininhibited = [0.00065, 0.00073, 0.00082, 0.00086]


weights = {}
weights['full'] = [0.0009, 0.0012, 0.0017, 0.0025] 
weights['no-feedforward'] = [0.00075, 0.0009, 0.00097, 0.00175] #0.0007
weights['no-feedback'] = [0.0007, 0.00078, 0.0009,  0.00111] # 0.00065
weights['disinhibited'] = [0.00061, 0.00068, 0.00077, 0.0009]

weights = {}
weights['full'] = [0.0009, 0.00105, 0.0012, 0.00145, 0.0017, 0.0020, 0.0022, 0.0025]
weights['no-feedforward'] = [0.0007, 0.00075, 0.00083, 0.0009, 0.00097, 0.0015, 0.00175, 0.0021, 0.00235]
weights['no-feedback'] = [0.00066, 0.0007, 0.00074, 0.00078, 0.00085, 0.0009, 0.00095, 0.00105, 0.00111]
weights['disinhibited'] = [0.00059, 0.00061, 0.00065, 0.00068, 0.00073, 0.00077, 0.00082, 0.00086, 0.0009]

tuned = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
shuffling = ['non-shuffled', 'shuffled']
grid_seeds = [1,2,3,4,5,6,7,8,9,10]

time_bin = 250
dur_ms = 2000
n_poisson = 20
spatial_bin = (time_bin/1000)*20
threshold = int(dur_ms/time_bin)
# threshold = 16

all_skaggs = []
for grid_seed in grid_seeds:
    for i, tuning in enumerate(tuned):
        print(i)
        weight=weights[tuning]
        print(weight)
        for w in weight:
            for s in shuffling:
                curr_grid, curr_granule = load_spikes(grid_seed,
                                                      tuning, s, w)
                grid_filtered, _ = aggr_n_filter(curr_grid, 'grid_spikes',
                                              threshold)
                gra_filtered, gra_unfiltered = aggr_n_filter(curr_granule,
                                                             'granule_spikes',
                                                             threshold)
                
                counts = [len(cell) for cell in gra_unfiltered]
                curr_gra_mean = np.mean(counts)/(n_poisson*dur_ms/1000)
                
                curr_skaggs_grid = skaggs_information(grid_filtered, dur_ms, time_bin,
                                                       phase_bin_size=360,
                                                       theta_bin_size=100)
                curr_skaggs_gra = skaggs_information(gra_filtered, dur_ms, time_bin,
                                                       phase_bin_size=360,
                                                       theta_bin_size=100)
                
                data_row = [curr_skaggs_grid, curr_skaggs_gra, curr_gra_mean, tuning, s, grid_seed, w]
                all_skaggs.append(data_row)



df = pd.DataFrame(all_skaggs, columns=['grid info',
                                       'info (bits/AP)',
                                       'mean rate',
                                       'tuning',
                                       'shuffling',
                                       'grid_seed',
                                       'weight'])

df['mean rate'] = df['mean rate'].astype('float')
df['info (bits/AP)'] = df['info (bits/AP)'].astype('float')



df_full = df.loc[df['tuning'] == 'full']
df_noff = df.loc[df['tuning'] == 'no-feedforward']
df_nofb = df.loc[df['tuning'] == 'no-feedback']
df_disinh = df.loc[df['tuning'] == 'disinhibited']

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.set_title('full')
ax2.set_title('no feedforward')
ax3.set_title('no feedback')
ax4.set_title('disinhibited')
sns.scatterplot(x='mean rate', y='info (bits/AP)', hue='shuffling', ax=ax1, data=df_full)
sns.scatterplot(x='mean rate', y='info (bits/AP)', hue='shuffling', ax=ax2, data=df_noff)
sns.scatterplot(x='mean rate', y='info (bits/AP)', hue='shuffling', ax=ax3, data=df_nofb)
sns.scatterplot(x='mean rate', y='info (bits/AP)', hue='shuffling', ax=ax4, data=df_disinh)


f, ax = plt.subplots()
realistic_means = df.loc[(df['mean rate']>0.2) & (df['mean rate']<0.3)]
# realistic_means = realistic_means.sort_values(by=['tuning', 'shuffling']). reset_index()
sns.violinplot(x='tuning', y='info (bits/AP)', hue='shuffling', ax=ax, data=realistic_means)
plt.title('Info in differently tuned networks with mean between 0.2-0.3 ')

df = pd.DataFrame(all_skaggs, columns=['grid info',
                                       'increased info (bits/AP)',
                                       'increase in mean rate',
                                       'tuning',
                                       'shuffling',
                                       'grid_seed',
                                       'weight'])

df['increase in mean rate'] = df['increase in mean rate'].astype('float') - 0.25
df['increased info (bits/AP)'] =  df['increased info (bits/AP)'].astype('float') - 0.95





df_full = df.loc[df['tuning'] == 'full']
df_noff = df.loc[df['tuning'] == 'no-feedforward']
df_nofb = df.loc[df['tuning'] == 'no-feedback']
df_disinh = df.loc[df['tuning'] == 'disinhibited']

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharey=True, sharex=True)
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.set_title('full')
ax2.set_title('no feedforward')
ax3.set_title('no feedback')
ax4.set_title('disinhibited')
sns.scatterplot(x='increase in mean rate', y='increased info (bits/AP)', hue='shuffling', ax=ax1, data=df_full)
sns.scatterplot(x='increase in mean rate', y='increased info (bits/AP)', hue='shuffling', ax=ax2, data=df_noff)
sns.scatterplot(x='increase in mean rate', y='increased info (bits/AP)', hue='shuffling', ax=ax3, data=df_nofb)
sns.scatterplot(x='increase in mean rate', y='increased info (bits/AP)', hue='shuffling', ax=ax4, data=df_disinh)

















df_nonshuffled = df.loc[df['shuffling'] == 'non-shuffled']
df_shuffled = df.loc[df['shuffling'] == 'shuffled']
sns.lmplot(x='mean rate', y='info (bits/AP)', hue='tuning', data=df_nonshuffled)
plt.title('Nonshuffled Skaggs information with changing firing rate')


sns.lmplot(x='mean rate', y='info (bits/AP)', hue='tuning', data=df_shuffled)
plt.title('Shuffled Skaggs information with changing firing rate')

tunings = ['full', 'full', 'noff','noff', 'nofb', 'nofb', 'disinhibited', 'disinhibited']
shufflings = 4*['non-shuffled', 'shuffled']


data = np.stack((granule_counts, skaggs_granule, shufflings, tunings), axis=1)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(data, columns=['mean rate', 'info (bits/AP)', 'shuffling', 'tunings'])

df['mean rate'] = df['mean rate'].astype('float')
df['info (bits/AP)'] = df['info (bits/AP)'].astype('float')

df_nonshuffled = df.loc[df['shuffling'] == 'non-shuffled']
df_shuffled = df.loc[df['shuffling'] == 'shuffled']
fig, ax = plt.subplots()
sns.catplot(x='mean rate', y='info (bits/AP)', hue='tunings', data=df_nonshuffled, ax=ax, kind='bar')

fig2, ax2 = plt.subplots()
sns.catplot(x='tunings', y='info (bits/AP)', hue='shuffling', data=df_shuffled, ax=ax2, kind='bar')


for i in range(2):
    



curr_grid_spikes, curr_granule_spikes = load_spikes(grid_seed,
                                                        'full', 'shuffled',
                                                        0.0009)


s_full_grid_spikes, s_full_granule_spikes = load_spikes(grid_seed,
                                                        'full', 'shuffled',
                                                        0.0009)
    

aggregated = aggr_n_filter(grid_spikes, 'grid_spikes', 8)
aggregated = aggr_n_filter(granule_spikes, 'granule_spikes', 8)

