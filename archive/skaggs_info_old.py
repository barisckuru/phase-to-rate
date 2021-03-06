#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:54:58 2021

@author: baris
"""


from neural_coding import load_spikes, rate_n_phase
from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
from scipy.spatial import distance
    

# =============================================================================
# skaggs info for rate-phase, mean of cells, mean of spatial bins, aggregated
# =============================================================================

def skaggs_information(spike_times, dur_ms, time_bin_size,
                        phase_bin_size=360, theta_bin_size=360):

    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    time_bin_s = time_bin_size/1000
    n_time_bins = int(dur_ms/time_bin_size)
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

# =============================================================================
# aggraegate spikes from poisson seeds
# =============================================================================

def aggr (all_spikes, shuffling, cell):
    grid_seeds = range(1,11)
    poisson_seeds = range(0,20)
    agg_spikes = []
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for grid in grid_seeds:
        spikes = [[] for _ in range(n_cell)]
        for poiss in poisson_seeds:
            for c in range(n_cell):
                spikes[c]+= list(all_spikes[grid][shuffling][cell][75][poiss][c])
                spikes[c].sort()
        agg_spikes.append(spikes)
    return agg_spikes


# =============================================================================
# filter insufficient cells
# =============================================================================

def filter_inact_granule(agg_spikes, threshold):
    filtered_cells = []
    n_cell = len(agg_spikes[0])
    n_grid = len(agg_spikes)
    for grid in range(n_grid):
        cells = []
        for cell in range(n_cell):
            # print(len(agg_spikes[grid][cell]))
            if len(agg_spikes[grid][cell])>threshold:
                cells.append(agg_spikes[grid][cell])
        filtered_cells.append(cells)
    return filtered_cells



# =============================================================================
# main function for all samples
# =============================================================================


grid_seeds = range(0,10)

phase_bin = 360
time_bin = 250
dur_ms = 2000

phase_bin_pi = phase_bin/180
spatial_bin = (time_bin/1000)*20

threshold = int((dur_ms/time_bin)*(360/phase_bin))

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

n_samples = 20
grid_seeds = np.arange(1,11,1)
tunes = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']

# =============================================================================
# load data
# =============================================================================
for tuning in tunes:
    all_spikes = {}
    for grid_seed in grid_seeds:
        path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
        
        # non-shuffled
        ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
        grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
        granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
        
        
        # shuffled
        s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
        s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
        s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
        
        print('shuffled path ok')
    
        all_spikes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
        all_spikes[grid_seed]["shuffled"] = {"grid": s_grid_spikes, "granule": s_granule_spikes}
        all_spikes[grid_seed]["non-shuffled"] = {"grid": grid_spikes, "granule": granule_spikes}
    
    
    all_ns_grid = aggr(all_spikes, 'non-shuffled', 'grid')
    all_ns_grid = filter_inact_granule(all_ns_grid, threshold)
    all_s_grid = aggr(all_spikes, 'shuffled', 'grid')
    all_s_grid = filter_inact_granule(all_s_grid, threshold)
    all_ns_granule = aggr(all_spikes, 'non-shuffled', 'granule')
    all_ns_granule = filter_inact_granule(all_ns_granule, threshold)
    all_s_granule = aggr(all_spikes, 'shuffled', 'granule')
    all_s_granule = filter_inact_granule(all_s_granule, threshold)
    
    ns_grid_skaggs = []
    s_grid_skaggs = []
    ns_granule_skaggs = []
    s_granule_skaggs = []
    
    for grid in grid_seeds:
        ns_grid = all_ns_grid[grid]
        s_grid = all_s_grid[grid]
        ns_granule = all_ns_granule[grid]
        s_granule = all_s_granule[grid]
        
        ns_grid_skaggs.append(skaggs_information(ns_grid, dur_ms, time_bin, phase_bin))
        s_grid_skaggs.append(skaggs_information(s_grid, dur_ms, time_bin, phase_bin))
        ns_granule_skaggs.append(skaggs_information(ns_granule, dur_ms, time_bin, phase_bin))
        s_granule_skaggs.append(skaggs_information(s_granule, dur_ms, time_bin, phase_bin))
        print(f'grid seed {grid}')
       
    all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                                ns_granule_skaggs, s_granule_skaggs))


    cell = 20*[tuning +' grid']+20*[tuning + ' granule']
    shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
    all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                                ns_granule_skaggs, s_granule_skaggs))
    skaggs_information = np.stack((all_skaggs, cell, shuffling), axis=1)
    
    
    if tuning == 'full':
        skaggs = skaggs_information
    else:
        skaggs = np.concatenate((skaggs, skaggs_information[20:, :]), axis=0)
        
phase_bin_pi = ', phase bin = ' + str(phase_bin_pi) + 'pi'
if int(phase_bin_pi) == 2:
    phase_bin_pi = ''
df_skaggs = pd.DataFrame(skaggs, columns=['info', 'cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
plt.close('all')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))
plt.title(f'Skaggs Information - Average of Population'
          +f'\n cells firing less than {threshold} spikes are filtered out'
          +f'\n 10 grid seeds, 20 poisson seeds aggregated,\n'
          +f'spatial bin = {spatial_bin} cm{phase_bin_pi}')



ns_grid_skaggs = []
s_grid_skaggs = []
ns_granule_skaggs = []
s_granule_skaggs = []

for grid in grid_seeds:
    ns_grid = all_ns_grid[grid]
    s_grid = all_s_grid[grid]
    ns_granule = all_ns_granule[grid]
    s_granule = all_s_granule[grid]
    
    ns_grid_skaggs.append(skaggs_information(ns_grid, dur_ms, time_bin, phase_bin))
    s_grid_skaggs.append(skaggs_information(s_grid, dur_ms, time_bin, phase_bin))
    ns_granule_skaggs.append(skaggs_information(ns_granule, dur_ms, time_bin, phase_bin))
    s_granule_skaggs.append(skaggs_information(s_granule, dur_ms, time_bin, phase_bin))
    print(f'grid seed {grid}')
   
all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                            ns_granule_skaggs, s_granule_skaggs))
cell = 20*['grid']+20*['granule']
shuffling = 2*(10*['non-shuffled']+10*['shuffled'])

all_skaggs= np.stack((all_skaggs, cell, shuffling), axis=1)

df_skaggs = pd.DataFrame(all_skaggs, columns=['info','cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
plt.close('all')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))
plt.title(f'{tuning} Skaggs Information - Average of Population'
          +f'\n cells firing less than {threshold} spikes are filtered out'
          +f'\n 10 grid seeds, 20 poisson seeds aggregated,\n'
          +f'spatial bin = {spatial_bin} cm, phase bin = {phase_bin_pi} pi'
          +f'\n #grid ~ {len(all_ns_grid[0])}, #granule ~ {len(all_ns_granule[0])}')






# labels = ((10*['EC non-shuffled']+10*['EC shuffled']) +
#           (10*['GC non-shuffled']+10*['GC shuffled']))
# tune = 40*[tuning + ]

cell = 20*[tuning +' grid']+20*[tuning + ' granule']
shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                            ns_granule_skaggs, s_granule_skaggs))
skaggs_information = np.stack((all_skaggs, cell, shuffling), axis=1)
import pickle 
with open(tuning + '_skaggs.pkl', 'wb') as f:
    pickle.dump(skaggs_information, f)
    
    
tunes = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
for t in tunes:
    if t == 'full':
        with open(t + '_skaggs.pkl', 'rb') as f:
            skaggs = pickle.load(f)
    else:
        with open(t + '_skaggs.pkl', 'rb') as f:
            skaggs = np.concatenate((skaggs, pickle.load(f)[20:,:]), axis=0)
            
        
df_skaggs = pd.DataFrame(skaggs, columns=['info', 'cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
plt.close('all')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))
plt.title(f'Skaggs Information - Average of Population'
          +f'\n cells firing less than {threshold} spikes are filtered out'
          +f'\n 10 grid seeds, 20 poisson seeds aggregated,\n'
          +f'spatial bin = {spatial_bin} cm, phase bin = {phase_bin_pi} pi')











# =============================================================================
# skaggs info for rate-phase, mean of cells, aggregated
# =============================================================================

def skaggs_information(spike_times, dur_ms, time_bin_size,
                        phase_bin_size, theta_bin_size=100):
    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    theta_bin_size_s = theta_bin_size/1000
    n_phase_bins = int(360/phase_bin_size)
    n_time_bins = int(dur_ms/time_bin_size)
    skaggs_all = np.zeros(n_cell)
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
        mean_rate = np.mean(rates[cell, :, :])
        for i in range(n_phase_bins):
            for j, phases_in_time in enumerate(phases):
                rate = rates[cell, i, j]
                info = (rate/mean_rate)*(np.log2(rate/mean_rate))
                if info == info: 
                    skaggs[i, j] = info
        skaggs_all[cell] = (1/(n_phase_bins*n_time_bins))*np.sum(skaggs)
    # skaggs_info = np.sum(skaggs_all)
    skaggs_info = np.mean(skaggs_all)
    return skaggs_info

# =============================================================================
# skaggs info average poisson seeds
# =============================================================================

def skaggs_information (spike_times, dur_ms, time_bin_size,
                        phase_bin_size, theta_bin_size=100):
    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    theta_bin_size_s = theta_bin_size/1000
    n_phase_bins = int(360/phase_bin_size)
    n_time_bins = int(dur_ms/time_bin_size)
    skaggs_all = np.zeros(n_cell)
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
        mean_rate = np.mean(rates[cell, :, :])
        # if mean_rate == 0:
        #     print('jajajaj')
        for i in range(n_phase_bins):
            for j, phases_in_time in enumerate(phases):
                rate = rates[cell, i, j]
                info = (rate/mean_rate)*(np.log2(rate/mean_rate))
                if info == info: 
                    skaggs[i, j] = info
        skaggs_all[cell] = (1/(n_phase_bins*n_time_bins))*np.sum(skaggs)
    # skaggs_info = np.sum(skaggs_all)
    skaggs_info = np.mean(skaggs_all)
    return skaggs_info


phase_bin = 18
time_bin = 100

phase_bin_pi = phase_bin/180
spatial_bin = (time_bin/1000)*20

ns_grid_skaggs = []
s_grid_skaggs = []
ns_granule_skaggs = []
s_granule_skaggs = []

mean_ns_grid_skaggs = []
mean_s_grid_skaggs = []
mean_ns_granule_skaggs = []
mean_s_granule_skaggs = []

grid_seeds = range(1,11)
poisson_seeds = range(0,20)



for grid in grid_seeds:
    for poiss in poisson_seeds:
        ns_grid = all_spikes[grid]['non-shuffled']['grid'][75][poiss]
        s_grid = all_spikes[grid]['shuffled']['grid'][75][poiss]
        ns_granule = all_spikes[grid]['non-shuffled']['granule'][75][poiss]
        s_granule = all_spikes[grid]['shuffled']['granule'][75][poiss]
    
        ns_grid_skaggs.append(skaggs_information(ns_grid, 2000, time_bin, phase_bin))
        s_grid_skaggs.append(skaggs_information(s_grid, 2000, time_bin, phase_bin))
        ns_granule_skaggs.append(skaggs_information(ns_granule, 2000, time_bin, phase_bin))
        s_granule_skaggs.append(skaggs_information(s_granule, 2000, time_bin, phase_bin))
    print(f'grid seed {grid}')
    mean_ns_grid_skaggs.append(np.mean(ns_grid_skaggs))
    mean_s_grid_skaggs.append(np.mean(s_grid_skaggs))
    mean_ns_granule_skaggs.append(np.mean(ns_granule_skaggs))
    mean_s_granule_skaggs.append(np.mean(s_granule_skaggs))
   
all_skaggs = np.concatenate((mean_ns_grid_skaggs, mean_s_grid_skaggs,
                            mean_ns_granule_skaggs, mean_s_granule_skaggs))
cell = 20*['grid']+20*['granule']
shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
all_skaggs= np.stack((all_skaggs, cell, shuffling), axis=1)

df_skaggs = pd.DataFrame(all_skaggs, columns=['info','cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
plt.close('all')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))
plt.title(f'Skaggs Information, \n 10 grid seeds, 20 poisson seeds averaged,\n'
          +f'spatial bin = {spatial_bin} cm, phase bin = {phase_bin_pi} pi')


# =============================================================================
# poisson seeds aggregated
# =============================================================================
 

def aggr (all_spikes, shuffling, cell):
    grid_seeds = range(1,11)
    poisson_seeds = range(0,20)
    agg_spikes = []
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for grid in grid_seeds:
        spikes = [[] for _ in range(n_cell)]
        for poiss in poisson_seeds:
            for c in range(n_cell):
                spikes[c]+= list(all_spikes[grid][shuffling][cell][75][poiss][c])
                spikes[c].sort()
        agg_spikes.append(spikes)
    return agg_spikes



all_ns_grid = aggr(all_spikes, 'non-shuffled', 'grid')
all_s_grid = aggr(all_spikes, 'shuffled', 'grid')
all_ns_granule = aggr(all_spikes, 'non-shuffled', 'granule')
all_s_granule = aggr(all_spikes, 'shuffled', 'granule')


grid_seeds = range(0,10)

phase_bin = 45
time_bin = 250

phase_bin_pi = phase_bin/180
spatial_bin = (time_bin/1000)*20

ns_grid_skaggs = []
s_grid_skaggs = []
ns_granule_skaggs = []
s_granule_skaggs = []

for grid in grid_seeds:
    ns_grid = all_ns_grid[grid]
    s_grid = all_s_grid[grid]
    ns_granule = all_ns_granule[grid]
    s_granule = all_s_granule[grid]
    
    ns_grid_skaggs.append(skaggs_information(ns_grid, 2000, time_bin, phase_bin))
    s_grid_skaggs.append(skaggs_information(s_grid, 2000, time_bin, phase_bin))
    ns_granule_skaggs.append(skaggs_information(ns_granule, 2000, time_bin, phase_bin))
    s_granule_skaggs.append(skaggs_information(s_granule, 2000, time_bin, phase_bin))
    print(f'grid seed {grid}')
   
all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                            ns_granule_skaggs, s_granule_skaggs))
cell = 20*['grid']+20*['granule']
shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
all_skaggs= np.stack((all_skaggs, cell, shuffling), axis=1)

df_skaggs = pd.DataFrame(all_skaggs, columns=['info','cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
plt.close('all')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))
plt.title(f'Skaggs Information, \n 10 grid seeds, 20 poisson seeds aggregated,\n'
          +f'spatial bin = {spatial_bin} cm, phase bin = {phase_bin_pi} pi')


# =============================================================================
# skaggs only rate
# =============================================================================




def skaggs_information (spike_times, dur_ms, time_bin_size):
    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    time_bin_s = time_bin_size
    n_time_bins = int(dur_ms/time_bin_size)
    skaggs_all = np.zeros(n_cell)
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
    skaggs_info = np.sum(skaggs_all)
    return skaggs_info


time_bin = 250
spatial_bin = (time_bin/1000)*20

ns_grid_skaggs = []
s_grid_skaggs = []
ns_granule_skaggs = []
s_granule_skaggs = []

mean_ns_grid_skaggs = []
mean_s_grid_skaggs = []
mean_ns_granule_skaggs = []
mean_s_granule_skaggs = []

grid_seeds = range(1,11)
poisson_seeds = range(0,20)


for grid in grid_seeds:
    for poiss in poisson_seeds:
        ns_grid = all_spikes[grid]['non-shuffled']['grid'][75][poiss]
        s_grid = all_spikes[grid]['shuffled']['grid'][75][poiss]
        ns_granule = all_spikes[grid]['non-shuffled']['granule'][75][poiss]
        s_granule = all_spikes[grid]['shuffled']['granule'][75][poiss]
    
        ns_grid_skaggs.append(skaggs_information(ns_grid, 2000, time_bin))
        s_grid_skaggs.append(skaggs_information(s_grid, 2000, time_bin))
        ns_granule_skaggs.append(skaggs_information(ns_granule, 2000, time_bin))
        s_granule_skaggs.append(skaggs_information(s_granule, 2000, time_bin))
    print(f'grid seed {grid}')
    mean_ns_grid_skaggs.append(np.mean(ns_grid_skaggs))
    mean_s_grid_skaggs.append(np.mean(s_grid_skaggs))
    mean_ns_granule_skaggs.append(np.mean(ns_granule_skaggs))
    mean_s_granule_skaggs.append(np.mean(s_granule_skaggs))
   
all_skaggs = np.concatenate((mean_ns_grid_skaggs, mean_s_grid_skaggs,
                            mean_ns_granule_skaggs, mean_s_granule_skaggs))
cell = 20*['grid']+20*['granule']
shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
all_skaggs= np.stack((all_skaggs, cell, shuffling), axis=1)

df_skaggs = pd.DataFrame(all_skaggs, columns=['info','cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
plt.close('all')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))
plt.title(f'Skaggs Information, \n 10 grid seeds, 20 poisson seeds averaged,\n'
          +f'spatial bin = {spatial_bin} cm')









# =============================================================================
# skaggs only rate aggregated
# =============================================================================

def aggr (all_spikes, shuffling, cell):
    grid_seeds = range(1,11)
    poisson_seeds = range(0,20)
    agg_spikes = []
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for grid in grid_seeds:
        spikes = [[] for _ in range(n_cell)]
        for poiss in poisson_seeds:
            for c in range(n_cell):
                spikes[c]+= list(all_spikes[grid][shuffling][cell][75][poiss][c])
                spikes[c].sort()
        agg_spikes.append(spikes)
    return agg_spikes



all_ns_grid = aggr(all_spikes, 'non-shuffled', 'grid')
all_s_grid = aggr(all_spikes, 'shuffled', 'grid')
all_ns_granule = aggr(all_spikes, 'non-shuffled', 'granule')
all_s_granule = aggr(all_spikes, 'shuffled', 'granule')


grid_seeds = range(1,10)

phase_bin = 45
time_bin = 250

phase_bin_pi = phase_bin/180
spatial_bin = (time_bin/1000)*20

ns_grid_skaggs = []
s_grid_skaggs = []
ns_granule_skaggs = []
s_granule_skaggs = []

for grid in grid_seeds:
    for poiss in poisson_seeds:
        ns_grid = all_ns_grid[grid]
        s_grid = all_s_grid[grid]
        ns_granule = all_ns_granule[grid]
        s_granule = all_s_granule[grid]
        
        ns_grid_skaggs.append(skaggs_information(ns_grid, 2000, time_bin))
        s_grid_skaggs.append(skaggs_information(s_grid, 2000, time_bin))
        ns_granule_skaggs.append(skaggs_information(ns_granule, 2000, time_bin))
        s_granule_skaggs.append(skaggs_information(s_granule, 2000, time_bin))
    print(f'grid seed {grid}')
   
all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                            ns_granule_skaggs, s_granule_skaggs))
cell = 20*['grid']+20*['granule']
shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
all_skaggs= np.stack((all_skaggs, cell, shuffling), axis=1)

df_skaggs = pd.DataFrame(all_skaggs, columns=['info','cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
plt.close('all')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))
plt.title(f'Skaggs Information, \n 10 grid seeds, 20 poisson seeds aggregated,\n'
          +f'spatial bin = {spatial_bin} cm')





def skaggs_information (spike_times, dur_ms, time_bin_size):
    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    time_bin_s = time_bin_size
    n_time_bins = int(dur_ms/time_bin_size)
    skaggs_all = np.zeros(n_cell)
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
            rate = rates[cell, j]
            info = (rate/mean_rate)*(np.log2(rate/mean_rate))
            if info == info: 
                skaggs[j] = info
        skaggs_all[cell] = (1/(n_time_bins))*np.sum(skaggs)
    skaggs_info = np.sum(skaggs_all)
    return skaggs_info