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
# =============================================================================
# # Load the data
# =============================================================================
# =============================================================================

    
    
def skaggs_information (spike_times, dur_ms, time_bin_size, phase_bin_size):
    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    time_bin_size_s = time_bin_size/1000
    n_phase_bins = int(360/phase_bin_size)
    n_time_bins = int(dur_s/time_bin_size_s)
    skaggs_all = np.zeros(n_cell)
    rates = np.zeros((n_cell, n_phase_bins, n_time_bins))
    for cell in range(n_cell):
        spikes = spike_times[cell]
        phases = [[] for _ in range(n_time_bins)]
        skaggs = np.zeros((n_phase_bins, n_time_bins))
        times = np.arange(0, dur_ms+time_bin_size, time_bin_size)
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = spikes[np.logical_and(spikes > time,
                                              spikes < times[j+1])]
            if curr_train.size > 0:
                phases[j] = list(curr_train % (time_bin_size) /
                                 (time_bin_size)*360)
        for i in range(n_phase_bins):
            for j, phases_in_time in enumerate(phases):
                phases_in_time = np.array(phases_in_time)
                count = ((phase_bin_size*(i) < phases_in_time) &
                                (phases_in_time < phase_bin_size*(i+1))).sum()
                rate = count*((1/time_bin_size_s)*(360/phase_bin_size))
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
        skaggs_info = np.mean(skaggs_all)
        mean_firing = np.mean(rates)
    return skaggs_info


ns_grid = all_spikes[1]['non-shuffled']['grid'][75][1]            
s_grid = all_spikes[1]['shuffled']['grid'][75][1]  
ns_granule = all_spikes[1]['non-shuffled']['granule'][75][1]  
s_granule = all_spikes[1]['shuffled']['granule'][75][1]  

ns_grid_skaggs= skaggs_information(ns_grid, 2000, 100, 18)
s_grid_skaggs= skaggs_information(s_grid, 2000, 100, 18)
ns_granule_skaggs= skaggs_information(ns_granule, 2000, 100, 18)
s_granule_skaggs= skaggs_information(s_granule, 2000, 100, 18)





trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

# trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
#                 71, 70, 69, 68, 67, 66, 65, 60]

n_samples = 20
grid_seeds = np.arange(1,11,1)

# grid_seeds = np.array([1])

tuning = 'full'

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
    
        ns_grid_skaggs.append(skaggs_information(ns_grid, 2000, 100, 18))
        s_grid_skaggs.append(skaggs_information(s_grid, 2000, 100, 18))
        ns_granule_skaggs.append(skaggs_information(ns_granule, 2000, 100, 18))
        s_granule_skaggs.append(skaggs_information(s_granule, 2000, 100, 18))
    print(f'grid seed {grid}')
    mean_ns_grid_skaggs.append(np.mean(ns_grid_skaggs))
    mean_s_grid_skaggs.append(np.mean(s_grid_skaggs))
    mean_ns_granule_skaggs.append(np.mean(ns_granule_skaggs))
    mean_s_granule_skaggs.append(np.mean(s_granule_skaggs))
    
all_skaggs = np.concatenate((mean_ns_grid_skaggs, mean_s_grid_skaggs,
                            mean_ns_granule_skaggs, mean_ns_granule_skaggs))
cell = 20*['grid']+20*['granule']
shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
all_skaggs= np.stack((all_skaggs, cell, shuffling), axis=1)

df_skaggs = pd.DataFrame(all_skaggs, columns=['info','cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')
sns.barplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))

plt.close('all')
sns.violinplot(data=df_skaggs, x='cell', y='info', hue='shuffling', 
            ci='sd', capsize=0.2, errwidth=(2))

plt.title('Skaggs Information')


            
I rather generate time bins instead of spatial bins bcs the spikes are in times, 
I can also translate spikes into locations since we have linear trajectories 
but it is unnecessary

    

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
            train = grid_model._randomize_grid_spikes(train, 100, time_ms=dur_ms)/1000
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

