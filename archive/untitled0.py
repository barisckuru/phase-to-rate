#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 12:36:35 2022

@author: baris
"""
import numpy as np
import os
import shelve

def load_spikes(path, cell_type, trajectories, n_samples):

    if not os.path.exists(path+'.dir'):
        print(path)
        raise Exception('Path does not exist!')

    storage = shelve.open(path)
    spikes = {}
    for traj in trajectories:
        requested_spikes = []
        traj_key = str(traj)
        poisson_seeds = storage[traj_key]["parameters"]["poisson_seeds"]
        if n_samples > len(poisson_seeds):
            raise Exception("Too much samples requested!")
        elif n_samples < 1:
            raise Exception("n_samples should be larger than 0!")
        else:
            poisson_seeds = poisson_seeds[0:n_samples]

        if cell_type == "grid":
            all_spikes = storage[traj_key]["grid_spikes"]
        elif cell_type == "granule":
            all_spikes = storage[traj_key]["granule_spikes"]
        else:
            raise Exception("Cell type does not exist!")
        for poisson in poisson_seeds:
            requested_spikes.append(all_spikes[poisson])
        spikes[traj] = requested_spikes
    storage.close()
    return spikes


path = '/home/baris/results/tempotron/adjusted_weight/no-feedback/collective/'
grid_spikes = []
granule_spikes = [] 
grid_seeds = range(1,31)
all_means = {}
ct_75 = 0
ct_60 = 0
for gseed in grid_seeds:
    fname = f'grid-seed_duration_shuffling_tuning_trajs_{gseed}_2000_non-shuffled_no-feedback-adjusted_75-60'
    load_dir = path+fname
    curr_gra_spikes = load_spikes(load_dir, 'granule', [75,60], 20)
    means_75 = []
    means_60 = []
    all_means[gseed] = {}
    for pseed in range(20):
        mean_75=[]
        mean_60=[]
        for cell in range(2000):
            mean_75.append(len(curr_gra_spikes[75][pseed][cell]))
            mean_60.append(len(curr_gra_spikes[60][pseed][cell]))
        means_75.append(np.mean(mean_75)/2)
        means_60.append(np.mean(mean_60)/2)
    all_means[gseed][75] = np.mean(means_75)
    all_means[gseed][60] = np.mean(means_60)
    if np.mean(means_75) <0.2:
        ct_75 +=1
        print(gseed)
    if np.mean(means_60) <0.3 and np.mean(means_60) > 0.2:
        ct_60 +=1
    
    
    
    
    
    
    
for gseed in grid_seeds:
    print(str(all_means[gseed][75]) + '   75')
    print(str(all_means[gseed][60]) + '   60')