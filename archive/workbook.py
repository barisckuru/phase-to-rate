#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:06:24 2021

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

from compress_pickle import dump, load



fname = '/home/baris/phase_coding/75-15_disinhibited_perceptron_speed_polar_inc.pkl'
with open(fname, 'rb') as handle:
    d = pickle.load(handle)



df_full = pd.DataFrame(a,
                  columns=['distance', 'speed', 'threshold_crossing',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'learning_rate'])


df_nofb = pd.DataFrame(b,
                  columns=['distance', 'speed', 'threshold_crossing',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'learning_rate'])
df_noff = pd.DataFrame(c,
                  columns=['distance', 'speed', 'threshold_crossing',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'learning_rate'])
df_disinh = pd.DataFrame(d,
                  columns=['distance', 'speed', 'threshold_crossing',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'learning_rate'])


with pd.ExcelWriter('perceptron_results.xlsx') as writer:
    df_full.to_excel(writer, sheet_name='full perceptron')
    df_nofb.to_excel(writer, sheet_name='nofb perceptron')
    df_noff.to_excel(writer, sheet_name='noff perceptron')
    df_disinh.to_excel(writer, sheet_name='disinhibited perceptron')

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
# grid_seeds = np.arange(1,11,1)

grid_seeds = np.array([1])

tuning = 'no-feedback'

granule_codes = np.zeros((80000, 20, 17, 4*len(grid_seeds)))
grid_codes = np.zeros((8000, 20, 17, 4*len(grid_seeds)))


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

    grid_codes[:, :, :, 4*(grid_seed-1)] = grid_rate_code
    grid_codes[:, :, :,4*(grid_seed-1)+1] = s_grid_rate_code
    grid_codes[:, :, :, 4*(grid_seed-1)+2] = grid_phase_code
    grid_codes[:, :, :, 4*(grid_seed-1)+3] = s_grid_rate_code
    granule_codes[:, :, :, 4*(grid_seed-1)] = granule_rate_code
    granule_codes[:, :, :, 4*(grid_seed-1)+1] = s_granule_rate_code
    granule_codes[:, :, :, 4*(grid_seed-1)+2] = granule_phase_code
    granule_codes[:, :, :, 4*(grid_seed-1)+3] = s_granule_phase_code
    



np.savez('all_codes_nofb_npz', grid_codes=grid_codes, granule_codes = granule_codes)

np.save('granule_codes_npz', granule_codes)
np.savez_compressed('all_codes_comp', grid_codes=grid_codes, granule_codes = granule_codes)

import pickle

with open('filename.pickle', 'wb') as handle:
    pickle.dump(all_codes, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)

print (all_codes == b)



trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

n_samples = 20
grid_seeds = np.arange(1,11,1)

tuning = 'full'
all_codes = {}
for grid_seed in grid_seeds:
    path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
    
    # non-shuffled
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    
    print('ns path ok')
       
    # shuffled
    s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
    s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
    s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
    
    print('shuffled path ok')
    
    all_codes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
    all_codes[grid_seed]["shuffled"] = {"grid": {}, "granule": {}}
    all_codes[grid_seed]["non-shuffled"] = {"grid": {}, "granule": {}}

    all_codes[grid_seed]['non-shuffled']['grid'] = {'spikes': grid_spikes}
    all_codes[grid_seed]['shuffled']['grid'] = {'spikes': s_grid_spikes}
    all_codes[grid_seed]['non-shuffled']['granule'] = {'spikes': granule_spikes}
    all_codes[grid_seed]['shuffled']['granule'] = {'spikes': s_granule_spikes}






import shelve
shelved = shelve.open('shelved_codes')
shelved['all_codes'] = all_codes





with open('comp_pickle_lzma', 'wb') as handle:
    dump(all_codes, handle, compression='lzma')
    
import time
start = time.time()
    
with open('comp_pickle_bz', 'wb') as handle:
    dump(all_codes, handle, compression='bz2')

stop = time.time()
print('time, sec, min, hour  ')
print(stop-start)
time_min = (stop-start)/60
time_hour = time_min/60
print(time_min)
print(time_hour)

import time
start = time.time()

with open('comp_pickle_bz', 'rb') as handle:
    loaded= load(handle, compression='bz2')
    
stop = time.time()
print('time, sec, min, hour  ')
print(stop-start)
time_min = (stop-start)/60
time_hour = time_min/60
print(time_min)
print(time_hour) 
    
loaded = load('comp_pickle_bz', )
