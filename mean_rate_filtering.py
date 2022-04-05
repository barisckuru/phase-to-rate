# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 13:20:49 2022

@author: Daniel
"""
import numpy as np
import os
import shelve
import pandas as pd
import sys


dirname = os.path.dirname(__file__)
path = os.path.join(
    dirname, 'data', 'tempotron')
files = [x for x in os.listdir(path) if 'grid-seed' in x and '.dat' in x]


grid_seed = []
shuffling = []
network = []
mean_rate = []
for f in files:
    f = f.split('.')[0]
    curr_path = os.path.join(path, f)
    curr_data = shelve.open(curr_path)
    averages = []
    for k1 in curr_data:
        for k2 in curr_data[k1]['granule_spikes']:
            n_s = [len(x) / 2 for x in curr_data[k1]['granule_spikes'][k2]]
            curr_avg = np.array(n_s).mean()
            averages.append(curr_avg)
    file_average = np.array(averages).mean()
    f_split = f.split('_')
    grid_seed.append(f_split[4])
    shuffling.append(f_split[6])
    network.append(f_split[7])
    mean_rate.append(file_average)

d = {'grid_seed': grid_seed,
     'shuffling': shuffling,
     'network': network,
     'mean_rate': mean_rate}
df = pd.DataFrame(data=d)

out_path = os.path.join(dirname, 'data', 'mean_rates.csv')
df.dropna(inplace=True)
df.to_csv(out_path)
