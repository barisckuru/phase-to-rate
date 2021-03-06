# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:09:58 2022

@author: daniel
"""

import shelve
import os
from tempotron.main import Tempotron
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import sys
import multiprocessing
import argparse

"""Parse Command Line inputs"""
pr = argparse.ArgumentParser(description='Local pattern separation paradigm')
pr.add_argument('-grid_seed',
                type=int,
                help='Which grid seed to process',
                default=4,
                dest='grid_seed')
pr.add_argument('-shuffling',
                type=str,
                help='Process shuffled or non-shuffled',
                default='non-shuffled',
                dest='shuffling')

args = pr.parse_args()
grid_seed = args.grid_seed
shuffling = args.shuffling

"""Parameters"""
seed = 91
np.random.seed(seed)
epochs = 200  # Number of learning epochs
total_time = 2000.0 # Simulation time
V_rest = 0.0  # Resting potential
learning_rate = 1e-3
n_cells = 2000
threshold = 50
tau = 10.0
tau_s = tau / 2.0
n_merge = 50
# efficacies = 1.8 * np.random.random(n_cells) - 0.50

trajectory_1 = '75'
trajectory_2 = '60'

cell_type = 'granule_spikes'
"""Load data"""
dirname = os.path.dirname(__file__)
example_data = os.path.join(
    dirname, 'data', 'tempotron',
    f'grid-seed_duration_shuffling_tuning_{grid_seed}_2000_{shuffling}_full')
data = shelve.open(example_data)

"""Structure and label spike times"""
spike_times1 = [(np.array(data[trajectory_1][cell_type][x][:n_cells], dtype=object), True) for x in data[trajectory_1][cell_type]]
spike_times2 = [(np.array(data[trajectory_2][cell_type][x][:n_cells], dtype=object), False) for x in data[trajectory_2][cell_type]]
all_spikes = np.array(spike_times1 + spike_times2, dtype=object)

"""Merge n granule cells"""
np.random.permutation(np.arange(n_cells))
for idx, pattern in enumerate(all_spikes):
    pattern_split = np.split(pattern[0], n_cells/n_merge)
    new_pattern = []
    for cluster in pattern_split:
        new_st = np.sort(np.concatenate(cluster))
        new_pattern.append(new_st)
    
    all_spikes[idx][0] = np.array(new_pattern, dtype=object)

# Initialize synaptic efficiencies
efficacies = np.random.rand(int(n_cells/n_merge))
print('synaptic efficacies:', efficacies, '\n')

tempotron = Tempotron(V_rest, tau, tau_s, efficacies,total_time, threshold, jit_mode=True, verbose=True)
pre_accuracy = tempotron.accuracy(all_spikes)
print("Pre-training accuracy: " + 
      str(pre_accuracy))
tempotron.plot_membrane_potential(0, total_time, all_spikes[0][0])
sys.exit()
training_result = tempotron.train(all_spikes, epochs, learning_rate=learning_rate)
pre_loss = training_result[1][0]
trained_loss = training_result[1][-1]
print(tempotron.efficacies)
trained_accuracy = tempotron.accuracy(all_spikes)
print("Post-training accuracy: " + 
      str(trained_accuracy))
tempotron.plot_membrane_potential(0, total_time, all_spikes[0][0])
plt.show()

"""Save results to database"""
grid_seed, duration, shuffling, network = example_data.split(os.sep)[-1].split("_")[-4:]

db_path = os.path.join(
    dirname, 'data', 'tempotron_merged_gcs.db')
con = sqlite3.connect(db_path)
cur = con.cursor()
cur.execute(f"""INSERT INTO tempotron_run VALUES 
            ({seed}, {epochs},{total_time},{V_rest},{tau},{tau_s},{threshold},
             {learning_rate},{n_cells},{trajectory_1},{trajectory_2},{pre_accuracy},
             {trained_accuracy}, {pre_loss}, {trained_loss}, {pre_loss-trained_loss},
             {float(trajectory_1) - float(trajectory_2)}, {grid_seed}, {duration}, 
             '{shuffling}', '{network}', '{cell_type}', '{file_id}', {n_merge})
            """)

con.commit()
con.close()


# con = sqlite3.connect(db_path)
# cur = con.cursor()
# rows = cur.execute(f"""SELECT * FROM tempotron_run""")
"""         
for x in rows:
    print(x)
            
"""
