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
import uuid

"""Parse Command Line inputs"""
pr = argparse.ArgumentParser(description='Local pattern separation paradigm')
pr.add_argument('-grid_seed',
                type=int,
                help='Which grid seed to process',
                default=4,
                dest='grid_seed')
pr.add_argument('-shuffling',
                type=str,
                help='Process shuffled or shuffled',
                default='non-shuffled',
                dest='shuffling')

args = pr.parse_args()
grid_seed = args.grid_seed
shuffling = args.shuffling

"""Parameters"""
seed = 91
np.random.seed(seed)
epochs = 10  # Number of learning epochs
total_time = 2000.0 # Simulation time
V_rest = 0.0  # Resting potential
learning_rate = 1e-3
n_cells = 200
threshold = 15
tau = 10.0
tau_s = tau / 4.0
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
spike_times1 = [(np.array(data[trajectory_1][cell_type][x][:n_cells], dtype=object), False) for x in data[trajectory_1][cell_type]]
spike_times2 = [(np.array(data[trajectory_2][cell_type][x][:n_cells], dtype=object), True) for x in data[trajectory_2][cell_type]]
all_spikes = np.array(spike_times1 + spike_times2, dtype=object)

# Initialize synaptic efficiencies
efficacies = np.random.rand(n_cells)
print('synaptic efficacies:', efficacies, '\n')
tempotron = Tempotron(V_rest, tau, tau_s, efficacies,total_time, threshold, jit_mode=True, verbose=True)

"""Find the threshold"""
tmax = [tempotron.compute_tmax(sts[0]) for sts in all_spikes]
vmax = [tempotron.compute_membrane_potential(tmax[idx], sts[0]) for idx, sts in enumerate(all_spikes)]
thr = np.array(vmax).mean()
tempotron.threshold = thr

pre_accuracy = tempotron.accuracy(all_spikes)
tempotron.plot_membrane_potential(0, 2000, all_spikes[0][0])
# sys.exit()
training_result = tempotron.train(all_spikes, epochs, learning_rate=learning_rate)
pre_loss = training_result[1][0]
trained_loss = training_result[1][-1]
trained_accuracy = tempotron.accuracy(all_spikes)
tempotron.plot_membrane_potential(0, total_time, all_spikes[0][0])

"""Save results to database"""
grid_seed, duration, shuffling, network = example_data.split(os.sep)[-1].split("_")[-4:]

db_path = os.path.join(
    dirname, 'data', 'tempotron_thresholds_mean.db')
con = sqlite3.connect(db_path)
cur = con.cursor()
file_id = str(uuid.uuid4())
cur.execute(f"""INSERT INTO tempotron_run VALUES 
            ({seed}, {epochs},{total_time},{V_rest},{tau},{tau_s},{tempotron.threshold},
             {learning_rate},{n_cells},{trajectory_1},{trajectory_2},{pre_accuracy},
             {trained_accuracy}, {pre_loss}, {trained_loss}, {pre_loss-trained_loss},
             {float(trajectory_1) - float(trajectory_2)}, {grid_seed}, {duration}, 
             '{shuffling}', '{network}', '{cell_type}', '{file_id}')
            """)

con.commit()
con.close()

"""Save loss and accuracy to file"""
array_path = os.path.join(
    dirname, 'data', 'arrays')
os.makedirs(array_path, exist_ok=True)
array_file = os.path.join(array_path, file_id)
np.save(array_file, np.array(training_result))


# con = sqlite3.connect(db_path)
# cur = con.cursor()
# rows = cur.execute(f"""SELECT * FROM tempotron_run""")
"""         
for x in rows:
    print(x)
            
"""
