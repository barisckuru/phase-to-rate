"""
Created on Mon Jan 17 10:09:58 2022

@author: daniel
"""

import shelve
import os
from tempotron.main import Tempotron
from phase_to_rate import information_measure
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import sys
import multiprocessing
import argparse
import uuid
import statsmodels.api as sm
import pingouin as pg
import elephant.statistics
from copy import deepcopy

"""Parse Command Line inputs"""
pr = argparse.ArgumentParser(description='Local pattern separation paradigm')
pr.add_argument('-grid_seed',
                type=int,
                help='Which grid seed to process',
                default=11,
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
epochs = 200  # Number of learning epochs
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
    dirname, 'data', 'tempotron', 'full', 'collective', 
    f'grid-seed_duration_shuffling_tuning_trajs_{grid_seed}_2000_{shuffling}_full_{trajectory_1}-{trajectory_2}')
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

training_result = tempotron.train(all_spikes, epochs, learning_rate=learning_rate)
pre_loss = training_result[1][0]
trained_loss = training_result[1][-1]
trained_accuracy = tempotron.accuracy(all_spikes)
tempotron.plot_membrane_potential(0, total_time, all_spikes[0][0])

"""Calculate Skaggs"""
spike_times1_clean = np.array(spike_times1, dtype=object)[:,0]

spike_times1_clean_inverted = [[] for x in range(200)]
for idx in range(200):
    spike_times1_clean_inverted[idx] = [x[idx] for x in spike_times1_clean]

spike_times1_ff = np.array([elephant.statistics.fanofactor(x) for x in spike_times1_clean_inverted])

spike_times1_aggregate = [[] for x in range(200)]
for idx_tr, tr in enumerate(spike_times1_clean):
    for idx_cell, cell in enumerate(tr):
        for spike in cell:
            spike_times1_aggregate[idx_cell].append(spike)
            
for idx, x in enumerate(spike_times1_aggregate):
    spike_times1_aggregate[idx].sort()


spike_times1_skaggs = information_measure.skaggs_information(spike_times1_aggregate, 2000, 100, agg=False)

spike_times1_aggregate_n_spikes = np.array([len(x) for x in spike_times1_aggregate])

spike_times2_clean = np.array(spike_times2, dtype=object)[:,0]

spike_times2_clean_inverted = [[] for x in range(200)]
for idx in range(200):
    spike_times2_clean_inverted[idx] = [x[idx] for x in spike_times2_clean]

spike_times2_ff = np.array([elephant.statistics.fanofactor(x) for x in spike_times2_clean_inverted])

spike_times2_aggregate = [[] for x in range(200)]
for idx_tr, tr in enumerate(spike_times2_clean):
    for idx_cell, cell in enumerate(tr):
        for spike in cell:
            spike_times2_aggregate[idx_cell].append(spike)
            
for idx, x in enumerate(spike_times2_aggregate):
    spike_times2_aggregate[idx].sort()

spike_times2_skaggs = information_measure.skaggs_information(spike_times2_aggregate, 2000, 100, agg=False)

spike_times2_aggregate_n_spikes = np.array([len(x) for x in spike_times2_aggregate])

all_spikes_n = spike_times1_aggregate_n_spikes + spike_times2_aggregate_n_spikes

"""Calculate ISI"""
spike_times1_isi = deepcopy(spike_times1_clean)

for seed_idx, seed in enumerate(spike_times1_clean):
    for cell_idx, cell in enumerate(spike_times1_clean[seed_idx]):
        spike_times1_isi[seed_idx][cell_idx] = np.diff(spike_times1_clean[seed_idx][cell_idx])

spike_times1_isi_aggregate = [[] for x in range(200)]
for idx_tr, tr in enumerate(spike_times1_isi):
    for idx_cell, cell in enumerate(tr):
        for spike in cell:
            spike_times1_isi_aggregate[idx_cell].append(spike)

spike_times1_isi_aggregate_mean = np.array([np.array(x).mean() for x in spike_times1_isi_aggregate])

plt.figure()
plt.scatter(spike_times1_skaggs[spike_times1_aggregate_n_spikes > 8], spike_times1_aggregate_n_spikes[spike_times1_aggregate_n_spikes > 8])
plt.xlabel("Skaggs Trajectory 1")
plt.ylabel("# Spikes")

skaggs_stats = pg.linear_regression(spike_times1_skaggs[spike_times1_aggregate_n_spikes > 8],
                                    tempotron.efficacies[spike_times1_aggregate_n_spikes > 8],
                                    add_intercept = True)

plt.figure()
plt.scatter(spike_times1_skaggs[spike_times1_aggregate_n_spikes > 8],
            tempotron.efficacies[spike_times1_aggregate_n_spikes > 8],
            alpha = 0.8)
plt.plot([0, spike_times1_skaggs[spike_times1_aggregate_n_spikes > 8].max()],
          [skaggs_stats['coef'][0], spike_times1_skaggs[spike_times1_aggregate_n_spikes > 8].max() * skaggs_stats['coef'][1] + skaggs_stats['coef'][0]],
          color='k')
plt.xlabel("Skaggs Trajectory 1")
plt.ylabel("Tempotron Efficacies")

ff_stats = pg.linear_regression(spike_times1_ff[spike_times1_aggregate_n_spikes > 8],
                                    tempotron.efficacies[spike_times1_aggregate_n_spikes > 8],
                                    add_intercept = True)

plt.figure()
plt.scatter(spike_times1_ff[spike_times1_aggregate_n_spikes > 8], tempotron.efficacies[spike_times1_aggregate_n_spikes > 8])
plt.plot([0, spike_times1_ff[spike_times1_aggregate_n_spikes > 8].max()],
          [ff_stats['coef'][0], spike_times1_ff[spike_times1_aggregate_n_spikes > 8].max() * ff_stats['coef'][1] + ff_stats['coef'][0]],
          color='k')
plt.xlabel("Fano Factor Trajectory 1")
plt.ylabel("Tempotron Efficacies")

n_spikes_stats = pg.linear_regression(spike_times1_aggregate_n_spikes[spike_times1_aggregate_n_spikes > 8],
                                    tempotron.efficacies[spike_times1_aggregate_n_spikes > 8],
                                    add_intercept = True)

plt.figure()
plt.scatter(spike_times1_aggregate_n_spikes[spike_times1_aggregate_n_spikes > 8], tempotron.efficacies[spike_times1_aggregate_n_spikes > 8])
plt.plot([0, spike_times1_aggregate_n_spikes[spike_times1_aggregate_n_spikes > 8].max()],
          [n_spikes_stats['coef'][0], spike_times1_aggregate_n_spikes[spike_times1_aggregate_n_spikes > 8].max() * n_spikes_stats['coef'][1] + n_spikes_stats['coef'][0]],
          color='k')
plt.xlabel("# Spikes Trajectory 1")
plt.ylabel("Tempotron Efficacies")


isi_stats = pg.linear_regression(spike_times1_isi_aggregate_mean[spike_times1_aggregate_n_spikes > 8],
                                    tempotron.efficacies[spike_times1_aggregate_n_spikes > 8],
                                    add_intercept = True, remove_na=True)

plt.figure()
plt.scatter(spike_times1_isi_aggregate_mean[spike_times1_aggregate_n_spikes > 8], tempotron.efficacies[spike_times1_aggregate_n_spikes > 8])
plt.plot([0, np.nanmax(spike_times1_isi_aggregate_mean[spike_times1_aggregate_n_spikes > 8])],
          [isi_stats['coef'][0], np.nanmax(spike_times1_isi_aggregate_mean[spike_times1_aggregate_n_spikes > 8]) * isi_stats['coef'][1] + isi_stats['coef'][0]],
          color='k')
plt.xlabel("mean ISI")
plt.ylabel("Tempotron Efficacies")


"""Save results to database"""
"""
grid_seed, duration, shuffling, network = example_data.split(os.sep)[-1].split("_")[-4:]

db_path = os.path.join(
    dirname, 'data', 'tempotron_thresholds_mean.db')
con = sqlite3.connect(db_path)
cur = con.cursor()
file_id = str(uuid.uuid4())

cur.execute(fr"INSERT INTO tempotron_run VALUES 
            ({seed}, {epochs},{total_time},{V_rest},{tau},{tau_s},{tempotron.threshold},
             {learning_rate},{n_cells},{trajectory_1},{trajectory_2},{pre_accuracy},
             {trained_accuracy}, {pre_loss}, {trained_loss}, {pre_loss-trained_loss},
             {float(trajectory_1) - float(trajectory_2)}, {grid_seed}, {duration}, 
             '{shuffling}', '{network}', '{cell_type}', '{file_id}')
            ")

con.commit()
con.close()
"""

"""Save loss and accuracy to file"""
"""
array_path = os.path.join(
    dirname, 'data', 'arrays')
os.makedirs(array_path, exist_ok=True)
array_file = os.path.join(array_path, file_id)
np.save(array_file, np.array(training_result))
"""

