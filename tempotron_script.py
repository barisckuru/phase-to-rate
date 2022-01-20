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

"""Parameters"""
seed = 0
np.random.seed(seed)
epochs = 100  # Number of learning epochs
total_time = 2000.0  # Simulation time
V_rest = 0.0  # Resting potential
tau = 10.0  # 
tau_s = 2.5
threshold = 24
learning_rate = 1e-4
n_cells = 20
efficacies = 1.8 * np.random.random(n_cells) - 0.50

trajectory_1 = '75'
trajectory_2 = '60'

"""Load data"""
dirname = os.path.dirname(__file__)
example_data = os.path.join(
    dirname, 'data', 'tempotron',
    'grid-seed_duration_shuffling_tuning_1_2000_non-shuffled_full')
data = shelve.open(example_data)

"""Structure and label spike times"""
spike_times1 = [(np.array(data[trajectory_1]['grid_spikes'][x][:n_cells], dtype=object), True) for x in data[trajectory_1]['grid_spikes']]
spike_times2 = [(np.array(data[trajectory_2]['grid_spikes'][x][:n_cells], dtype=object), False) for x in data[trajectory_2]['grid_spikes']]
all_spikes = np.array(spike_times1 + spike_times2, dtype=object)

print('synaptic efficacies:', efficacies, '\n')

tempotron = Tempotron(V_rest, tau, tau_s, efficacies, threshold, jit_mode=True, verbose=True)
print("Pre-training accuracy: " + 
      str(tempotron.accuracy(all_spikes)))
tempotron.plot_membrane_potential(0, total_time, all_spikes[0][0])

tempotron.train(all_spikes, epochs, learning_rate=learning_rate)
print(tempotron.efficacies)
print("Post-training accuracy: " + 
      str(tempotron.accuracy(all_spikes)))
tempotron.plot_membrane_potential(0, total_time, all_spikes[0][0])
plt.show()
