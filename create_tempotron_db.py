# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 12:51:20 2022

@author: Daniel
"""

import sqlite3
import os
import numpy as np

dirname = os.path.dirname(__file__)
db_path = os.path.join(
    dirname, 'data', 'tempotron.db')

con = sqlite3.connect(db_path)

cur = con.cursor()

seed = 0
np.random.seed(seed)
epochs = 100  # Number of learning epochs
total_time = 2000.0  # Simulation time
V_rest = 0.0  # Resting potential
tau = 10.0  # 
tau_s = 2.5
threshold = 7
learning_rate = 1e-4
n_cells = 20
efficacies = 1.8 * np.random.random(n_cells) - 0.50

trajectory_1 = '75'
trajectory_2 = '74'

cur.execute('''CREATE TABLE tempotron_run
            (tempotron_seed INT, epochs INT, time FLOAT, Vrest FLOAT, tau FLOAT,
             tau_s FLOAT, threshold FLOAT, learning_rate FLOAT, n_cells INT,
             trajectory_one FLOAT, trajectory_two, pre_accuracy FLOAT,
             trained_accuracy FLOAT, pre_loss FLOAT, trained_loss FLOAT,
             delta_loss FLOAT, distance FLOAT,
             grid_seed INT, duration FLOAT, shuffling VARCHAR(255), 
             network VARCHAR(255), cell_type VARCHAR(255)
             )''')
            

con.commit()

con.close()