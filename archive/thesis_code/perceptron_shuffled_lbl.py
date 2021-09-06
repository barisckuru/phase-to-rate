#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:00:50 2020

@author: bariskuru
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from poiss_inp_gen_short_traj import inhom_poiss
from scipy import interpolate
from skimage.measure import profile_line
import os
from pearsonr_ct_bin import ct_a_bin, pearson_r

#BUILD THE NETWORK

class Net(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)
    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        return y

#TRAIN THE NETWORK

def train_net(net, train_data, train_labels, n_iter=1000, lr=1e-4):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    track_loss = []
    loss_fn = nn.MSELoss()
    for i in range(n_iter):
        out = net(train_data)
        loss = torch.sqrt(loss_fn(out, labels))
        
        # Compute gradients
        optimizer.zero_grad()
        loss.backward()
    
        # Update weights
        optimizer.step()
    
        # Store current value of loss
        track_loss.append(loss.item())  # .item() needed to transform the tensor output of loss_fn to a scalar
        
        # Track progress
        if (i + 1) % (n_iter // 5) == 0:
          print(f'iteration {i + 1}/{n_iter} | loss: {loss.item():.3f}')

    return track_loss, out

#Count the number of spikes in bins 

def binned_ct(arr, bin_size_ms, dt_ms=25, time_ms=5000):
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    n_traj = arr.shape[1]
    counts = np.empty((n_bins, n_cells, n_traj))
    for i in range(n_bins):
        for index, value in np.ndenumerate(arr):
            counts[i][index] = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            #search and count the number of spikes in the each bin range
    return counts

#Parametersfor the grid cell poisson input generation
savedir = os.getcwd()
n_grid = 200 
max_rate = 20
seed = 100
dur_ms = 2000
bin_size = 100
n_bin = int(dur_ms/bin_size)
dur_s = int(dur_ms/1000)
speed_cm = 20
field_size_cm = 100
traj_size_cm = dur_s*speed_cm

seed_1s = np.arange(100,105,1)
grids = grid_population(n_grid, max_rate, seed=seed_1s[0], arr_size=200)


def spike_ct(par_trajs):

    seed_2s = np.arange(200,210,1)
    n_traj = par_trajs.shape[0]
    poiss_spikes = []
    # counts_750 = np.empty((len(seed_2s), n_bin*n_grid))
    counts_745 = np.empty((len(seed_2s), n_bin*n_grid))
    for idx, seed_2 in enumerate(seed_2s):
        par_trajs_pf, dt_s = draw_traj(grids, n_grid, par_trajs, arr_size=200, field_size_cm = field_size_cm, dur_ms=dur_ms, speed_cm=speed_cm)
        curr_spikes = inhom_poiss(par_trajs_pf, n_traj, seed=seed_2, dt_s=dt_s, traj_size_cm=traj_size_cm)
        poiss_spikes.append(curr_spikes)
        # counts_750[idx, :] = binned_ct(curr_spikes, bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_745[idx,:] = binned_ct(curr_spikes, bin_size, time_ms=dur_ms)[:,:,1].flatten()
    # counts = np.vstack((counts_750, counts_745))
    return counts_745


sim_traj_cts = spike_ct(np.array([75, 74.5]))
plt.imshow(sim_traj_cts, aspect='auto')
diff_traj_cts = spike_ct(np.array([75, 60]))

data_sim = torch.FloatTensor(sim_traj_cts)
data_diff = torch.FloatTensor(diff_traj_cts)
labels = np.array([[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],[0, 1],[0, 1],[0, 1],[0, 1],[0, 1]])
np.random.shuffle(labels)


labels = torch.FloatTensor([[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],
                            [0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]) 

lr = 1e-3
n_iter = 200
seed_4s = [0,1,2,3,4,5,6,7,8,9]
plt.figure()
plt.title('Learning for different Poisson inputs from same the Trajectory(75cm) \n same grid pop, diff poiss seeds \n10 diff torch seeds, learning rate = '+str(lr))
plt.xlabel('Epochs')
plt.ylabel('RMSE Loss')


th_cross_sim = []
for seed_4 in seed_4s:
    torch.manual_seed(seed_4)
    net_sim = Net(4000,2)
    train_loss_sim, out_sim = train_net(net_sim, data_sim, labels, n_iter=n_iter, lr=lr)
    th_cross_sim.append(np.argmax(np.array(train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        plt.plot(train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        plt.plot(train_loss_sim, 'b-')
        
plt.annotate(str(th_cross_sim), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)


th_cross_diff = []
for seed_4 in seed_4s:
    torch.manual_seed(seed_4)
    net_diff = Net(4000,2)
    train_loss_diff, out_diff = train_net(net_diff, data_diff, labels, n_iter=n_iter, lr=lr)
    th_cross_diff.append(np.argmax(np.array(train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        plt.plot(train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        plt.plot(train_loss_diff, 'r-')
    
plt.legend()

plt.annotate(str(th_cross_sim)+'\n'+str(th_cross_diff), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)

print(list(net_diff.parameters())[0])

from scipy.stats.stats import pearsonr
