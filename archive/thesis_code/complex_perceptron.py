#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:13:01 2020

@author: baris
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os
from rate_n_phase_codes import phase_code, spike_ct

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
    # loss_fn = nn.L1Loss()
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


#Parametersfor the grid cell poisson input generation
savedir = os.getcwd()
n_grid = 200 
max_rate = 20
dur_ms = 2000
bin_size = 100
n_bin = int(dur_ms/bin_size)
dur_s = dur_ms/1000
speed_cm = 20
field_size_cm = 100
traj_size_cm = int(dur_s*speed_cm)
inp_len = 2*n_bin*n_grid

#Parameters for perceptron
lr = 1e-2
n_iter = 10000

#Initialize the figures


fig3, ax3 = plt.subplots()
ax3.set_title('Complex Phase&Rate Code Perceptron Loss '+str(dur_ms)+'ms \n multip torch seeds, learning rate = '+str(lr))
ax3.set_xlabel('Epochs')
ax3.set_ylabel('RMSE Loss')

#Seeds: seed1 for grids, seed2 for poiss spikes, seed4 for network
seed_1s = np.arange(100,120,1)
seed_2s = np.arange(200,205,1)
seed_4s = np.arange(0,20,1)

#Intialize empty arrays&lists to fill with data
sample_size = 2*seed_2s.shape[0]
n_sampleset = seed_4s.shape[0]
complex_code_sim = np.empty((sample_size, inp_len, n_sampleset))
complex_code_diff = np.empty((sample_size, inp_len, n_sampleset))


rate_phase_th_cross_sim = []
rate_phase_th_cross_diff = []

labels = torch.FloatTensor([[1, 0],[1, 0],[1, 0],[1, 0],[1, 0],
                            [0, 1],[0, 1],[0, 1],[0, 1],[0, 1]]) 

out_len = labels.shape[1]

for idx, seed_4 in enumerate(seed_4s):
    
    sim_traj = np.array([75, 74.5])
    diff_traj = np.array([75, 60])
    
    #Input generation
    phases_sim, rate_trajs_sim, dt_s = phase_code(sim_traj, dur_ms, seed_1s[idx], seed_2s)
    phases_diff, rate_trajs_diff, dt_s = phase_code(diff_traj, dur_ms, seed_1s[idx], seed_2s)

    sim_traj_cts = spike_ct(rate_trajs_sim, dur_ms)
    diff_traj_cts = spike_ct(rate_trajs_diff, dur_ms)

    complex_sim_y = sim_traj_cts*np.sin(phases_sim)
    complex_sim_x = sim_traj_cts*np.cos(phases_sim)
    complex_sim = np.concatenate((complex_sim_y, complex_sim_x), axis=1)
    complex_diff_y = diff_traj_cts*np.sin(phases_diff)
    complex_diff_x = diff_traj_cts*np.cos(phases_diff)
    complex_diff = np.concatenate((complex_diff_y, complex_diff_x), axis=1)
    #Normalization

    complex_sim = complex_sim/np.amax(complex_sim)
    complex_diff = complex_diff/np.amax(complex_diff)
    
    #fill arrays to save the data

    complex_code_sim[:,:,idx] = complex_sim
    complex_code_diff[:,:,idx] = complex_diff
    
    print('data done!')

    #Into tensor

    complex_sim = torch.FloatTensor(complex_sim)
    complex_diff = torch.FloatTensor(complex_diff)

    
    torch.manual_seed(seed_4)
    net_rate_phase_sim = Net(inp_len, out_len)
    rate_phase_train_loss_sim, rate_phase_out_sim = train_net(net_rate_phase_sim, complex_sim, labels, n_iter=n_iter, lr=lr)
    rate_phase_th_cross_sim.append(np.argmax(np.array(rate_phase_train_loss_sim) < 0.2))
    if seed_4 == seed_4s[0]:
        ax3.plot(rate_phase_train_loss_sim, 'b-', label='75cm vs 74.5cm')
    else:
        ax3.plot(rate_phase_train_loss_sim, 'b-')
        
    torch.manual_seed(seed_4)
    net_rate_phase_diff = Net(inp_len, out_len)
    rate_phase_train_loss_diff, rate_phase_out_diff = train_net(net_rate_phase_diff, complex_diff, labels, n_iter=n_iter, lr=lr)
    rate_phase_th_cross_diff.append(np.argmax(np.array(rate_phase_train_loss_diff) < 0.2))
    if seed_4 == seed_4s[0]:
        ax3.plot(rate_phase_train_loss_diff, 'r-', label='75cm vs 60cm')
    else:
        ax3.plot(rate_phase_train_loss_diff, 'r-')
  
ax3.legend()

ax3.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')

# plt.legend()

# plt.annotate(str(th_cross_sim)+'\n'+str(th_cross_diff), (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=9)
fname = 'complex_rate_n_phase_perceptron_norm_'+str(dur_ms)+'ms_'+str(n_iter)+'_iter_'+str(lr)+'_lr'
np.savez(fname, 
         complex_code_sim = complex_code_sim,
         complex_code_diff = complex_code_diff,
        
         rate_phase_th_cross_diff=rate_phase_th_cross_diff, 
         rate_phase_th_cross_sim=rate_phase_th_cross_sim,
        
         n_grid = n_grid, 
         max_rate = max_rate,
         dur_ms = dur_ms,
         bin_size = bin_size,
         n_bin = n_bin,
         dur_s = dur_s,
         speed_cm = speed_cm,
         field_size_cm = field_size_cm,
         traj_size_cm = traj_size_cm,
         inp_len = inp_len,
         lr = lr,
         n_iter = n_iter,
         sample_size = sample_size,
         n_sampleset = n_sampleset,
         labels = labels,
         sim_traj = sim_traj,
         diff_traj = diff_traj,
         seed_1s = seed_1s,
         seed_2s = seed_2s,
         seed_4s = seed_4s)


