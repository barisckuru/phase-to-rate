#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 12:34:47 2021

@author: baris
"""


import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os, glob
from rate_n_phase_code_gra import phase_code, overall_spike_ct
import time
import copy
from neuron import h, gui  # gui necessary for some parameters to h namespace
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim



#parameters
lr=1e-4
n_samples = 20
grid_s = 420


#labels, output for training the network, 5 for each trajectory



def label(n_poiss):
    a = np.tile([1, 0], (n_poiss,1))
    b = np.tile([0, 1], (n_poiss,1))
    labels = np.vstack((a,b))
    labels = torch.FloatTensor(labels) 
    out_len = labels.shape[1]
    return labels, out_len

#BUILD THE NETWORK

class Net(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)
    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        return y

#TRAIN THE NETWORK

def train_net(net, train_data, labels, n_iter=1000, lr=1e-4):
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



def shuffle_perceptron(shuffled_code, nonshuffled_code, lr=1e-4, n_iter=10000, th=0.2, grid_seeds = np.array([grid_s])):
    
    n_sample, inp_len = shuffled_code.shape
    n_poiss = int(n_sample/2)
    perc_seeds = grid_seeds-100
    n_grid_seed = grid_seeds.shape[0]
    #threshold crossing points
    shuffled_th_cross = np.zeros(n_grid_seed)
    nonshuffled_th_cross = np.zeros(n_grid_seed)
    labels, out_len = label(n_poiss)
    #Into tensor
    shuffled_code = torch.FloatTensor(shuffled_code)
    nonshuffled_code = torch.FloatTensor(nonshuffled_code)
    
    for i in range(n_grid_seed):
        perc_seed = perc_seeds[i]
        torch.manual_seed(perc_seed)
        net_shuffled = Net(inp_len, out_len)
        train_loss_shuffled, out_shuffled = train_net(net_shuffled, shuffled_code, labels, n_iter=n_iter, lr=lr)
        torch.manual_seed(perc_seed)
        net_nonshuffled = Net(inp_len, out_len)
        train_loss_nonshuffled, out_nonshuffled = train_net(net_nonshuffled, nonshuffled_code, labels, n_iter=n_iter, lr=lr)
        shuffled_th_cross[i] = np.argmax(np.array(train_loss_shuffled) < th)
        nonshuffled_th_cross[i] = np.argmax(np.array(train_loss_nonshuffled) < th)
    return shuffled_th_cross, nonshuffled_th_cross



'loading data'


path1 = '/home/baris/results/grid_mixed_input/new/'





#nonshuffled 75-74.5-74-73.5
# t75_grid = load['grid_rate_code'][0:5,:,0]
# grid_rate_ns[0,5*ct:5+5*ct,:] = t75_grid
#t75_grid, t74.5_grid such names are repeated in the loading process of othe files
#they are just to load the 

grid_rate_ns = np.empty((16,n_samples,8000))
gra_rate_ns = np.empty((16,n_samples,80000))
grid_phase_ns = np.empty((16,n_samples,8000))
gra_phase_ns = np.empty((16,n_samples,80000))
npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*non*75.0*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[0,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[1,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[2,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[3,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[0,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[1,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[2,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[3,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[0,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[1,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[2,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[3,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[0,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[1,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[2,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[3,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    if ct==(int(n_samples/5)):
        break
   
#shuffled 75-74.5-74-73.5 

grid_rate_s = np.empty((16,n_samples,8000))
gra_rate_s = np.empty((16,n_samples,80000))
grid_phase_s = np.empty((16,n_samples,8000))
gra_phase_s = np.empty((16,n_samples,80000))
npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*75.0*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[0,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[1,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[2,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[3,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[0,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[1,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[2,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[3,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[0,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[1,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[2,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[3,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[0,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[1,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[2,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[3,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break


#nonshuffled 73-72.5-72-71.5

npzfiles = []
ct=0
for file in sorted(glob.glob(os.path.join(path1,'*non*73.0*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[4,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[5,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[6,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[7,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[4,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[5,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[6,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[7,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[4,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[5,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[6,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[7,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[4,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[5,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[6,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[7,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    if ct==(int(n_samples/5)):
        break 
   
#shuffled  73.72.5-72-71.5


npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*73.0*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[4,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[5,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[6,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[7,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[4,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[5,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[6,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[7,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[4,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[5,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[6,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[7,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[4,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[5,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[6,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[7,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break


#nonshuffled 71-70.5-70-69

npzfiles = []
ct=0
for file in sorted(glob.glob(os.path.join(path1,'*non*69*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[8,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[9,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[10,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[11,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[8,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[9,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[10,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[11,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[8,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[9,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[10,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[11,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[8,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[9,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[10,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[11,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    if ct==(int(n_samples/5)):
        break 
   
#shuffled  71-70.5-70-69


npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*69*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[8,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[9,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[10,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[11,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[8,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[9,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[10,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[11,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[8,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[9,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[10,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[11,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[8,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[9,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[10,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[11,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break


#nonshuffled 68-67-66-65

npzfiles = []
ct=0
for file in sorted(glob.glob(os.path.join(path1,'*non*68*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[12,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[13,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[14,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[15,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[12,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[13,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[14,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[15,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[12,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[13,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[14,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[15,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[12,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[13,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[14,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[15,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    if ct==(int(n_samples/5)):
        break 
   
#shuffled  68-67-66-65


npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*68*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[12,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[13,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[14,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[15,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[12,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[13,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[14,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[15,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[12,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[13,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[14,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[15,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[12,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[13,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[14,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[15,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break

grid_s_ths_rate = [] 
grid_ns_ths_rate = [] 
gra_s_ths_rate = [] 
gra_ns_ths_rate = [] 


for i in range(16):
    trj1= 0
    trj2= i+1
    if trj2==16:
        break
    
    shuffled_grid_rate = np.vstack((grid_rate_s[trj1,:,:], grid_rate_s[trj2,:,:]))
    nonshuffled_grid_rate = np.vstack((grid_rate_ns[trj1,:,:], grid_rate_ns[trj2,:,:]))    
    shuffled_gra_rate = np.vstack((gra_rate_s[trj1,:,:], gra_rate_s[trj2,:,:]))
    nonshuffled_gra_rate = np.vstack((gra_rate_ns[trj1,:,:], gra_rate_ns[trj2,:,:]))
    
    grid_s_th_rate, grid_ns_th_rate = shuffle_perceptron(shuffled_grid_rate, nonshuffled_grid_rate, lr=lr)
    gra_s_th_rate, gra_ns_th_rate = shuffle_perceptron(shuffled_gra_rate, nonshuffled_gra_rate, lr=lr)
    
    grid_s_ths_rate.append(grid_s_th_rate)
    grid_ns_ths_rate.append(grid_ns_th_rate)
    gra_s_ths_rate.append(gra_s_th_rate)
    gra_ns_ths_rate.append(gra_ns_th_rate)
    
   
grid_s_ths_phase = [] 
grid_ns_ths_phase = [] 
gra_s_ths_phase = [] 
gra_ns_ths_phase = []     
   
for i in range(16):
    trj1= 0
    trj2= i+1
    if trj2==16:
        break
    
    shuffled_grid_phase = np.vstack((grid_phase_s[trj1,:,:], grid_phase_s[trj2,:,:]))
    nonshuffled_grid_phase = np.vstack((grid_phase_ns[trj1,:,:], grid_phase_ns[trj2,:,:]))    
    shuffled_gra_phase = np.vstack((gra_phase_s[trj1,:,:], gra_phase_s[trj2,:,:]))
    nonshuffled_gra_phase = np.vstack((gra_phase_ns[trj1,:,:], gra_phase_ns[trj2,:,:]))
    
    grid_s_th_phase, grid_ns_th_phase = shuffle_perceptron(shuffled_grid_phase, nonshuffled_grid_phase, lr=lr)
    gra_s_th_phase, gra_ns_th_phase = shuffle_perceptron(shuffled_gra_phase, nonshuffled_gra_phase, lr=lr)
    
    grid_s_ths_phase.append(grid_s_th_phase)
    grid_ns_ths_phase.append(grid_ns_th_phase)
    gra_s_ths_phase.append(gra_s_th_phase)
    gra_ns_ths_phase.append(gra_ns_th_phase)    



grid_phase_s.shape




fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharey='row', sharex='col')

fig1.suptitle('Perceptron learning speed - '+str(n_samples)+' samples \n grid seed= '+str(grid_s)+', lr = '+format(lr, '.1E'))
xticks = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10]
ax1.plot(xticks, 1/np.array(grid_s_ths_rate), 'k-', label='shuffled')
ax1.plot(xticks, 1/np.array(grid_ns_ths_rate), 'b--', label='nonshuffled')
ax1.legend()
ax1.set_title('grid rate code')
ax1.set_ylabel('Speed (1/N)')

ax2.plot(xticks, 1/np.array(grid_s_ths_phase), 'k-', label='shuffled')
ax2.plot(xticks, 1/np.array(grid_ns_ths_phase), 'b--', label='nonshuffled')
ax2.legend()
ax2.set_title('grid phase code')



ax3.plot(xticks, 1/np.array(gra_s_ths_rate), 'k-', label='shuffled')
ax3.plot(xticks, 1/np.array(gra_ns_ths_rate), 'b--', label='nonshuffled')
ax3.legend()
ax3.set_xlabel('distance (cm)')
ax3.set_title('granule rate code')
ax3.set_ylabel('Speed (1/N)')


ax4.plot(xticks, 1/np.array(gra_s_ths_phase), 'k-', label='shuffled')
ax4.plot(xticks, 1/np.array(gra_ns_ths_phase), 'b--', label='nonshuffled')
ax4.legend()
ax4.set_title('granule phase code')
ax4.set_xlabel('distance (cm)')
plt.tight_layout()

save_dir = '/home/baris/figures/'
fig1.savefig(save_dir+'perceptron_speed_'+str(n_samples)+'_samples_grid_seed_'+str(grid_s)+'.png', dpi=200)
