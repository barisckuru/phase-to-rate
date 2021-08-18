#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:15:14 2020

@author: baris
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os
from rate_n_phase_code_gra import phase_code, overall_spike_ct
import time
import copy
from neuron import h, gui  # gui necessary for some parameters to h namespace
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim



start = time.time()
# Where to search for nrnmech.dll file. Must be adjusted for your machine. For pyDentate
dll_files = [("C:\\Users\\DanielM\\Repos\\models_dentate\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
              "C:\\Users\\daniel\\Repos\\nrnmech.dll",
              ("C:\\Users\\Holger\\danielm\\models_dentate\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
              ("C:\\Users\\Daniel\\repos\\"
              "dentate_gyrus_Santhakumar2005_and_Yim_patterns\\"
              "dentategyrusnet2005\\nrnmech.dll"),
              ("/home/baris/Python/mechs_7-6_linux/"
                "x86_64/.libs/libnrnmech.so")]

for x in dll_files:
    if os.path.isfile(x):
        dll_dir = x
print("DLL loaded from: " + dll_dir)
h.nrn_load_dll(dll_dir)

#Parameters for the grid cell poisson input generation
savedir = os.getcwd()+'/asd'
n_grid = 200 
n_gra = 2000
max_rate = 20
dur_ms = 2000
bin_size = 100
n_bin = int(dur_ms/bin_size)
dur_s = dur_ms/1000
speed_cm = 20
field_size_cm = 100
traj_size_cm = int(dur_s*speed_cm)
inp_len = n_bin*n_gra

#Parameters for perceptron
lr = 5e-4 #was 5e-3 and good for 500ms, and for 2000ms 5e-4 was set
n_iter = 10000

pp_weight=1.5e-4

#Seeds
grid_seeds = np.arange(519,520,1)
poiss_seeds = np.arange(200,205,1)
perc_seeds = grid_seeds-100

n_poiss = poiss_seeds.shape[0]
n_network = perc_seeds.shape[0]

#Intialize zeros arrays&lists to fill with data
sample_size = 2*poiss_seeds.shape[0]
n_sampleset = perc_seeds.shape[0]
rate_code_sim = np.zeros((sample_size, 2*inp_len, n_sampleset))
rate_code_diff = np.zeros((sample_size, 2*inp_len, n_sampleset))
phase_code_sim = np.zeros((sample_size, 2*inp_len, n_sampleset))
phase_code_diff = np.zeros((sample_size, 2*inp_len, n_sampleset))
complex_code_sim = np.zeros((sample_size, 2*inp_len, n_sampleset))
complex_code_diff = np.zeros((sample_size, 2*inp_len, n_sampleset))
#thresholg crossing points
rate_th_cross_sim = []
rate_th_cross_diff = []
phase_th_cross_sim = []
phase_th_cross_diff = []
complex_th_cross_sim = []
complex_th_cross_diff = []
#RMSE loss in each epoch
rate_rmse_sim = []
rate_rmse_diff = []
phase_rmse_sim = []
phase_rmse_diff = []
complex_rmse_sim = []
complex_rmse_diff = []

#labels, output for training the network, 5 for each trajectory
for i in poiss_seeds:
    a = np.tile([1, 0], (len(poiss_seeds),1))
    b = np.tile([0, 1], (len(poiss_seeds),1))
    labels = np.vstack((a,b))
labels = torch.FloatTensor(labels) 
out_len = labels.shape[1]

#similar & distinct trajectories
sim_traj = np.array([75, 74.5])
diff_traj = np.array([75, 60])
n_traj = sim_traj.shape[0]

#Initialize the figures
fig1, ax1 = plt.subplots()
ax1.set_title('Rate Code (phases=0) Perceptron Loss '+str(dur_ms)+'ms \n' +str(n_poiss)+ ' Poisson seeds, '+str(n_network)+' net-grid seeds, learning rate = '+str(lr))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('RMSE Loss')
fig2, ax2 = plt.subplots()
ax2.set_title('Phase Code (rates=mean) Perceptron Loss '+str(dur_ms)+'ms \n' +str(n_poiss)+ ' Poisson seeds, '+str(n_network)+' net-grid seeds, learning rate = '+str(lr))
ax2.set_xlabel('Epochs')
ax2.set_ylabel('RMSE Loss')
fig3, ax3 = plt.subplots()
ax3.set_title('Complex Phase&Rate Code Perceptron Loss '+str(dur_ms)+'ms \n' +str(n_poiss)+ ' Poisson seeds, '+str(n_network)+' net-grid seeds, learning rate = '+str(lr))
ax3.set_xlabel('Epochs')
ax3.set_ylabel('RMSE Loss')



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

#main loop
# generate the grid data with different seeds
# put them into phase and rate code funstions and collect the data for perceptron
# generate the network with different seeds and plot the change in loss


for idx, perc_seed in enumerate(perc_seeds):
   
    #Input generation
    #Rate trajs with phase info; oscillations implemented in the rate profile
    grid_phases_sim,  gra_phases_sim, grid_spikes_sim, gra_spikes_sim, rate_trajs_sim, dt_s, theta_phase_sim, phase_code_dir_sim, overall_dir_sim = phase_code(sim_traj, dur_ms, grid_seeds[idx], poiss_seeds, pp_weight)
    grid_phases_diff, gra_phases_diff, grid_spikes_diff, gra_spikes_diff, rate_trajs_diff, dt_s, theta_phase_diff, phase_code_dir_diff, overall_dir_diff = phase_code(diff_traj, dur_ms, grid_seeds[idx], poiss_seeds, pp_weight)
    #grid and granule spike counts \ rate codes
    grid_sim_traj_cts, gra_sim_traj_cts = overall_spike_ct(grid_spikes_sim, gra_spikes_sim, dur_ms, poiss_seeds, n_traj=n_traj)
    grid_diff_traj_cts, gra_diff_traj_cts = overall_spike_ct(grid_spikes_diff, gra_spikes_diff, dur_ms, poiss_seeds, n_traj=n_traj)
    
    #change the rate code to mean where it is not 0
    # cts_for_phase_sim = copy.deepcopy(gra_sim_traj_cts)
    # cts_for_phase_sim[cts_for_phase_sim!=0]=np.mean(gra_sim_traj_cts) #was 1
    # cts_for_phase_diff = copy.deepcopy(gra_diff_traj_cts)
    # cts_for_phase_diff[cts_for_phase_diff!=0]=np.mean(gra_diff_traj_cts)
    
    #change rate code to mean of non zeros where it is nonzero
    cts_for_phase_sim = copy.deepcopy(gra_sim_traj_cts)
    cts_for_phase_sim[cts_for_phase_sim!=0]=np.mean(cts_for_phase_sim[cts_for_phase_sim!=0]) #was 1
    cts_for_phase_diff = copy.deepcopy(gra_diff_traj_cts)
    cts_for_phase_diff[cts_for_phase_diff!=0]=np.mean(cts_for_phase_diff[cts_for_phase_diff!=0])
    
    #rate code with constant 45 deg phase
    phase_of_rate_code = np.pi/4
    rate_y_sim = gra_sim_traj_cts*np.sin(phase_of_rate_code)
    rate_x_sim = gra_sim_traj_cts*np.cos(phase_of_rate_code)
    rate_sim =  np.concatenate((rate_y_sim, rate_x_sim), axis=1)
    rate_y_diff = gra_diff_traj_cts*np.sin(phase_of_rate_code)
    rate_x_diff = gra_diff_traj_cts*np.cos(phase_of_rate_code)
    rate_diff =  np.concatenate((rate_y_diff, rate_x_diff), axis=1)
    
    #phase code with phase and mean rate 
    phase_y_sim = cts_for_phase_sim*np.sin(gra_phases_sim)
    phase_x_sim = cts_for_phase_sim*np.cos(gra_phases_sim)
    phase_sim =  np.concatenate((phase_y_sim, phase_x_sim), axis=1)
    phase_y_diff = cts_for_phase_diff*np.sin(gra_phases_diff)
    phase_x_diff = cts_for_phase_diff*np.cos(gra_phases_diff)
    phase_diff =  np.concatenate((phase_y_diff, phase_x_diff), axis=1)
    #complex code with rate and phase
    complex_sim_y = gra_sim_traj_cts*np.sin(gra_phases_sim)
    complex_sim_x = gra_sim_traj_cts*np.cos(gra_phases_sim)
    complex_sim = np.concatenate((complex_sim_y, complex_sim_x), axis=1)
    complex_diff_y = gra_diff_traj_cts*np.sin(gra_phases_diff)
    complex_diff_x = gra_diff_traj_cts*np.cos(gra_phases_diff)
    complex_diff = np.concatenate((complex_diff_y, complex_diff_x), axis=1)
    
    #Normalization
    # gra_sim_traj_cts = gra_sim_traj_cts/np.amax(gra_sim_traj_cts)
    # gra_diff_traj_cts = gra_diff_traj_cts/np.amax(gra_diff_traj_cts)
    # gra_phases_sim = phase_sim/np.amax(phase_sim)
    # gra_phases_diff = phase_diff/np.amax(phase_diff)
    # complex_sim = complex_sim/np.amax(complex_sim)
    # complex_diff = complex_diff/np.amax(complex_diff)
    
    #input shape arrange; fill the rate array with 0 phases for the y coordinate, no phase information
    #if phases are 0 then due to sin(0)=0, cos(0)=1, half of them would be zeros and the rest would be the rates
    # phase_zeros = np.zeros((gra_sim_traj_cts.shape[0], gra_sim_traj_cts.shape[1]))
    # gra_sim_traj_cts = np.concatenate((gra_sim_traj_cts, phase_zeros), axis=1)
    # gra_diff_traj_cts = np.concatenate((gra_diff_traj_cts, phase_zeros), axis=1)

    #fill arrays to save the data
    rate_code_sim[:,:,idx] = rate_sim
    rate_code_diff[:,:,idx] = rate_diff
    phase_code_sim[:,:,idx] = phase_sim
    phase_code_diff[:,:,idx] = phase_diff
    complex_code_sim[:,:,idx] = complex_sim
    complex_code_diff[:,:,idx] = complex_diff
    
    print('data done!')

    #Into tensor
    rate_sim = torch.FloatTensor(rate_sim)
    rate_diff = torch.FloatTensor(rate_diff)
    phase_sim = torch.FloatTensor(phase_sim)
    phase_diff = torch.FloatTensor(phase_diff)
    complex_sim = torch.FloatTensor(complex_sim)
    complex_diff = torch.FloatTensor(complex_diff)

    #initate the network with diff types of inputs and plot the change in loss
    #rate code
    torch.manual_seed(perc_seed)
    net_rate_sim = Net(inp_len*2, out_len)
    rate_train_loss_sim, rate_out_sim = train_net(net_rate_sim, rate_sim, labels, n_iter=n_iter, lr=lr)
    rate_rmse_sim.append(rate_train_loss_sim)
    rate_th_cross_sim.append(np.argmax(np.array(rate_train_loss_sim) < 0.2))
    if perc_seed == perc_seeds[0]:
        ax1.plot(rate_train_loss_sim, 'b-', label=str(sim_traj[0])+'cm vs '+str(sim_traj[1])+'cm')
    else:
        ax1.plot(rate_train_loss_sim, 'b-')
        
    torch.manual_seed(perc_seed)
    net_rate_diff = Net(inp_len*2, out_len)
    rate_train_loss_diff, rate_out_diff = train_net(net_rate_diff, rate_diff, labels, n_iter=n_iter, lr=lr)
    rate_rmse_diff.append(rate_train_loss_diff)
    rate_th_cross_diff.append(np.argmax(np.array(rate_train_loss_diff) < 0.2))
    if perc_seed == perc_seeds[0]:
        ax1.plot(rate_train_loss_diff, 'r-', label=str(diff_traj[0])+'cm vs '+str(diff_traj[1])+'cm')
    else:
        ax1.plot(rate_train_loss_diff, 'r-')
        
    #phase code        
    torch.manual_seed(perc_seed)
    net_phase_sim = Net(inp_len*2, out_len)
    phase_train_loss_sim, out_sim = train_net(net_phase_sim, phase_sim, labels, n_iter=n_iter, lr=lr)
    phase_rmse_sim.append(phase_train_loss_sim)
    phase_th_cross_sim.append(np.argmax(np.array(phase_train_loss_sim) < 0.2))
    if perc_seed == perc_seeds[0]:
        ax2.plot(phase_train_loss_sim, 'b-', label=str(sim_traj[0])+'cm vs '+str(sim_traj[1])+'cm')
    else:
        ax2.plot(phase_train_loss_sim, 'b-')
        
    torch.manual_seed(perc_seed)
    net_phase_diff = Net(inp_len*2, out_len)
    phase_train_loss_diff, out_diff = train_net(net_phase_diff, phase_diff, labels, n_iter=n_iter, lr=lr)
    phase_rmse_diff.append(phase_train_loss_diff)
    phase_th_cross_diff.append(np.argmax(np.array(phase_train_loss_diff) < 0.2))
    if perc_seed == perc_seeds[0]:
        ax2.plot(phase_train_loss_diff, 'r-', label=str(diff_traj[0])+'cm vs '+str(diff_traj[1])+'cm')
    else:
        ax2.plot(phase_train_loss_diff, 'r-')
        
    #complex code
    torch.manual_seed(perc_seed)
    net_complex_sim = Net(inp_len*2, out_len)
    complex_train_loss_sim, complex_out_sim = train_net(net_complex_sim, complex_sim, labels, n_iter=n_iter, lr=lr)
    complex_rmse_sim.append(complex_train_loss_sim)
    complex_th_cross_sim.append(np.argmax(np.array(complex_train_loss_sim) < 0.2))
    if perc_seed == perc_seeds[0]:
        ax3.plot(complex_train_loss_sim, 'b-', label=str(sim_traj[0])+'cm vs '+str(sim_traj[1])+'cm')
    else:
        ax3.plot(complex_train_loss_sim, 'b-')
        
    torch.manual_seed(perc_seed)
    net_complex_diff = Net(inp_len*2, out_len)
    complex_train_loss_diff, complex_out_diff = train_net(net_complex_diff, complex_diff, labels, n_iter=n_iter, lr=lr)
    complex_rmse_diff.append(complex_train_loss_diff)
    complex_th_cross_diff.append(np.argmax(np.array(complex_train_loss_diff) < 0.2))
    if perc_seed == perc_seeds[0]:
        ax3.plot(complex_train_loss_diff, 'r-', label=str(diff_traj[0])+'cm vs '+str(diff_traj[1])+'cm')
    else:
        ax3.plot(complex_train_loss_diff, 'r-')
    
ax1.legend()
ax2.legend()
ax3.legend()

#add threshold line at 0.2
ax1.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')
ax2.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')
ax3.plot(np.arange(0,n_iter), 0.2*np.ones(n_iter), '--g')

save_dir = '/home/baris/repo/perceptron_results/'
fname = 'granule_rate_n_phase_perceptron_net-seed'+str(perc_seeds)+'_'+str(dur_ms)+'ms'

np.savez(save_dir+fname, 
         rate_code_sim = rate_code_sim,
         rate_code_diff = rate_code_diff,
         phase_code_sim = phase_code_sim,
         phase_code_diff = phase_code_diff,
         complex_code_sim = complex_code_sim,
         complex_code_diff = complex_code_diff,
         
         grid_spikes_sim = grid_spikes_sim,
         grid_spikes_diff = grid_spikes_diff,
         gra_spikes_sim = gra_spikes_sim,
         gra_spikes_diff = gra_spikes_diff,
         
         rate_rmse_sim = rate_rmse_sim,
         rate_rmse_diff = rate_rmse_diff,
         phase_rmse_sim = phase_rmse_sim,
         phase_rmse_diff = phase_rmse_diff,
         complex_rmse_sim = complex_rmse_sim,
         complex_rmse_diff = complex_rmse_diff,
         
         rate_th_cross_sim=rate_th_cross_sim, 
         rate_th_cross_diff=rate_th_cross_diff,
         phase_th_cross_sim=phase_th_cross_sim, 
         phase_th_cross_diff=phase_th_cross_diff,
         complex_th_cross_diff=complex_th_cross_diff, 
         complex_th_cross_sim=complex_th_cross_sim,
        
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
         grid_seeds = grid_seeds,
         poiss_seeds = poiss_seeds,
         perc_seeds = perc_seeds)

stop = time.time()
print('time, sec, min, hour  ')
print(stop-start)
time_min = (stop-start)/60
time_hour = time_min/60
print(time_min)
print(time_hour)


