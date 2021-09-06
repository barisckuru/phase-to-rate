#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 17:13:07 2020

@author: baris
"""

gra_spikes = gra_spikes_diff[0]
f=10
n_phase_bins=360
dur_ms=500
dur_s = dur_ms/1000
T = 1/f
time_bin_size = T
times = np.arange(0, dur_s+time_bin_size, time_bin_size)
n_time_bins = int(dur_s/T)

def mean_phase(spikes, T, n_phase_bins, n_time_bins, times):
    rad = n_phase_bins/360*2*np.pi
    spikes=np.array(spikes)
    # phases = rad*np.ones((spikes.shape[0], n_time_bins, spikes.shape[1]))
    phases = np.zeros((spikes.shape[0], n_time_bins, spikes.shape[1]))
    spikes_s = spikes/1000
    for idx, val in np.ndenumerate(spikes_s):
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = val[np.logical_and(val > time, val < times[j+1])]
            if curr_train.size != 0:
                print(curr_train)
                phases[idx[0],j,idx[1]] = np.mean(curr_train%(T)/(T)*rad)
    # phases = phases.reshape(-1, phases.shape[-1]).T   
    print('PHASES shape:' + str(phases.shape))
    return phases

curr_train = (np.array([0.0282,0.1]))%(0.05)/(0.05)*rad

def mean_phase(spikes, bin_size_ms, n_phase_bins, dur_ms):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells =spikes.shape[0] 
    n_traj = spikes.shape[1]
    rad = n_phase_bins/360*2*np.pi
    spikes=np.array(spikes)
    # phases = rad*np.ones((spikes.shape[0], n_time_bins, spikes.shape[1]))
    phases = np.zeros((n_bins, n_cells, n_traj))

    for i in range(n_bins):
        for idx, val in np.ndenumerate(spikes):
            curr_train = val[((bin_size_ms*(i) < val) & (val < bin_size_ms*(i+1)))]
            if curr_train.size != 0:
                phases[i][idx] = np.mean(curr_train%(bin_size_ms)/(bin_size_ms)*rad)

    return phases

def binned_ct(arr, bin_size_ms, dt_ms=25, time_ms=5000):
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    n_traj = arr.shape[1]
    counts = np.zeros((n_bins, n_cells, n_traj))
    print('binned_ct curr_ct and its shape: \n')
    for i in range(n_bins):
        for index, value in np.ndenumerate(arr):
            curr_ct = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            counts[i][index] = curr_ct
            print(curr_ct)
            #search and count the number of spikes in the each bin range
    return counts



mean_phases = mean_phase(gra_spikes, 100, n_phase_bins, dur_ms)
binned_cts = binned_ct(gra_spikes, 100, dt_ms=25, time_ms=dur_ms)

n_zeros = np.sum(binned_cts==0)

phases = phases.reshape(-1, phases.shape[-1]).T  

mean_phases[0].shape

gra_spikes.shape



mean_phases=mean_filler(mean_phases)

mean_phases.shape
binned_cts.shape

def mean_filler(phases):
    mean_phases = np.mean(phases[phases!=0])
    phases[phases==0] = mean_phases
    return phases

3624

factor_sim = 1 / stats.mode(gra_sim_traj_cts[gra_sim_traj_cts!=0], axis=None)[0][0]
gra_sim_traj_cts = gra_sim_traj_cts[:,:40000]*factor_sim
factor_diff = 1 / stats.mode(gra_diff_traj_cts[gra_diff_traj_cts!=0], axis=None)[0][0]
gra_diff_traj_cts = gra_diff_traj_cts[:,:40000]*factor_diff


stats.mode(gra_sim_traj_cts[gra_sim_traj_cts!=0], axis=None)[0][0]



grid_spikes = grid_spikes_sim[0]

plt.figure()
plt.eventplot(grid_spikes[:,0])

gra_spikes = gra_spikes_sim[0]

plt.figure()
plt.eventplot(gra_spikes[:,0])



for param in net.parameters():
  print(param.data)



for param in net_rate_phase_sim.parameters():
    print(param.data.shape)
    weights = np.array(param.data)
    break


loaded = np.load('pyDentate_out_weight_0.00015_seed2_200.npz', allow_pickle = True)


gran_out = loaded['granule_output']

net_rate_phase_sim.parameters().data



count = 0

for i in gra_spikes_sim[0][:,0]:
    # np.warnings.filterwarnings('ignore')
    if i.size != 0 and np.sum(np.logical_and(i>50,i<250))>0:
        # print(i)
        count += 1
count
        


count=0
np.sum(gra_sim_traj_cts[gra_sim_traj_cts==0]=1)

phases = np.array([[0,1,2,3],[0,0,2,0]])
mean = np.mean(phases[phases!=0])


a = [1,2,3,4]

b = copy.deepcopy(a) 

b[1]= 1


np.sum(gra_sim_traj_cts==0)
gra_phases_sim[4,0]


19401+19399+19479+19374+19432

gridspikes_diff = grid_spikes_diff[0]

a = np.array([15,150,100])
np.sum(a>50)>0
a.size != 0

np.sum(grid_spikes_diff[0][0,0]>
       grid_spikes_diff[0]
       v
       v
       grid_spikes_diff[0].shape

grid_spikes_diff[0][0,0]

gra_spikes_diff[0].shape

gra_spikes_diff[0][0,0]

overall_dir_diff.shape

plt.figure()
plt.imshow(theta_phase_diff[:,:,0])
plt.colorbar()

plt.figure()
plt.imshow(phase_code_dir_diff[:,:,1])
plt.colorbar()

plt.figure()
plt.imshow(overall_dir_sim[:,:,0])
plt.colorbar()



plt.figure()
plt.plot(gra_sim_traj_cts[3,:])
plt.title('Granule Spike Counts in 500ms(5*100ms) ')
plt.ylabel('Normalized Spike Counts')
plt.xlabel('Granule Cells (5*2000)')

import glob
import numpy as np
npzfiles = []
rate_code_sim =[]
rate_code_diff =[]
phase_code_sim =[]
phase_code_diff = []
complex_code_sim = []
complex_code_diff= []
for file in glob.glob("*.npz"):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    rate_code_sim.append(load['rate_code_sim'])
    rate_code_diff.append(load['rate_code_diff'])
    phase_code_sim.append(load['phase_code_sim'])
    phase_code_diff.append(load['phase_code_diff'])
    complex_code_sim.append(load['complex_code_sim'])
    complex_code_diff.append(load['complex_code_diff'])
    
np.savez('granule_rate_n_phase_perceptron_2000ms_net-seeds_410-419', 
         rate_code_sim = rate_code_sim,
         rate_code_diff = rate_code_diff,
         phase_code_sim = phase_code_sim,
         phase_code_diff = phase_code_diff,
         complex_code_sim = complex_code_sim,
         complex_code_diff = complex_code_diff)

a = np.array([1,2,3])
b = np.array([2,3,4])
c = np.array([3,4,5])
d= np.array([4,5,6])

e = []
e.append(a)
e.append(b)
e.append(c)
e.append(d)

e = np.array(e)
e


loaded = np.load('granule_rate_n_phase_perceptron_net-seeds_419_417_416.npz', allow_pickle=True)

l