#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 13:09:42 2021

@author: baris
"""
import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.stats import pearsonr,  spearmanr
import os
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq






#Grid and granule phase distributions form normal network
bin_size_ms = 100
n_phase_bins=360
dur_ms = 2000
#spike to phase converter
def spike2phase(spikes, bin_size_ms, n_phase_bins, dur_ms):
    spike_phases=np.empty(0)
    n_bins = int(dur_ms/bin_size_ms)
    rad = n_phase_bins/360*2*np.pi    
    for i in range(10):
        spike = spikes[i]
        for i in range(n_bins):
            for idx, val in np.ndenumerate(spike):
                curr_train = val[((bin_size_ms*(i) < val) & (val < bin_size_ms*(i+1)))]
                if curr_train.size != 0:
                    spike_phases = np.concatenate((spike_phases, curr_train%(bin_size_ms)/(bin_size_ms)*rad))
    return spike_phases
            
path = '/home/baris/results/grid_mixed_input/full/71-70-65-60/non-shuffled'

# path = '/home/baris/results/thesis_data/noinh/71-70-65-60/'
path = '/home/baris/results/thesis_data/diff_poiss/75-74.5-74-73.5/'

gra_p_dist=np.empty(0)
grid_p_dist=np.empty(0)
ori_grid_p_dist=np.empty(0)
counts_grid=np.empty(0)
counts_granule=np.empty(0)
npzfiles = []
for file in sorted(glob.glob(os.path.join(path,'*.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    gra_spikes = np.concatenate((load['gra_spikes_sim'], load['gra_spikes_diff']))
    grid_spikes = np.concatenate((load['grid_spikes_sim'], load['grid_spikes_diff']))
    # ori_grid_spikes = np.concatenate((load['ori_grid_spikes_sim'], load['ori_grid_spikes_diff']))
    
    gra_phases = spike2phase(gra_spikes, bin_size_ms, n_phase_bins, dur_ms)
    # print(gra_phases.shape)
    gra_p_dist = np.concatenate((gra_p_dist, gra_phases))      
    grid_p_dist = np.concatenate((grid_p_dist, spike2phase(grid_spikes, bin_size_ms, n_phase_bins, dur_ms)))
    # ori_grid_p_dist = np.concatenate((grid_p_dist, spike2phase(ori_grid_spikes, bin_size_ms, n_phase_bins, dur_ms)))

gra_p_dist = gra_p_dist/np.pi
grid_p_dist = grid_p_dist/np.pi
plt.close('all')
phase_dist= np.append(grid_p_dist, gra_p_dist)
cell = ['grid']*(grid_p_dist.shape[0]) + ['granule']*(gra_p_dist.shape[0])
phase_df = pd.DataFrame({'phase distribution ($\pi$)': phase_dist,
                   'cell': pd.Categorical(cell)})

fig_inh = sns.histplot(data=phase_df, x='phase distribution ($\pi$)', kde=True, hue='cell', binwidth=(2*np.pi/180))
plt.title('Full Network (Non-Shuffled)')
sns.set(context='paper',style='darkgrid',palette='deep',
        font='Arial',font_scale=1.5,color_codes=True,rc={'figure.figsize':(8,4)})
fig_inh.set(ylim=(0,100000))

plt.tight_layout()

save_dir = '/home/baris/figures/'
plt.savefig(save_dir+'non-shuffled_full_phase_dist_ind_spikes.eps', dpi=200)
plt.savefig(save_dir+'non-shuffled_full_phase_dist_ind_spikes.png', dpi=200)



# path = '/home/baris/results/grid_mixed_input/noff/71-70-65-60/'

path = '/home/baris/results/thesis_data/noff/71-70-65-60/'
counts_grid=np.empty(4000)
counts_granule=np.empty(40000)
npzfiles = []

for file in sorted(glob.glob(os.path.join(path,'*.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    gra_spikes = np.concatenate((load['gra_spikes_sim'], load['gra_spikes_diff']))
    grid_spikes = np.concatenate((load['grid_spikes_sim'], load['grid_spikes_diff']))
    # ori_grid_spikes = np.concatenate((load['ori_grid_spikes_sim'], load['ori_grid_spikes_diff']))
    ct_grid, ct_gra  = overall_spike_ct(grid_spikes, gra_spikes, 10)
    plt.figure()
    plt.imshow(ct_gra, aspect='auto')
    counts_grid = np.vstack((counts_grid, ct_grid))
    counts_granule = np.vstack((counts_granule, ct_gra))
    
np.sum(counts_grid)
np.sum(counts_granule)
def binned_ct(arr, bin_size_ms, dt_ms=25, time_ms=5000):
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    n_traj = arr.shape[1]
    counts = np.zeros((n_bins, n_cells, n_traj))
    for i in range(n_bins):
        for index, value in np.ndenumerate(arr):
            curr_ct = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            counts[i][index] = curr_ct
            #search and count the number of spikes in the each bin range
    return counts



def spike_ct (grid_spikes, gra_spikes, n):
    for i in range(n):
        grid_spikes
    

def overall_spike_ct(grid_spikes, gra_spikes, n_samples, dur_ms=2000, n_traj=2, bin_size=100):
    n_bin = int(dur_ms/bin_size)
    dur_s = dur_ms/1000
    n_grid = 200
    n_granule = 2000
    counts_grid_1 = np.zeros((n_samples, n_bin*n_grid))
    counts_grid_2 = np.zeros((n_samples, n_bin*n_grid))
    counts_gra_1 = np.zeros((n_samples, n_bin*n_granule))
    counts_gra_2 = np.zeros((n_samples, n_bin*n_granule))

    for idx in range(n_samples):
        counts_grid_1[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_grid_2[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
        counts_gra_1[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_gra_2[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
    counts_grid = np.vstack((counts_grid_1, counts_grid_2))
    counts_granule = np.vstack((counts_gra_1, counts_gra_2))
    return counts_grid, counts_granule   





path = '/home/baris/results/grid_mixed_input/diff_poiss/full/75-74.5-74-73.5/'


#diff poiss seeds
import glob

fname= os.path.join(path,'*75*420*')

glob.glob(fname)



fname1 = 'shuffled_full_diff_poiss_traj_75-74.5-74-73.5_net-seed420_2000ms.npz'
fname2 = 'non-shuffled_full_diff_poiss_traj_75-74.5-74-73.5_net-seed420_2000ms.npz'

shuffled_save_dir = path+fname1
nonshuffled_save_dir = path+fname2

load_shuffled = np.load(shuffled_save_dir, allow_pickle=True)
load_nonshuffled = np.load(nonshuffled_save_dir, allow_pickle=True)


#grid codes

shuffled_grid_spi = load_shuffled['grid_spikes_sim'][0]

nonshuffled_grid_spi = load_nonshuffled['grid_spikes_sim'][0]



shuffled_gra_spi = load_shuffled['ori_grid_spikes_sim'][0]

nonshuffled_gra_spi = load_nonshuffled['ori_grid_spikes_sim'][0]



shuffled_gra_rate = load_shuffled['grid_rate_code']
nonshuffled_gra_rate = load_nonshuffled['grid_rate_code']


shuffled_grid_phase = load_shuffled['grid_phase_code']
nonshuffled_grid_phase = load_nonshuffled['grid_phase_code']

traj = 0
grid_shuffled_th_cross_7170_rate, grid_nonshuffled_th_cross_7170_rate = shuffle_perceptron(shuffled_grid_rate, nonshuffled_grid_rate, traj)

traj = 0
grid_shuffled_th_cross_7170_phase, grid_nonshuffled_th_cross_7170_phase = shuffle_perceptron(shuffled_grid_phase, nonshuffled_grid_phase, traj)


def sample_select(code, idx1, idx2):
    all_samples = np.concatenate((code[:,:,0], code[:,:,1]), axis=0)
    # samples = np.concatenate((code[idx1[0]:idx1[1],:,0], code[idx2[0]:idx2[1],:,1]), axis=0)
    samples = np.concatenate((all_samples[idx1[0]:idx1[1],:], all_samples[idx2[0]:idx2[1],:]), axis=0)
    return samples



#Pearson R
shuffled_gra_phase_1 = load_shuffled['gra_phase_code'][0,:,0]
shuffled_gra_phase_2 = load_shuffled['gra_phase_code'][0,:,0]

#gra
pearsonr(load_shuffled['gra_phase_code'][5,:,0], load_shuffled['gra_phase_code'][5,:,1])
pearsonr(load_nonshuffled['gra_phase_code'][5,:,0], load_nonshuffled['gra_phase_code'][5,:,1])

pearsonr(load_shuffled['gra_rate_code'][5,:,0], load_shuffled['gra_rate_code'][5,:,1])
pearsonr(load_nonshuffled['gra_rate_code'][5,:,0], load_nonshuffled['gra_rate_code'][5,:,1])

#grid
pearsonr(load_shuffled['grid_phase_code'][5,:,0], load_shuffled['grid_phase_code'][5,:,1])
pearsonr(load_nonshuffled['grid_phase_code'][5,:,0], load_nonshuffled['grid_phase_code'][5,:,1])

pearsonr(load_shuffled['grid_rate_code'][5,:,0], load_shuffled['grid_rate_code'][5,:,1])
pearsonr(load_nonshuffled['grid_rate_code'][5,:,0], load_nonshuffled['grid_rate_code'][5,:,1])




#perceptron
#gra

shuffled_gra_rate = sample_select(load_shuffled['gra_rate_code'], [5,10], [15,20])
nonshuffled_gra_rate = sample_select(load_nonshuffled['gra_rate_code'], [5,10], [15,20])

shuffled_gra_phase = sample_select(load_shuffled['gra_phase_code'], [5,10], [15,20])
nonshuffled_gra_phase = sample_select(load_nonshuffled['gra_phase_code'], [5,10], [15,20])


shuffled_th_cross_rate, nonshuffled_th_cross_rate = shuffle_perceptron(shuffled_gra_rate, nonshuffled_gra_rate)
shuffled_th_cross_phase, nonshuffled_th_cross_phase = shuffle_perceptron(shuffled_gra_phase, nonshuffled_gra_phase)


#grid

shuffled_grid_rate = sample_select(load_shuffled['grid_rate_code'], [5,10], [15,20])
nonshuffled_grid_rate = sample_select(load_nonshuffled['grid_rate_code'], [5,10], [15,20])

shuffled_grid_phase = sample_select(load_shuffled['grid_phase_code'], [5,10], [15,20])
nonshuffled_grid_phase = sample_select(load_nonshuffled['grid_phase_code'], [5,10], [15,20])


shuffled_th_cross_rate, nonshuffled_th_cross_rate = shuffle_perceptron(shuffled_grid_rate, nonshuffled_grid_rate)
shuffled_th_cross_phase, nonshuffled_th_cross_phase = shuffle_perceptron(shuffled_grid_phase, nonshuffled_grid_phase)


#


traj = 1
shuffled_th_cross_6560, nonshuffled_th_cross_6560 = shuffle_perceptron(shuffled_gra_rate, nonshuffled_gra_rate, traj)


pearsonr(shuffled_gra_rate[0,:,0], nonshuffled_gra_rate[5,:,0])
pearsonr(shuffled_gra_rate[0,:,1], shuffled_gra_rate[1,:,1])

shuffled_gra_spi[3,0]


ct = 0
for idx, val in np.ndenumerate(shuffled_gra_spi):
    ct += 1
    if val.shape[0] != 0 and shuffled_gra_spi[idx].shape[0] != 0:
        
        
        
    
ct = 0
for i in range(2000):
    # print(i)
    if shuffled_gra_spi[i,0].shape[0] != 0 and nonshuffled_gra_spi[i,0].shape[0] != 0:
        
        ct += 1
            
        
ct = 0
for i in range(2000):
    if shuffled_gra_spi[i,0].shape[0] != 0:
        
        ct += 1
    
    
ct = 0
for i in range(2000):
    if shuffled_gra_spi[i,1].shape[0] != 0:
        
        ct += 1    
    
ct = 0
for i in range(2000):
    if shuffled_gra_spi[i,1].shape[0] != 0 and nonshuffled_gra_spi[i,1].shape[0] != 0:
        # print(shuffled_gra_spi[i,1])
        # print(nonshuffled_gra_spi[i,1])
        ct += 1
     

shuffled_gra_spi[3,0] != np.empty(0, dtype=np.float64)

shuffled_gra_spi[1,0].shape[0] > 0




Cøp


load = np.load('/home/baris/results/thesis_data/noinh/71-70-65-60/rate_n_phase_codes_noinh_traj_71-70-65-60_net-seed411_2000ms.npz',
               allow_pickle=True)


gra_spikes = load['gra_spikes_sim']
grid_spikes = load['grid_spikes_sim']



import numpy.random as rd

load = np.load('/home/baris/results/thesis_data/noinh/71-70-65-60/rate_n_phase_codes_noinh_traj_71-70-65-60_net-seed411_2000ms.npz',
               allow_pickle=True)


load = np.load('/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/rate_n_phase_traj_diff_poiss_71-70-65-60_net-seeds_410-429_2000ms.npz',
               allow_pickle=True)
path='/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/'
grid_rate, grid_phase, grid_complex, gra_rate, gra_phase, gra_complex = all_codes(path)

gra_spikes = load['gra_spikes_sim']
grid_spikes = load['grid_spikes_sim']

grid_spikes[0][0,0].shape



np.sort(rd.uniform(0,2000,29))


%

def inhom_poiss(arr, n_traj, dur_s, poiss_seed=0, dt_s=0.025):
    #length of the trajectory that mouse went
    np.random.seed(poiss_seed)
    n_cells = arr.shape[0]
    spi_arr = np.zeros((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            np.random.seed(poiss_seed+grid_idc)
            rate_profile = arr[grid_idc,:,i]
            asig = AnalogSignal(rate_profile,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=dur_s*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)
            curr_train = stg.inhomogeneous_poisson_process(asig)
            spi_arr[grid_idc, i] = np.array(curr_train.times*1000) #time conv to ms
    return spi_arr


n, bins, pathces = plt.hist(grid_p_dist, bins = np.arange(0, 2.001, 2/100))

norm_n = n/np.max(n)

path = '/home/baris/results/grid_mixed_input'
np.savez(os.path.join(path,'norm_grid_phase_dist'), 
             grid_norm_dist = norm_n)

(
[]
def rand_grid_spike(n_spikes, total_spikes_bin= 20):
    import random
    np.load('/home/baris/results/grid_mixed_input/norm_grid_phase_dist.npz')
    dt_s = 0.001
    dur_s = 0.1
    phase_prof = (total_spikes_bin/np.sum(norm_n))*norm_n/(dt_s)
    asig = AnalogSignal(phase_prof,
                            units=1*pq.Hz,
                            t_start=0*pq.s,
                            t_stop=dur_s*pq.s,
                            sampling_period=dt_s*pq.s,
                            sampling_interval=dt_s*pq.s)
    curr_train = stg.inhomogeneous_poisson_process(asig, as_array=True, refractory_period=(dt_s*pq.s))
    spikes_ms = random.choices(curr_train*1000, k=n_spikes)
    return spikes_ms

spikes = np.zeros(0)


for i in range(1000):
    spikes = np.append(spikes, rand_grid_spike(10))
    
plt.hist(spikes, bins = np.arange(0, 101, 1))

plt.hist(n, bins = np.arange(0, 2.001, 2/100))

plt.hist(curr_train*1000, bins = np.arange(0, 101, 1))


plt.hist(grid_phases_sim.flatten(), bins = np.arange(0, 2*np.pi, 2*np.pi/100))
plt.hist(gra_phases_sim.flatten(), bins = np.arange(0, 2*np.pi, 2*np.pi/100))

plt.close('all')
plt.plot(grid_rate_code.flatten())
plt.figure()
plt.plot(gra_rate_code.flatten())
np.max(grid_phases_sim)
plt.plot(phase_prof)

np.sum(phase_prof)
max(curr_train*1000)


curr_train.shape

0.001 


load = np.load('/home/baris/results/thesis_data/noinh/71-70-65-60/rate_n_phase_codes_noinh_traj_71-70-65-60_net-seed410_2000ms.npz')


cts = load['grid_sim_traj_cts']
np.max(cts)




for index, value in np.ndenumerate(arr):

dt=0.1    
cells = grid_spikes_sim[0][:,0]

spikes_diff = np.array([np.diff(x) for x in cells])

for i in spikes_diff:
    print((i<dt).any())



















