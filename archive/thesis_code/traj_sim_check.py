#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:00:26 2021

@author: baris
"""
import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from scipy import interpolate
from skimage.measure import profile_line
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
import os
import copy
from scipy import ndimage

from neuron import h, gui  # gui necessary for some parameters to h namespace
import net_tunedrev
import scipy.stats as stats
from scipy.stats import pearsonr





#Parameters for the grid cell poisson input generation
savedir = os.getcwd()
n_grid = 200 
n_granule = 2000
max_rate = 1
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
bin_size = 100
speed_cm = 20
field_size_cm = 100
def_dt_s = 0.025
new_dt_s = 0.002
dt_s = new_dt_s
dur_ms = 2000
n_bin = int(dur_ms/bin_size)
dur_s = dur_ms/1000
traj_size_cm = int(dur_s*speed_cm)
inp_len = n_bin*n_granule


lr_grid = 5e-4
lr_gra = 5e-3 #was 5e-3 and good for 500ms, and for 2000ms 5e-4 was set
n_iter = 10000
th = 0.2
pp_weight=9e-4

#Seeds
grid_seeds = 510
poiss_seeds = np.arange(200,202,1)
perc_seeds = grid_seeds-100

n_poiss = poiss_seeds.shape[0]
n_network = 1 #perc_seeds.shape[0]

#similar & distinct trajectories
sim_traj = np.array([75, 74.5])
diff_traj = np.array([74, 65])
n_traj = sim_traj.shape[0]

#Intialize zeros arrays&lists to fill with data
sample_size = 2*poiss_seeds.shape[0]
n_sampleset = 1 #perc_seeds.shape[0]


def rate2dist(grids, spacings, max_rate):
    grid_dist = np.zeros((grids.shape[0], grids.shape[1], grids.shape[2]))
    for i in range(grids.shape[2]):
        grid = grids[:,:,i]
        spacing = spacings[i]
        trans_dist_2d = (np.arccos(((grid*3/(2*max_rate))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)
        grid_dist[:,:,i] = (trans_dist_2d/(spacing/2))/2
        # grid_dist[:,:,i][grid_dist[:,:,i]>0.5] = 0.5
    return grid_dist

#interpolation
def interp(arr, dur_s, dt_s, new_dt_s):
    arr_len = arr.shape[1]
    t_arr = np.linspace(0, dur_s, arr_len)
    if new_dt_s != dt_s: #if dt given is different than default_dt_s(0.025), then interpolate
        new_len = int(dur_s/new_dt_s)
        new_t_arr = np.linspace(0, dur_s, new_len)
        f = interpolate.interp1d(t_arr, arr, axis=1)
        interp_arr = f(new_t_arr)
    return interp_arr, new_t_arr


def inhom_poiss(arr, n_traj, dur_s, seed_2=0, dt_s=0.025):
    #length of the trajectory that mouse went
    np.random.seed(seed_2)
    n_cells = arr.shape[0]
    spi_arr = np.zeros((n_cells, n_traj), dtype = np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            np.random.seed(seed_2+grid_idc)
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



def mean_phase(spikes, bin_size_ms, n_phase_bins, dur_ms):
    n_bins = int(dur_ms/bin_size_ms)
    n_cells =spikes.shape[0] 
    n_traj = spikes.shape[1]
    rad = n_phase_bins/360*2*np.pi

    phases = np.zeros((n_bins, n_cells, n_traj))

    for i in range(n_bins):
        for idx, val in np.ndenumerate(spikes):
            curr_train = val[np.logical_and((bin_size_ms*(i) < val), (val < bin_size_ms*(i+1)))]
            if curr_train.size != 0:
                phases[i][idx] = np.mean((curr_train%(bin_size_ms))/(bin_size_ms)*rad)

    return phases

def mean_filler(phases):
    mean_phases = np.mean(phases[phases!=0])
    phases[phases==0] = mean_phases
    return phases

#Count the number of spikes in bins 

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


a = np.array([287,254,230,201,204,215])
a[a!=0]=np.mean(a[a!=0])

np.mean((a%(100))/(100))

def perceptron(sim_traj_cts, diff_traj_cts, phases_sim, phases_diff, perc_seed, lr):
    
    #threshold crossing points
    th_cross = np.zeros(6)
    #change rate code to mean of non zeros where it is nonzero
    cts_for_phase_sim = copy.deepcopy(sim_traj_cts)
    cts_for_phase_sim[cts_for_phase_sim!=0]=np.mean(cts_for_phase_sim[cts_for_phase_sim!=0]) #was 1
    cts_for_phase_diff = copy.deepcopy(diff_traj_cts)
    cts_for_phase_diff[cts_for_phase_diff!=0]=np.mean(cts_for_phase_diff[cts_for_phase_diff!=0])
    
    #rate code with constant 45 deg phase
    phase_of_rate_code = np.pi/4
    rate_y_sim = sim_traj_cts*np.sin(phase_of_rate_code)
    rate_x_sim = sim_traj_cts*np.cos(phase_of_rate_code)
    rate_sim =  np.concatenate((rate_y_sim, rate_x_sim), axis=1)
    rate_y_diff = diff_traj_cts*np.sin(phase_of_rate_code)
    rate_x_diff = diff_traj_cts*np.cos(phase_of_rate_code)
    rate_diff =  np.concatenate((rate_y_diff, rate_x_diff), axis=1)
    
    #phase code with phase and mean rate 
    phase_y_sim = cts_for_phase_sim*np.sin(phases_sim)
    phase_x_sim = cts_for_phase_sim*np.cos(phases_sim)
    phase_sim =  np.concatenate((phase_y_sim, phase_x_sim), axis=1)
    phase_y_diff = cts_for_phase_diff*np.sin(phases_diff)
    phase_x_diff = cts_for_phase_diff*np.cos(phases_diff)
    phase_diff =  np.concatenate((phase_y_diff, phase_x_diff), axis=1)
    #complex code with rate and phase
    complex_sim_y = sim_traj_cts*np.sin(phases_sim)
    complex_sim_x = sim_traj_cts*np.cos(phases_sim)
    complex_sim = np.concatenate((complex_sim_y, complex_sim_x), axis=1)
    complex_diff_y = diff_traj_cts*np.sin(phases_diff)
    complex_diff_x = diff_traj_cts*np.cos(phases_diff)
    complex_diff = np.concatenate((complex_diff_y, complex_diff_x), axis=1)
    
    rate_code = np.stack((rate_sim, rate_diff), axis=2)
    phase_code = np.stack((phase_sim, phase_diff), axis=2)
    complex_code = np.stack((complex_sim, complex_diff), axis=2)

    return rate_code, phase_code, complex_code

def _add_direction(dist_trajs, n_traj, n_grid):
    #infer the direction out of rate of change in the location
    direction = np.diff(dist_trajs, axis=1)
    #last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:,-1:,:]), axis=1)
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    direction = -direction
    dist_trajs = dist_trajs*direction
    dist_trajs = ndimage.gaussian_filter1d(dist_trajs, sigma=1, axis=1)
    return dist_trajs
    

def phase_code(trajs, dur_ms, seed_1, seed_2s, pp_weight, speed, f=10, shift_deg=200):
    dur_s = dur_ms/1000
    T = 1/f
    bin_size_ms = T*1000
    n_phase_bins = 360
    n_time_bins = int(dur_ms/bin_size_ms)
    bins_size_deg = 1
    n_traj = trajs.shape[0]
    dt_s = new_dt_s
    
    grids, spacings, grid_dist = grid_population(n_grid, max_rate, seed=seed_1, arr_size=arr_size)
    grid_dist = rate2dist(grids, spacings, max_rate)
    dist_trajs, dt_s = draw_traj(grid_dist, n_grid, trajs, dur_ms=dur_ms)
    
    rate_trajs, rate_dt_s = draw_traj(grids, n_grid, trajs, dur_ms=dur_ms)
    rate_trajs, rate_t_arr = interp(rate_trajs, dur_s, def_dt_s, new_dt_s)
    
    # theta = (np.sin(f*2*np.pi*dist_t_arr)+1)/2

    # direction = _direction(dist_trajs, n_traj, n_grid)
    
    direction = np.diff(dist_trajs, axis=1)
    #last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:,-1:,:]), axis=1)
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    direction = -direction
    # direction[np.logical_and(direction < 0.1, direction > -0.1)] = 1
    # direction = ndimage.gaussian_filter(direction, sigma=1)
    
    traj_dist_dir = dist_trajs*direction
    traj_dist_dir = ndimage.gaussian_filter1d(traj_dist_dir, sigma=2, axis=1)
    traj_dist_dir, dist_t_arr = interp(traj_dist_dir, dur_s, def_dt_s, new_dt_s)
    one_theta_phase = (2*np.pi*(dist_t_arr%T)/T)%(2*np.pi)
    theta_phase = np.repeat(one_theta_phase[np.newaxis,:], 200, axis=0)
    theta_phase =  np.repeat(theta_phase[:,:,np.newaxis], n_traj, axis=2)
    
    factor = shift_deg/360 #change the phase shift from 360 degrees to 240 degrees
    # sigma = ((spacings/2)/speed_cm)/3
    # c = np.pi/(4*sigma*2*pi*f)
    c = 180/360
    firing_phase_dir = 2*np.pi*(traj_dist_dir+0.5)*c
    phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-theta_phase))
    # phase_code_dir = np.sin(firing_phase_dir-theta_phase)
    scaling_factor = 6 #to equalize the max freq to 20Hz
    constant_mv = 0.16
    overall_dir = phase_code_dir*rate_trajs*speed_cm*constant_mv*scaling_factor
    
    grid_phases_1 = np.zeros((len(seed_2s), n_time_bins*n_grid))
    grid_phases_2 = np.zeros((len(seed_2s), n_time_bins*n_grid))
    gra_phases_1 = np.zeros((len(seed_2s), n_time_bins*n_granule))
    gra_phases_2 = np.zeros((len(seed_2s), n_time_bins*n_granule))
    grid_spikes = np.zeros(len(seed_2s), dtype = np.ndarray)
    gra_spikes = np.zeros(len(seed_2s), dtype = np.ndarray)
    for idx, seed_2 in enumerate(seed_2s):
        curr_grid_spikes = inhom_poiss(overall_dir, n_traj, dur_s, dt_s=dt_s, seed_2=seed_2)

        # curr_gra_spikes = pyDentate(curr_grid_spikes, seed_1, seed_2, n_traj, dur_ms, pp_weight)[0]

        #grid phases
        curr_grid_phases = mean_phase(curr_grid_spikes, bin_size_ms, n_phase_bins, dur_ms)
        grid_phases_1[idx, :] = curr_grid_phases[:,:,0].flatten()
        grid_phases_2[idx, :] = curr_grid_phases[:,:,1].flatten()

        #granule phases
        # curr_gra_phases = mean_phase(curr_gra_spikes, bin_size_ms, n_phase_bins, dur_ms)
        # curr_gra_phases = mean_filler(curr_gra_phases)
        #this separation is necessary bcs 
        #when there are multiple seeds, same types of inputs are grouped together, 
        #then all are stacked in one array at the end
        # gra_phases_1[idx, :] = curr_gra_phases[:,:,0].flatten()
        # gra_phases_2[idx, :] = curr_gra_phases[:,:,1].flatten()
        grid_spikes[idx] = copy.deepcopy(curr_grid_spikes)
        # gra_spikes[idx] = copy.deepcopy(curr_gra_spikes)
    grid_phases = np.vstack((grid_phases_1, grid_phases_2))
    gra_phases = np.vstack((gra_phases_1, gra_phases_2))
    return grid_phases, gra_phases, grid_spikes, gra_spikes, rate_trajs, dt_s, theta_phase, phase_code_dir, overall_dir,firing_phase_dir,dist_trajs,traj_dist_dir,direction, spacings, grids  

def overall_spike_ct(grid_spikes, gra_spikes, dur_ms, seed_2s, n_traj=2):
    n_bin = int(dur_ms/bin_size)
    dur_s = dur_ms/1000
    counts_grid_1 = np.zeros((len(seed_2s), n_bin*n_grid))
    counts_grid_2 = np.zeros((len(seed_2s), n_bin*n_grid))
    counts_gra_1 = np.zeros((len(seed_2s), n_bin*n_granule))
    counts_gra_2 = np.zeros((len(seed_2s), n_bin*n_granule))

    for idx, seed_2 in enumerate(seed_2s):
        counts_grid_1[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
        counts_grid_2[idx,:] = binned_ct(grid_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
        # counts_gra_1[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,0].flatten()
        # counts_gra_2[idx,:] = binned_ct(gra_spikes[idx], bin_size, time_ms=dur_ms)[:,:,1].flatten()
    counts_grid = np.vstack((counts_grid_1, counts_grid_2))
    counts_granule = np.vstack((counts_gra_1, counts_gra_2))
    return counts_grid, counts_granule   

grid_phases_sim, gra_phases_sim, grid_spikes_sim, gra_spikes_sim, rate_trajs_sim, dt_s, theta_phase_sim, phase_code_dir_sim, overall_dir_sim,firing_phase_sim,dist_trajs_sim,traj_dist_dir_sim, direction_sim,spacings_sim, grids_sim = phase_code(sim_traj, dur_ms, grid_seeds, poiss_seeds, pp_weight, speed_cm)
grid_phases_diff, gra_phases_diff, grid_spikes_diff, gra_spikes_diff, rate_trajs_diff, dt_s, theta_phase_diff, phase_code_dir_diff, overall_dir_diff,firing_phase_diff,dist_trajs_diff,traj_dist_dir_diff,direction_diff, spacings_diff, grids_diff = phase_code(diff_traj, dur_ms, grid_seeds, poiss_seeds, pp_weight, speed_cm)
#grid and granule spike counts \ rate codes
grid_sim_traj_cts, gra_sim_traj_cts = overall_spike_ct(grid_spikes_sim, gra_spikes_sim, dur_ms, poiss_seeds, n_traj=n_traj)
grid_diff_traj_cts, gra_diff_traj_cts = overall_spike_ct(grid_spikes_diff, gra_spikes_diff, dur_ms, poiss_seeds, n_traj=n_traj)
 
grid_rate_code, grid_phase_code, grid_complex_code = perceptron(grid_sim_traj_cts, grid_diff_traj_cts, grid_phases_sim, grid_phases_diff, perc_seeds, lr_grid)
gra_rate_code, gra_phase_code, gra_complex_code = perceptron(gra_sim_traj_cts, gra_diff_traj_cts, gra_phases_sim, gra_phases_diff, perc_seeds, lr_gra)

plt.figure()
plt.plot(grid_sim_traj_cts[0])

cts_for_phase_sim = copy.deepcopy(grid_sim_traj_cts)
cts_for_phase_sim[cts_for_phase_sim!=0]=np.mean(cts_for_phase_sim[cts_for_phase_sim!=0])

plt.figure()
plt.plot(cts_for_phase_sim[0])

np.mean(cts_for_phase_sim)

plt.figure()
plt.plot(overall_dir_sim[0,:,0])
plt.plot(phase_code_dir_sim[0,:,0])
plt.plot(theta_phase_sim[0,:,0])
plt.plot(firing_phase_sim[0,:,0])
plt.plot(dist_trajs_sim[0,:,0])
plt.plot(traj_dist_dir_sim[0,:,0])
plt.plot(direction_sim[0,:,0])

traj_dist = direction_sim*dist_trajs_sim


traj_dist = ndimage.gaussian_filter(traj_dist, sigma=1)
traj_dist, dist_t_arr = interp(traj_dist, dur_s, def_dt_s, new_dt_s)
plt.figure()
plt.plot(traj_dist[0,:,0])


plt.figure()
plt.plot(overall_dir_sim[5,:,1])
plt.plot(phase_code_dir_sim[5,:,0])
plt.plot(theta_phase_sim[5,:,0])
plt.figure()
plt.imshow(grids_sim[:,:,5])

plt.figure()
plt.imshow(grids_sim[:,:,0])



rate_75 = grid_rate_code[0,:,0]
rate_75_2 = grid_rate_code[1,:,0]
rate_745 = grid_rate_code[2,:,0]
rate_74 = grid_rate_code[0,:,1]
rate_65 = grid_rate_code[2,:,1]
pr_rate = pearsonr(rate_75, rate_745)
pr_rate1 = pearsonr(rate_75, rate_75_2)
pr_rate2 = pearsonr(rate_75, rate_74)
pr_rate3 = pearsonr(rate_75, rate_65)

phase_75 = grid_phase_code[0,:,0]
phase_75_2 = grid_phase_code[1,:,0]
phase_745 = grid_phase_code[2,:,0]
phase_74 = grid_phase_code[0,:,1]
phase_65 = grid_phase_code[2,:,1]
pr_phase = pearsonr(phase_75, phase_745)
pr_phase1 = pearsonr(phase_75, phase_75_2)
pr_phase2 = pearsonr(phase_75, phase_74)
pr_phase3 = pearsonr(phase_75, phase_65)


complex_75 = grid_complex_code[0,:,0]
complex_75_2 = grid_complex_code[1,:,0]
complex_745 = grid_complex_code[2,:,0]
complex_74 = grid_complex_code[0,:,1]
complex_65 = grid_complex_code[2,:,1]
pr_complex = pearsonr(complex_75, complex_745)
pr_complex1 = pearsonr(complex_75, complex_75_2)
pr_complex2 = pearsonr(complex_75, complex_74)
pr_complex3 = pearsonr(complex_75, complex_65)


from scipy.spatial import distance

def cosine_mean(x,y):
    cos_mean = np.mean([np.mean(x), np.mean(y)])
    return 1 - distance.cosine(x-cos_mean, y-cos_mean)

    


1- distance.cosine(rate_75, rate_745)
1- distance.cosine(rate_75, rate_75_2)
1- distance.cosine(rate_75, rate_74)
1- distance.cosine(rate_75, rate_65)


1- distance.cosine(phase_75, phase_745)
1- distance.cosine(phase_75, phase_74)
1- distance.cosine(phase_75, phase_75_2)
1- distance.cosine(phase_75, phase_65)


cosine_mean(phase_75, phase_745)
cosine_mean(phase_75, phase_74)
cosine_mean(phase_75, phase_75_2)
cosine_mean(phase_75, phase_65)

1- distance.cosine(complex_75, complex_745)
1- distance.cosine(complex_75, complex_75_2)
1- distance.cosine(complex_75, complex_74)
1- distance.cosine(complex_75, complex_65)

1- distance.correlation(complex_75, complex_745)
1- distance.correlation(complex_75, complex_75_2)
1- distance.correlation(complex_75, complex_74)
1- distance.correlation(complex_75, complex_65)


plt.figure()
plt.plot(grid_phase_code[0,:,0])
plt.plot(grid_phase_code[2,:,0])


plt.figure()
plt.plot(grid_phases_sim[0,:])
plt.plot(grid_phases_sim[2,:])
pearsonr(grid_phases_sim[0,:][2200:2400],grid_phases_sim[2,:][2200:2400])

pearsonr(phase_75[4000:4200],phase_745[4000:4200])
pearsonr(phase_75[7000:7200],phase_745[7000:7200])
plt.figure()
plt.plot(firing_phase_sim[0,:,0])
plt.plot(firing_phase_diff[0,:,1])


plt.figure()
plt.eventplot(grid_phases_sim[:,2200:2400], lineoffsets=[1,2,3,4])
plt.figure()
plt.eventplot(grid_phases_sim[:,3600:3800], lineoffsets=[1,2,3,4])
plt.eventplot(grid_phases_sim[0,:][2200:2400])
plt.eventplot(grid_phases_sim[2,:][2200:2400])

plt.plot(grid_phase_code[2,:,0])
plt.figure()
xx, yy = np.meshgrid(np.arange(0, 200, 1), np.arange(0, 200, 1))
fig = plt.figure()
ax = plt.gca(projection = '3d')
surf = ax.plot_surface(xx, yy, grids_sim[:,:,0])

# speed = np.diff(dist_trajs_sim, axis=1)
# speed = np.concatenate((speed, speed[:,-1:,:]), axis=1)
# speed = speed*1000

#     direction[direction < 0] = -1
#     direction[direction > 0] = 1
#     direction[direction == 0] = -1
#     direction = -direction

# plt.plot(speed[0,:,0])
# plt.plot(-traj_dist_dir[0,:,0])

# traj_dist_dir = speed*dist_trajs_sim

plt.figure()
xs = np.arange(0,11,1)
ys = np.sqrt(np.array([200,164,136,116,104,100,104,116,136,164,200]))
plt.plot(xs,ys)

direc = np.arange(-1,1.1,0.2)

plt.plot(ys[:10]*np.diff(ys))

plt.plot(15*np.diff(ys))


data = np.load('rate_n_phase_codes_diff-poiss_traj_72-71-65-60_net-seed410_200ms.npz', allow_pickle=True)

grid_phase_code = data['grid_phase_code']
gra_phase_code = data['gra_phase_code']
grid_rate_code = data['grid_rate_code']
gra_rate_code = data['gra_rate_code']
grid_complex_code = data['grid_complex_code']
gra_complex_code = data['gra_complex_code']


grid_phase_code.shape
gra_rate_code.shape

pearsonr(grid_phase_code[0,400:800,0], grid_phase_code[2,400:800,0])
pearsonr(gra_phase_code[0,4000:8000,0], gra_phase_code[2,4000:8000,0])

pearsonr(grid_rate_code[0,400:800,0], grid_rate_code[2,400:800,0])
pearsonr(gra_rate_code[0,4000:8000,0], gra_rate_code[2,4000:8000,0])

pearsonr(grid_complex_code[0,400:800,0],grid_complex_code[2,400:800,0])
pearsonr(gra_complex_code[0,4000:8000,0], gra_complex_code[2,4000:8000,0])




