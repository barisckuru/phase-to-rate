#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 11:31:09 2021

@author: baris
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from rate_n_phase_code_gra import rate2dist, _direction, interp, inhom_poiss
from scipy import interpolate
from skimage.measure import profile_line
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
from scipy import ndimage

def interp(arr, def_dur_s, new_dt_s):
    arr_len = arr.shape[0]
    t_arr = np.linspace(0, def_dur_s, arr_len)
    new_len = int(def_dur_s/new_dt_s)
    new_t_arr = np.linspace(0, def_dur_s, new_len)
    interp_arr = np.interp(new_t_arr, t_arr, arr)
    return interp_arr, new_t_arr

def inhom_poiss(rate_profile, dur_s, poiss_seeds, dt_s=0.002):
    rate_profile = rate_profile
    asig = AnalogSignal(rate_profile,
                            units=1*pq.Hz,
                            t_start=0*pq.s,
                            t_stop=dur_s*pq.s,
                            sampling_period=dt_s*pq.s,
                            sampling_interval=dt_s*pq.s)
    n_poiss = poiss_seeds.shape[0]
    spi_arr = np.zeros((n_poiss), dtype = np.ndarray)
    for idx, seed in enumerate(poiss_seeds):
        np.random.seed(seed)
        curr_train = stg.inhomogeneous_poisson_process(asig)
        spi_arr[idx] = np.array(curr_train.times) #time conv to ms
    return spi_arr


def overall(spacing, center, orientation, dur):
    spacing = spacing
    center = center
    orientation = orientation
    field_size_m = 1
    field_size_cm = field_size_m*100
    arr_size = 200
    max_rate = 1
    new_dt_s = 0.002
    dur_s = dur
    f=10
    T = 1/f
    arr = grid_maker(spacing, orientation, center, arr_size, [field_size_m,field_size_m], max_rate)
    
    
    trans_dist_2d = (np.arccos(((arr*3/(2*max_rate))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)
    trans_norm_2d = (trans_dist_2d/(spacing/2))
    
    
    traj_loc = profile_line(trans_norm_2d, (99,0), (99,(40*dur)-1), mode='constant')
    traj_rate = profile_line(arr, (99,0), (99,(40*dur)-1))
    
    
    
    def_dt = dur_s/traj_loc.shape[0]
    traj_rate, rate_t_arr = interp(traj_rate, dur_s, new_dt_s)
    def_time = np.arange(0, dur_s, def_dt)
    time_hr = rate_t_arr
    shift = 0
    theta = (np.sin(f*2*np.pi*def_time+shift)+1)/2
    theta_hr = (np.sin(f*2*np.pi*time_hr+shift)+1)/2
    phase = (2*np.pi*(def_time%T)/T + shift)%(2*np.pi)
    phase_hr = (2*np.pi*(time_hr%T)/T + shift)%(2*np.pi)
    
    
    #direction
    direction = np.diff(traj_loc)
    direction = np.append(direction, direction[-1])
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    direction = -direction
    
    traj_loc_dir = traj_loc*direction
    traj_loc_dir = ndimage.gaussian_filter(traj_loc_dir, sigma=1)
    traj_loc_dir, loc_t_arr = interp(traj_loc_dir, dur_s, new_dt_s)
    loc_hr = np.arange(0, field_size_cm, field_size_cm/traj_loc_dir.shape[0])
    
    factor = 180/360
    firing_phase_dir = 2*np.pi*(traj_loc_dir+0.5)*factor
    phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-phase_hr))
    overall_dir = phase_code_dir*traj_rate*0.16*20*5
    return overall_dir, arr, time_hr

'Trajectory -Overall - Spikes'
###################################
# spacing = 20
# center = [100,100]
# orientation = 30
# dur = 2
# overall_dir, rate, time_hr = overall(spacing, center, orientation, dur)
# poiss_seeds = np.arange(105,110,1)
# spikes = inhom_poiss(overall_dir, dur, poiss_seeds)

# plt.close('all')
# sns.reset_orig()
# sns.set(style='dark', palette='deep', font='Arial',font_scale=1.5,color_codes=True)
# cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
# f1, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw={'height_ratios':[1,1]}, figsize=(12,8))

# f2, ax = plt.subplots(figsize=(11,5))
# plt.imshow(rate[80:120,:80],cmap=cmap, extent=[0,40,60,40], aspect='auto')
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_title('Trajectory')

# ax1.plot(time_hr, overall_dir)
# ax1.set_title('Overall Rate Profile')
# ax1.set_ylabel('Frequency (Hz)')
    
# ax2.eventplot(spikes, linewidth=0.7, linelengths=0.5)
# ax2.set_title('Spikes')
# ax2.set_yticks(np.arange(5))
# ax2.set_yticklabels(['seed1', 'seed2', 'seed3', 'seed4', 'seed5'])
# ax2.set_xlabel('Time (s)')

# xpos = np.arange(-0.02,dur+0.1,0.1)
# for xc in xpos:
#     ax1.axvline(xc, color='k', linestyle='-.', linewidth=0.5)
#     ax2.axvline(xc, color="0.5", linestyle='-.', linewidth=0.5)

# # f1.tight_layout(pad)
# save_dir = '/home/baris/figures/'
# f1.savefig(save_dir+'overall-firing_spacing_center_orientation_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.eps', dpi=200)
# f1.savefig(save_dir+'overall-firing_spacing_center_orientation_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.png', dpi=200)
# f2.savefig(save_dir+'trajectory_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.eps', dpi=200)
# f2.savefig(save_dir+'trajectory_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.png', dpi=200)


##############################################


def precession_spikes(overall, dur):
    dt_s = 0.002
    t_sec = dur
    norm_overall = overall_dir
    asig = AnalogSignal(norm_overall,
                                        units=1*pq.Hz,
                                        t_start=0*pq.s,
                                        t_stop=t_sec*pq.s,
                                        sampling_period=dt_s*pq.s,
                                        sampling_interval=dt_s*pq.s)
    
    time_bin_size = 0.1
    T=time_bin_size
    times = np.arange(0, 5+time_bin_size, time_bin_size) 
    
    n_time_bins = int(t_sec/time_bin_size)
    bins_size_deg = 7.2
    phase_norm_fact = 360/bins_size_deg
    n_phase_bins = int(720/bins_size_deg)
    
    phases = [ [] for _ in range(n_time_bins) ]
    n = 1000
    for i in range(n):
        train = np.array(stg.inhomogeneous_poisson_process(asig).times)
        # phase_deg.append(train%T/(t_sec*T)*360)
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = train[np.logical_and(train > time, train < times[j+1])]
            # if curr_train != []:
            if curr_train.size>0:
                phases[j] += list(curr_train%(T)/(T)*360)
                phases[j] += list(curr_train%(T)/(T)*360+360)
                
    
    counts = np.empty((n_phase_bins, n_time_bins))
                      
    for i in range(n_phase_bins):
        for j, phases_in_time in enumerate(phases):
            phases_in_time = np.array(phases_in_time) 
            counts[i][j] = ((bins_size_deg*(i) < phases_in_time) & (phases_in_time < bins_size_deg*(i+1))).sum()
    f=10
    norm_freq = counts*phase_norm_fact*f/n
    norm_freq = ndimage.gaussian_filter(norm_freq, sigma=[1,1])
    return norm_freq

##############################
'phase precession figures'
spacing = 40
center = [100,100]
orientation = 25
dur = 5

overall_dir, rate, _ = overall(spacing, center, orientation, dur)
spike_phases = precession_spikes(overall_dir, dur)

cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
cmap2 = sns.color_palette('RdYlBu_r', as_cmap=True)
f2, (ax1, ax2) = plt.subplots(2,1, sharex=False, gridspec_kw={'height_ratios':[1,2]}, figsize=(7,9))
# f2.tight_layout(pad=0.1)
im2 = ax2.imshow(spike_phases, aspect=1/7, cmap=cmap, extent=[0,100,720,0])
ax2.set_ylim((0,720))
#, extent=[0,100,0,720]
# ax2.title('Mean Firing Frequency ('+str(n)+ ' trials)\n spacing=' + str(spacing)+' cm'
#           +'    phase='+str(center[0])+'    orientation='+str(orientation))
im1 = ax1.imshow(rate[50:150,:], cmap=cmap2)
#, extent=[0,100,80,20]
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title('\u0394=' + str(spacing)+'       [$x_{0}$ $y_{0}$]='+str(100-np.array(center))+'       \u03A8='+str(orientation), loc='center')
         # \n phase precession=240deg, spacing='+str(spacing)+'cm, timebin=0.100s, phasebin=1deg')
ax2.set_xlabel('Position (cm)')
ax2.set_ylabel('Theta phase (deg)')
f2.subplots_adjust(right=0.8)
# cax = f2.add_axes([0.60,0.16,0.015,0.35])
cax = f2.add_axes([0.80,0.16,0.025,0.35])
cbar = f2.colorbar(im2, cax=cax)
cbar.set_label('Hz', labelpad=15, rotation=270)

save_dir = '/home/baris/figures/'
f2.savefig(save_dir+'phase_prec_spacing_center_orientation_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.eps', dpi=200)
f2.savefig(save_dir+'phase_prec_spacing_center_orientation_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.png', dpi=200)
#######################################################
'''

#grid maker with gratings
def grid_maker_grating(spacing, orientation, pos_peak, arr_size, sizexy, max_rate):
    #define the params from input here, scale the resulting array for maxrate and sperate the xy for size and shift
    arr_size = arr_size
    x, y = pos_peak
    pos_peak = np.array([x,y])
    max_rate = max_rate
    lambda_spacing = spacing*(arr_size/100) #100 required for conversion, they have probably used 100*100 matrix in 
    k = (4*np.pi)/(lambda_spacing*np.sqrt(3))
    degrees = orientation
    theta = np.pi*(degrees/180)
    meterx, metery = sizexy
    arrx = meterx*arr_size # *arr_size for defining the 2d array size
    arry = metery*arr_size
    dims = np.array([arrx,arry])
    rate = np.ones(dims)
    g1 = np.ones(dims)
    g2 = np.ones(dims)
    g3 = np.ones(dims)
    # dist = np.ones(dims)
    #implementation of grid function
    # 3 ks for 3 cos gratings with different angles
    k1 = ((k/np.sqrt(2))*np.array((np.cos(theta+(np.pi)/12) + np.sin(theta+(np.pi)/12),
          np.cos(theta+(np.pi)/12) - np.sin(theta+(np.pi)/12)))).reshape(2,)
    k2 = ((k/np.sqrt(2))*np.array((np.cos(theta+(5*np.pi)/12) + np.sin(theta+(5*np.pi)/12),
          np.cos(theta+(5*np.pi)/12) - np.sin(theta+(5*np.pi)/12)))).reshape(2,)
    k3 = ((k/np.sqrt(2))*np.array((np.cos(theta+(9*np.pi)/12) + np.sin(theta+(9*np.pi)/12),
          np.cos(theta+(9*np.pi)/12) - np.sin(theta+(9*np.pi)/12)))).reshape(2,)
    

    #.reshape is only need when function is in the loop(shape somehow becomes (2,1) otherwise normal shape is already (2,)
    for i in range(dims[0]):
        for j in range(dims[1]):
            curr_dist = np.array([i,j]-pos_peak)
            # dist[i,j] = (np.arccos(np.cos(np.dot(k1, curr_dist)))+
            #              np.arccos(np.cos(np.dot(k2, curr_dist)))+np.arccos(np.cos(np.dot(k3, curr_dist))))/3
            g1[i,j] = (np.cos(np.dot(k1, curr_dist))+1/2)*2/3
            g2[i,j] = (np.cos(np.dot(k2, curr_dist))+1/2)*2/3
            g3[i,j] = (np.cos(np.dot(k3, curr_dist))+1/2)*2/3
            rate[i,j] = (np.cos(np.dot(k1, curr_dist))+
               np.cos(np.dot(k2, curr_dist))+ np.cos(np.dot(k3, curr_dist)))/3
    rate = max_rate*2/3*(rate+1/2)   # arr is the resulting 2d grid out of 3 gratings
    # dist = (dist/np.pi)*(3/4)
    return rate, g1, g2, g3


##########################
'Sinusoidal gratings and grid field'
spacing = 30
orientation = 30
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
max_freq = 1
rate, g1, g2, g3 = grid_maker_grating(spacing, orientation, [100, 100], arr_size, [field_size_m,field_size_m], max_freq)
sns.reset_orig()
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
f1, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharey=True, figsize=(12,28))
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax4.set_xticklabels([])
ax4.set_yticklabels([])

im1 = ax1.imshow(g1, cmap=cmap, extent=[0,100,100,0])
im2= ax2.imshow(g3, cmap=cmap, extent=[0,100,100,0])
im3 =ax3.imshow(g2, cmap=cmap, extent=[0,100,100,0])
im4 = ax4.imshow(rate, cmap=cmap, extent=[0,100,100,0])
f1.subplots_adjust(right=0.8)
cax = f1.add_axes([0.83,0.40,0.01,0.18])
cbar = f1.colorbar(im4, cax=cax)
#################################




###############################
"3D grid fields - linear distance"
sns.reset_orig()
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
spacing = 90
orientation = 30
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
max_freq = 1
rate, g1, g2, g3 = grid_maker_grating(spacing, 15, [100, 100], arr_size, [field_size_m,field_size_m], max_freq)
xx, yy = np.meshgrid(np.arange(0, 100, 0.5), np.arange(0, 100, 0.5))
f2= plt.figure(figsize=(11,7))
im1 = f2.gca(projection = '3d')
surf1 = im1.plot_surface(xx, yy, rate, cmap=cmap)
im1.set_xlabel('cm', fontsize=20)
im1.set_ylabel('cm', fontsize=20)
im1.set_zlabel('normalized rate', fontsize=20, rotation=270)

dist =  (np.arccos(((rate*3/(2*max_freq))-1/2))*np.sqrt(2))*np.sqrt(6)/(4*np.pi)
xx, yy = np.meshgrid(np.arange(0, 100, 0.5), np.arange(0, 100, 0.5))
f3= plt.figure()
im2 = f3.gca(projection = '3d')
surf2 = im2.plot_surface(xx, yy, dist, cmap=cmap)
im2.set_xlabel('cm', fontsize=20)
im2.set_ylabel('cm', fontsize=20)
im2.set_zlabel('normalized distance', fontsize=20, rotation=270)

save_dir = '/home/baris/figures/'

fig1.savefig(save_dir+str(cell_name)+'_encoding_speed_dur_'+str(dur)+'_ngrid_'+str(n_grid_seed)+'_lr_'+str(lr)+'.eps', dpi=150)
fig1.savefig(save_dir+str(cell_name)+'_encoding_speed_dur_'+str(dur)+'_ngrid_'+str(n_grid_seed)+'_lr_'+str(lr)+'.png', dpi=150)


#############################


###########################
'Parallel trajectories'
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.figure()
ax = plt.gca()
im = ax.imshow(arr, cmap=cmap, interpolation=None, extent=[0,100,100,0])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='3%', pad=0.1)
# ax.set_title('Parallel Trajectories')
ax.set_xlabel('cm')
ax.set_ylabel('cm')
cbar = plt.colorbar(im, cax=cax)
ax.plot(np.arange(0,41,1), 75*np.ones(41), 'darkgreen',linestyle='dashdot', linewidth=4)
cbar.set_label('Normalized rate', labelpad=20, rotation=270)
trajectories = np.array([74.5, 74, 73.5, 73, 72.5, 72, 71.5, 71, 70, 65, 60])
for i in trajectories:
    ax.plot(np.arange(0,41,1), i*np.ones(41), 'darkgreen',linestyle='dashdot', linewidth=4)
################################
'''