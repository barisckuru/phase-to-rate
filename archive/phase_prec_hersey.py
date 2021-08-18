#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:40:05 2020

@author: bariskuru
"""
import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from scipy import interpolate
from skimage.measure import profile_line
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
from scipy import ndimage

from matplotlib.colors import ListedColormap



def grid_maker(spacing, orientation, pos_peak, arr_size, sizexy, max_rate):
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

sns.set(context='paper', palette='RdYlBu_r', font='Calibri',font_scale=1.5,color_codes=True)

spacing = 30
orientation = 30
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
max_freq = 1
rate, g1, g2, g3 = grid_maker(spacing, orientation, [100, 100], arr_size, [field_size_m,field_size_m], max_freq)
sns.reset_orig()




##########################
'''Sinusoidal gratings and grid field'''
spacing = 30
orientation = 30
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
max_freq = 1
rate, g1, g2, g3 = grid_maker(spacing, orientation, [100, 100], arr_size, [field_size_m,field_size_m], max_freq)
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
spacing = 90
orientation = 30
field_size_m = 1
field_size_cm = field_size_m*100
arr_size = 200
max_freq = 1
rate, g1, g2, g3 = grid_maker(spacing, 15, [100, 100], arr_size, [field_size_m,field_size_m], max_freq)
xx, yy = np.meshgrid(np.arange(0, 100, 0.5), np.arange(0, 100, 0.5))
f2= plt.figure()
im1 = f2.gca(projection = '3d')
surf1 = im1.plot_surface(xx, yy, rate, cmap=cmap)
im1.set_xlabel('cm', fontsize=20)
im1.set_ylabel('cm', fontsize=20)
im1.set_zlabel('normalized rate', fontsize=20, rotation=45)

dist =  (np.arccos(((rate*3/(2*max_freq))-1/2))*np.sqrt(2))*np.sqrt(6)/(4*np.pi)
xx, yy = np.meshgrid(np.arange(0, 100, 0.5), np.arange(0, 100, 0.5))
f3= plt.figure()
im2 = f3.gca(projection = '3d')
surf2 = im2.plot_surface(xx, yy, dist, cmap=cmap)
im2.set_xlabel('cm', fontsize=20)
im2.set_ylabel('cm', fontsize=20)
im2.set_zlabel('normalized distance', fontsize=20, rotation=45)
#############################

dist = ndimage.gaussian_filter(dist, sigma=3)


plt.close()
plt.figure()
plt.imshow(arr, cmap=cmap, interpolation=None, extent=[0,100,100,0])
# plt.plot(np.arange(0,100,1), 50*np.ones(100), 'r')
# plt.title('2D rate profile of a grid cell \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('cm')
plt.ylabel('cm')
cbar = plt.colorbar()
cbar.set_label('Hz', rotation=270)
plt.figure()
trans_dist_2d = spacing/2 * np.arccos(3*arr/40-0.5)/2
plt.imshow(trans_norm_2d, cmap=cmap, interpolation=None, extent=[0,100,100,0])
plt.plot(np.arange(0,100,1), 50*np.ones(100), 'r')
plt.title('Transformed distances from peaks in 2D \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('cm')
plt.ylabel('cm')
cbar = plt.colorbar()
cbar.set_label('cm', rotation=270)

sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)

traj = profile_line(arr, (99,0), (99,200-1))
plt.figure()
plt.plot(traj)
plt.xticks(np.arange(40,240,40), np.arange(20,120,20))
plt.title('Rate profile on the trajectory \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Frequency (Hz)')


traj2 = profile_line(trans_dist_2d, (99,0), (99,200-1))
plt.figure()
plt.plot(traj2)
plt.xticks(np.arange(40,240,40), np.arange(20,120,20))
plt.title('Distance from closest grid peak \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Distance (cm)')


traj3 = profile_line(trans_norm_2d, (99,0), (99,200-1))
plt.figure()
plt.plot(traj3)
plt.xticks(np.arange(40,240,40), np.arange(20,120,20))
plt.title('Distance from closest grid peak \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Distance (cm)')


traj2[np.argmax(traj2)]



plt.close('all')
traj_rate = profile_line(arr, (99,0), (99,200-1))
# trans_norm_2d = np.arccos(3*arr/40-0.5)/2
trans_dist_2d = (np.arccos(((arr*3/(2*max_freq))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)

trans_dist_2d = (np.arccos(((arr*3/(2*max_freq))-1/2))*(np.sqrt(3)*spacing)/(4*np.pi))
trans_norm_2d = (trans_dist_2d/(spacing/2))/2
trans_norm_2d = trans_dist_2d
traj_loc = profile_line(trans_norm_2d, (99,0), (99,200-1), mode='constant')
traj_rate = profile_line(arr, (99,0), (99,200-1))

plt.figure()
plt.plot(profile_line(trans_dist_2d, (99,0), (99,200-1))/(spacing/2))
plt.figure()
plt.plot(traj_loc)

#interpolation
def interp(arr, def_dur_s, new_dt_s):
    arr_len = arr.shape[0]
    t_arr = np.linspace(0, def_dur_s, arr_len)
    new_len = int(def_dur_s/new_dt_s)
    new_t_arr = np.linspace(0, def_dur_s, new_len)
    interp_arr = np.interp(new_t_arr, t_arr, arr)
    return interp_arr, new_t_arr


new_dt_s = 0.002
dur_s = 5
def_dt = dur_s/traj_loc.shape[0]
# traj_loc, loc_t_arr = interp(traj_loc, dur_s, new_dt_s)
# traj_loc = ndimage.gaussian_filter(traj_loc, sigma=1)
traj_rate, rate_t_arr = interp(traj_rate, dur_s, new_dt_s)

f=10
T = 1/f
def_time = np.arange(0, dur_s, def_dt)
time_hr = rate_t_arr
# shift = 3*np.pi/2
shift = 0
theta = (np.sin(f*2*np.pi*def_time+shift)+1)/2
theta_hr = (np.sin(f*2*np.pi*time_hr+shift)+1)/2

phase = (2*np.pi*(def_time%T)/T + shift)%(2*np.pi)
phase_hr = (2*np.pi*(time_hr%T)/T + shift)%(2*np.pi)


#infer the direction out of rate of change in the location
direction = np.diff(traj_loc)


# loc_hr = np.arange(0, field_size_cm, field_size_cm/traj_loc.shape[0])

#last element is same with the -1 element of diff array
direction = np.append(direction, direction[-1])

direction = np.diff(traj_loc)
#last element is same with the -1 element of diff array
direction = np.append(direction, direction[-1])
direction[direction < 0] = -1
direction[direction > 0] = 1
direction[direction == 0] = 1
direction = -direction
# direction[np.logical_and(direction < 0.1, direction > -0.1)] = 1
direction = ndimage.gaussian_filter(direction, sigma=2)

threshold = 0.00000
# direction[direction < -threshold] = -1
# direction[direction > threshold] = 1
# direction = -direction*(1/max(direction))
# # direction = ndimage.gaussian_filter(direction, sigma=3)
# direction[np.logical_and(direction < 0.1, direction > -0.1)] = 1
# direction = ndimage.gaussian_filter(direction, sigma=2)



traj_loc_dir = traj_loc*direction
traj_loc_dir = ndimage.gaussian_filter(traj_loc_dir, sigma=2)
traj_loc_dir, loc_t_arr = interp(traj_loc_dir, dur_s, new_dt_s)

loc_hr = np.arange(0, field_size_cm, field_size_cm/traj_loc_dir.shape[0])

factor = 200/360
firing_phase_dir = 2*np.pi*(traj_loc_dir+0.5)*factor
phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-phase_hr))
overall_dir = phase_code_dir*traj_rate*0.16*20*2

plt.figure()
plt.plot(np.arange(0,200,1), direction)

plt.title('Rate of change at distance from the closest peak \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Rate of change (normalized)')


plt.figure()
plt.plot(loc_hr, direction)
plt.title('Direciton of movement in reference to the closest peak \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Binary Direction')

# direction defines if animal goes into a grid field or goes out of a grid field
plt.figure()
plt.plot(loc_hr, traj_loc_dir)
plt.title('Relative location in reference to the closest peak \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Trajectory (cm)')
plt.ylabel('Location/Spacing')

plt.figure()
plt.plot(time_hr, firing_phase_dir/np.pi)
plt.title('Preferred Firing Phase \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Time (s)')
plt.ylabel('Phase (pi)')

plt.figure()
plt.plot(time_hr, phase_code_dir)
plt.title('Phase Code \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Time (s)')
plt.ylabel('Phase code')



plt.figure()

f = plt.figure(figsize=(27,12))
ax = f.add_subplot(211)
# plt.plot(rate_t_arr, traj_rate)
plt.plot(time_hr, overall_dir)
# plt.xticks(np.arange(40,240,40), np.arange(20,120,20))
plt.title('Rate profile on the trajectory  \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Time (s)')
plt.ylabel('Frequency(normalized)')

ax2 = f.add_subplot(212)
plt.plot(time_hr, phase_code_dir/max(phase_code_dir), label='MPO')
plt.plot(time_hr, theta_hr, label='LFP')
plt.title('Phase Precesion')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Phase')
plt.legend()





#until heee

plt.figure()


time_plt = np.arange(0, dur_s, new_dt_s)



plt.figure()
plt.plot(time_hr, overall_dir)
plt.title('Firing Rate \n grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Time (s)')
plt.ylabel('Hz')

plt.figure()
plt.plot(phase)
plt.plot(theta)
plt.plot(time_hr, 4*theta_hr)

plt.figure()
plt.plot(traj_rate)


###### poisson spikes

dt_s = 0.002
t_sec = 5
norm_overall = overall_dir
asig = AnalogSignal(norm_overall,
                                    units=1*pq.Hz,
                                    t_start=0*pq.s,
                                    t_stop=t_sec*pq.s,
                                    sampling_period=dt_s*pq.s,
                                    sampling_interval=dt_s*pq.s)


time_bin_size = 0.1
times = np.arange(0, 5+time_bin_size, time_bin_size) 

n_time_bins = int(t_sec/time_bin_size)
bins_size_deg = 1
n_phase_bins = int(720/bins_size_deg)

phases = [ [] for _ in range(n_time_bins) ]
n = 5000
for i in range(n):
    train = np.array(stg.inhomogeneous_poisson_process(asig))
    # phase_deg.append(train%T/(t_sec*T)*360)
    for j, time in enumerate(times):
        if j == times.shape[0]-1:
            break
        curr_train = train[np.logical_and(train > time, train < times[j+1])]
        # if curr_train != []:
        if curr_train.size>0:
            phases[j] += list(curr_train%(T)/(T)*360)
            phases[j] += list(curr_train%(T)/(T)*360+360)
            
# f1 = plt.figure(figsize=(18,8))
# plt.eventplot(phases, lineoffsets=np.linspace(time_bin_size,t_sec,n_time_bins), linelengths=0.07, linewidths = 1, orientation='vertical')
# # train = np.array(stg.inhomogeneous_poisson_process(asig))
# plt.title('Phases of Poisson Spikes \n' +str(n)+' trials, grid spacing = ' + str(spacing)+' cm')
# plt.xlabel('Time (s)')
# plt.ylabel('Phase (deg)')


counts = np.empty((n_phase_bins, n_time_bins))
                  
for i in range(n_phase_bins):
    for j, phases_in_time in enumerate(phases):
        phases_in_time = np.array(phases_in_time) 
        counts[i][j] = ((bins_size_deg*(i) < phases_in_time) & (phases_in_time < bins_size_deg*(i+1))).sum()
f=10
# norm_factor = 75
# norm_freq = counts*f*n_phase_bins/n
norm_freq = counts*f/n
f2 = plt.figure(figsize=(27,12))
plt.imshow(norm_freq, aspect=1/7, cmap='RdYlBu_r', extent=[0,100,720,0])
plt.ylim((0,720))
#, extent=[0,100,0,720]
plt.title('Mean Firing Frequency ('+str(n)+ ' trials)')
         # \n phase precession=240deg, spacing='+str(spacing)+'cm, timebin=0.100s, phasebin=1deg')
plt.xlabel('Position (cm)')
plt.ylabel('Theta phase (deg)')
cbar = plt.colorbar()
cbar.set_label('Hz', rotation=270)
# liste = times

# plt.imshow(counts*f/n, cmap='RdYlBu_r')






f2 = plt.figure(figsize=(27,12))
ax3 = f2.add_subplot(211)
plt.plot(time_hr, overall_dir)
plt.title('Overall Firing Rate \n grid spacing = ' + str(spacing)+' cm')
# plt.xlabel('Time (s)')
plt.ylabel('Hz')

ax4 = f2.add_subplot(212, sharex=ax3)
plt.eventplot(phases, lineoffsets=np.linspace(T,t_sec,40), linelengths=0.07, linewidths = 1, orientation='vertical')
# train = np.array(stg.inhomogeneous_poisson_process(asig))
plt.title('Phases of Poisson Spikes \n' +str(n)+' trials, grid spacing = ' + str(spacing)+' cm')
plt.xlabel('Time (s)')
plt.ylabel('Phase (deg)')


a_train = np.array(stg.inhomogeneous_poisson_process(asig))
np.sum(np.logical_and(a_train>2, a_train<3))

b_train = np.array(stg.inhomogeneous_poisson_process(asig))

plt.eventplot(phase_deg, lineoffsets=np.arange(1,n+1,1))
plt.title('Poisson Spikes')
plt.xlabel('Time (s)')
plt.ylabel('Trials w diff seeds')



from scipy import integrate

y_int = integrate.cumtrapz(overall_dir, time_hr, initial=0)
plt.figure()
plt.plot(time_hr, y_int)


np.mean(overall_dir[1150:1350])

np.mean(overall_dir[199:1100])

plt.figure()
plt.plot(np.arange(0,200,1), overall_dir[1150:1350])
