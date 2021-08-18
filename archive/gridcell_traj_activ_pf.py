#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:13:13 2020

@author: bariskuru
"""

import os 
import numpy as np
from scipy.stats import skewnorm
from skimage.measure import profile_line


time_sec = int(field_size_cm/speed_cm)
time_ms = time_sec*1000
start_cm = 75
dt = (time_sec)/arr_size #for parallel one
dt_ms = dt*1000
size2cm = arr_size/field_size_cm
start_idc = int((size2cm)*(start_cm)-1)

def grid_maker(spacing, orientation, pos_peak, arr_size, sizexy, max_rate, seed_1):
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
    arr = np.ones(dims)
    
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
            curr_pos = np.array([i,j]-pos_peak)
            arr[i,j] = (np.cos(np.dot(k1, curr_pos))+
               np.cos(np.dot(k2, curr_pos))+ np.cos(np.dot(k3, curr_pos)))/3
    arr = max_rate*2/3*(arr+1/2)   # arr is the resulting 2d grid out of 3 gratings      
    return arr

# skewed normal distribution for grid_spc
median_spc = 43
spc_max = 100
max_value = spc_max - 24 #24 is the apprx (median - np.median(random)) before locating median of dist 
skewness = 6  #Negative values are left skewed, positive values are right skewed.
grid_spc = skewnorm.rvs(a = skewness,loc=spc_max, size=n_grid)  #Skewnorm function
grid_spc = grid_spc - min(grid_spc)      #Shift the set so the minimum value is equal to zero.
grid_spc = grid_spc / max(grid_spc)      #Standadize all the vlues between 0 and 1. 
grid_spc = grid_spc * spc_max         #Multiply the standardized values by the maximum value.
grid_spc = grid_spc + (median_spc - np.median(grid_spc))

grid_ori = np.random.randint(0, high=60, size=[n_grid,1]) #uniform dist btw 0-60 degrees
grid_phase = np.random.randint(0, high=(arr_size-1), size=[n_grid,2]) #uniform dist grid phase

# create a 3d array with grids for n_grid
all_grids = np.zeros((arr_size, arr_size, n_grid))#empty array
for i in range(n_grid):
    x = grid_phase[i][0]
    y = grid_phase[i][1]
    arr = grid_maker(grid_spc[i], grid_ori[i], [x, y], arr_size, [1,1], 20, seed_1)
    all_grids[:, :, i] = arr

mean_grid = np.mean(all_grids, axis=2)




"TRAJECTORIES drawn for a moving mouse"
# in field_size_cm*field_size_cm field
#result is the activation profile of each cell in all_grids

#commented out part for tilt_traj are for variable uncut lenght of trajs

#empty arrays
traj = np.empty((n_grid,arr_size))
traj2 = np.empty((n_grid,arr_size))
par_traj = np.empty((n_grid,arr_size,8))
tilt_traj = np.empty((n_grid,arr_size,8))
# tilt_traj = [] 

#indices for parallel and tilted trajectories
x = np.arange(n_traj-1)
par_idc = np.insert(start_idc-(size2cm*(2**x)), 0, start_idc)
par_idc = par_idc.astype(int)
par_idc_cm = ((par_idc+1)/size2cm).astype(int)
dev_deg = (2**x)
dev_deg = np.insert(dev_deg,0,0)
dev_deg[7] = 36.999
radian = np.radians(dev_deg)
deviation = np.round(arr_size*np.tan(radian))
deviation = deviation.astype(int)
tilt_idc = start_idc-deviation

#draw the trajectories

for j in range(n_traj):
    idc = par_idc[j]
    tilt = tilt_idc[j]
    for i in range(n_grid):
        traj[i,:] = profile_line(all_grids[:,:,i], (idc,0), (idc,arr_size-1))
        traj2[i,:] = profile_line(all_grids[:,:,i], (start_idc,0), (tilt, arr_size-1))[:arr_size]
        # sloping trajectories are cut down to the same length of array here
        # traj2.append(profile_line(all_grids[:,:,i], (start_idc,0), (tilt, arr_size-1)))
    par_traj[:,:,j] = traj
    tilt_traj[:,:,j]= traj2
    # tilt_traj.append(traj2)
    # traj2 = []
    
cum_par = np.sum(par_traj, axis=1)*(dt)
cum_tilt = np.sum(tilt_traj, axis=1)*(dt)


