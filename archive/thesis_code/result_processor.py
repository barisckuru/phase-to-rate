#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:47:37 2021

@author: baris
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.stats import pearsonr,  spearmanr
import os

#75-74.5-74-73.5
#73-72.5-72-71.5
#71-70-65-60

def all_seeds(path = '/home/baris/results/perceptron_th_n_codes/results_factor_5/nofb/71-70-65-60/'):
    dur_ms = 2000
    path = path
    npzfiles = []
    grid_rate_code =[]
    grid_phase_code =[]
    grid_complex_code =[]
    grid_th_cross = []
    gra_rate_code = []
    gra_phase_code= []
    gra_complex_code= []
    gra_th_cross= []
    for file in sorted(glob.glob(os.path.join(path,'*.npz'))):
        npzfiles.append(file)
        load = np.load(file, allow_pickle=True)
        grid_rate_code.append(load['grid_rate_code'])
        grid_phase_code.append(load['grid_phase_code'])
        grid_complex_code.append(load['grid_complex_code'])
        # grid_th_cross.append(load['grid_th_cross'])
        gra_rate_code.append(load['gra_rate_code'])
        gra_phase_code.append(load['gra_phase_code'])
        gra_complex_code.append(load['gra_complex_code'])
        # gra_th_cross.append(load['gra_th_cross'])
        
    sim_traj = load['sim_traj']
    diff_traj = load['diff_traj']
    np.savez(os.path.join(path,'rate_n_phase_traj_diff_poiss_'+str(sim_traj[0])+'-'+str(sim_traj[1])+'-'+str(diff_traj[0])+'-'+str(diff_traj[1])+'_net-seeds_410-429_'+str(dur_ms)+'ms'), 
             grid_rate_code = grid_rate_code,
             grid_phase_code = grid_phase_code,
             grid_complex_code = grid_complex_code,
             #grid_th_cross = grid_th_cross,
             
             gra_rate_code = gra_rate_code,
             gra_phase_code = gra_phase_code,
             gra_complex_code = gra_complex_code)
             #gra_th_cross = gra_th_cross)

# all_seeds()

def all_seed_spikes(path = '/home/baris/results/perceptron_th_n_codes/results_factor_5/nofb/71-70-65-60/'):
    dur_ms = 2000
    path = path
    npzfiles = []
    grid_rate_code =[]
    grid_phase_code =[]
    grid_complex_code =[]
    grid_th_cross = []
    gra_rate_code = []
    gra_phase_code= []
    gra_complex_code= []
    gra_th_cross= []
    for file in sorted(glob.glob(os.path.join(path,'*.npz'))):
        npzfiles.append(file)
        load = np.load(file, allow_pickle=True)
        grid_rate_code.append(load['grid_rate_code'])
        grid_phase_code.append(load['grid_phase_code'])
        grid_complex_code.append(load['grid_complex_code'])
        # grid_th_cross.append(load['grid_th_cross'])
        gra_rate_code.append(load['gra_rate_code'])
        gra_phase_code.append(load['gra_phase_code'])
        gra_complex_code.append(load['gra_complex_code'])
        # gra_th_cross.append(load['gra_th_cross'])
        
    sim_traj = load['sim_traj']
    diff_traj = load['diff_traj']
    np.savez(os.path.join(path,'rate_n_phase_traj_diff_poiss_'+str(sim_traj[0])+'-'+str(sim_traj[1])+'-'+str(diff_traj[0])+'-'+str(diff_traj[1])+'_net-seeds_410-429_'+str(dur_ms)+'ms'), 
             grid_rate_code = grid_rate_code,
             grid_phase_code = grid_phase_code,
             grid_complex_code = grid_complex_code,
             #grid_th_cross = grid_th_cross,
             
             gra_rate_code = gra_rate_code,
             gra_phase_code = gra_phase_code,
             gra_complex_code = gra_complex_code)
             #gra_th_cross = gra_th_cross)



def all_codes(path):
    grid_rate = []
    grid_phase = []
    grid_complex = []
    gra_rate = []
    gra_phase=[]
    gra_complex=[]
    npzfiles = []
    for file in sorted(glob.glob(os.path.join(path,'*.npz')), reverse=True):
        npzfiles.append(file)
        datum = np.load(file, allow_pickle=True)
        for i in range(2):
            grid_rate.append(datum['grid_rate_code'][:,:5,:,i])
            grid_rate.append(datum['grid_rate_code'][:,5:10,:,i])
            grid_phase.append(datum['grid_phase_code'][:,:5,:,i])
            grid_phase.append(datum['grid_phase_code'][:,5:10,:,i])
            grid_complex.append(datum['grid_complex_code'][:,:5,:,i])
            grid_complex.append(datum['grid_complex_code'][:,5:10,:,i])
            
            gra_rate.append(datum['gra_rate_code'][:,:5,:,i])
            gra_rate.append(datum['gra_rate_code'][:,5:10,:,i])
            gra_phase.append(datum['gra_phase_code'][:,:5,:,i])
            gra_phase.append(datum['gra_phase_code'][:,5:10,:,i])
            gra_complex.append(datum['gra_complex_code'][:,:5,:,i])
            gra_complex.append(datum['gra_complex_code'][:,5:10,:,i])
            
    grid_rate = np.array(grid_rate)
    grid_phase = np.array(grid_phase)
    grid_complex = np.array(grid_complex)
    gra_rate = np.array(gra_rate)
    gra_phase = np.array(gra_phase)
    gra_complex = np.array(gra_complex)
    return grid_rate, grid_phase, grid_complex, gra_rate, gra_phase, gra_complex

def all_phases(path = '/home/baris/results/perceptron_th_n_codes/results_factor_5/noinh/71-70-65-60/'):
    grid_rate = []
    grid_phase = []
    gra_rate = []
    gra_phase=[]
    npzfiles = []
    for file in sorted(glob.glob(os.path.join(path,'*.npz')), reverse=True):
        npzfiles.append(file)
        datum = np.load(file, allow_pickle=True)
        grid_rate.append(datum['grid_sim_traj_cts'])
        # grid_rate.append(datum['grid_diff_traj_cts'][:,5:10,:,i])
        grid_phase.append(datum['grid_phases_sim'])
        # grid_phase.append(datum['grid_phases_diff'][:,5:10,:,i])
        
        gra_rate.append(datum['gra_sim_traj_cts'])
        # gra_rate.append(datum['gra_diff_traj_cts'][:,5:10,:,i])
        gra_phase.append(datum['gra_phases_sim'])
        # gra_phase.append(datum['gra_phases_diff'][:,5:10,:,i])

    grid_rate = np.array(grid_rate)
    grid_phase = np.array(grid_phase)
    gra_rate = np.array(gra_rate)
    gra_phase = np.array(gra_phase)
    return grid_rate, grid_phase, gra_rate, gra_phase

# grid_rate, grid_phase, gra_rate, gra_phase = all_phases()





def collector(path = '/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/71-70-65-60'):
    dur_ms = 2000
    npzfiles = []
    grid_rate_code =[]
    grid_phase_code =[]
    grid_complex_code =[]
    grid_th_cross = []
    gra_rate_code = []
    gra_phase_code= []
    gra_complex_code= []
    gra_th_cross= []
    for file in sorted(glob.glob(os.path.join(path,'*.npz'))):
        npzfiles.append(file)
        load = np.load(file, allow_pickle=True)
        grid_rate_code.append(load['grid_rate_code'])
        grid_phase_code.append(load['grid_phase_code'])
        grid_complex_code.append(load['grid_complex_code'])
        gra_rate_code.append(load['gra_rate_code'])
        gra_phase_code.append(load['gra_phase_code'])
        gra_complex_code.append(load['gra_complex_code'])
        
    sim_traj = load['sim_traj']
    diff_traj = load['diff_traj']
    np.savez(os.path.join(path,'rate_n_phase_traj_'+str(sim_traj[0])+'-'+str(sim_traj[1])+'-'+str(diff_traj[0])+'-'+str(diff_traj[1])+'_net-seeds_410-419_'+str(dur_ms)+'ms'), 
             grid_rate_code = grid_rate_code,
             grid_phase_code = grid_phase_code,
             grid_complex_code = grid_complex_code,
             gra_rate_code = gra_rate_code,
             gra_phase_code = gra_phase_code,
             gra_complex_code = gra_complex_code)
# collector()


