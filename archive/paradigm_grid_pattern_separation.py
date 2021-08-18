#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 18:13:28 2020

@author: bariskuru & DanielM
"""

from neuron import h, gui  # gui necessary for some parameters to h namespace
import numpy as np
import net_tunedrev
import os
import argparse
import scipy.stats as stats
from grid_poiss_input_gen import inhom_poiss
import time

start = time.time()
# Handle command line inputs
pr = argparse.ArgumentParser(description='Local pattern separation paradigm')
pr.add_argument('-savedir',
                type=str,
                help='complete directory where data is saved',
                default=os.getcwd(),
                dest='savedir')
pr.add_argument('-scale',
                type=int,
                help='standard deviation of gaussian distribution',
                default=1000,
                dest='input_scale')
pr.add_argument('-pp_weight',
                type=float,
                help='standard deviation of gaussian distribution',
                default=1e-3,
                dest='pp_weight')


args = pr.parse_args()
savedir = args.savedir
input_scale = args.input_scale

# Where to search for nrnmech.dll file. Must be adjusted for your machine.
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



seed_1 = 102 #seed_1 for granule cell generation
seed_2 = 202 #seed_2 inh poisson input generation
seed_3 = seed_1+50 #seed_3 for network generation & simulation
seeds=np.array([seed_1, seed_2, seed_3])

#number of cells
n_grid = 200 
n_granule = 2000
n_mossy = 60
n_basket = 24
n_hipp = 24

# parameters for gc grid generator
arr_size = 200 # arr_size is the side length of the square field as array length
field_size_cm = 100 #size of the field in real life 
speed_cm = 20 # speed of the virtual mouse
dur_ms = (field_size_cm/speed_cm)*1000
dur_ms = 5000
n_traj = 8 # n_traj trajectories are produced for parallel and sloping individually

np.random.seed(seed_1) #seed_1 for granule cell generation
#execute the script for grid generation and obtain the rate profiles of each grid cells for different trajectories
stream = open("gridcell_traj_activ_pf.py") #spacings, orientations, phases, trajectory indices are defined in this file
# as a result all grids, rate profiles for each trajectory are generated in this file
read_file = stream.read()
exec(read_file)


np.random.seed(seed_3) # seed_3 for connections in the network

# Randomly choose target cells for the GridCell lines
gauss_gc = stats.norm(loc=1000, scale=input_scale)
gauss_bc = stats.norm(loc=12, scale=(input_scale/float(n_granule))*n_basket)
pdf_gc = gauss_gc.pdf(np.arange(n_granule))
pdf_gc = pdf_gc/pdf_gc.sum()
pdf_bc = gauss_bc.pdf(np.arange(n_basket))
pdf_bc = pdf_bc/pdf_bc.sum()
GC_indices = np.arange(n_granule)
start_idc = np.random.randint(0, n_granule-1, size=n_grid)

PP_to_GCs = []
for x in start_idc:
    curr_idc = np.concatenate((GC_indices[x:n_granule], GC_indices[0:x]))
    PP_to_GCs.append(np.random.choice(curr_idc, size=100, replace=False,
                                      p=pdf_gc))

PP_to_GCs = np.array(PP_to_GCs)

BC_indices = np.arange(n_basket)
start_idc = np.array(((start_idc/float(n_granule))*24), dtype=int)

PP_to_BCs = []
for x in start_idc:
    curr_idc = np.concatenate((BC_indices[x:24], BC_indices[0:x]))
    PP_to_BCs.append(np.random.choice(curr_idc, size=1, replace=False,
                                      p=pdf_bc))

PP_to_BCs = np.array(PP_to_BCs)


# generate temporal patterns out of grid cell act profiles as an input for pyDentate
input_grid_out = inhom_poiss(par_traj, n_traj, dt_s=0.0001, seed=seed_2)

np.random.seed(seed_3) #seed_3 again for network generation & simulation

granule_output = np.empty((n_granule, n_traj), dtype = np.ndarray)
mossy_output = np.empty((n_mossy, n_traj), dtype = np.ndarray)
basket_output = np.empty((n_basket, n_traj), dtype = np.ndarray)
hipp_output = np.empty((n_hipp, n_traj), dtype = np.ndarray)

for trj in range(n_traj):
    nw = net_tunedrev.TunedNetwork(seed_3, input_grid_out[:,trj], PP_to_GCs, PP_to_BCs)
    # Attach voltage recordings to all cells
    nw.populations[0].voltage_recording(range(n_granule))
    nw.populations[1].voltage_recording(range(n_mossy))
    nw.populations[2].voltage_recording(range(n_basket))
    nw.populations[3].voltage_recording(range(n_hipp))
    # Run the model
    
    """Initialization for -2000 to -100"""
    h.cvode.active(0)
    dt = 0.1
    h.steps_per_ms = 1.0/dt
    h.finitialize(-60)
    h.t = -2000
    h.secondorder = 0
    h.dt = 10
    while h.t < -100:
        h.fadvance()
        
    h.secondorder = 2
    h.t = 0
    h.dt = 0.1
    
    """Setup run control for -100 to 1500"""
    h.frecord_init()  # Necessary after changing t to restart the vectors
    while h.t < dur_ms:
        h.fadvance()
    print("Done Running")
    
    granule_output[:,trj] =  np.array([cell[0].as_numpy() for cell in nw.populations[0].ap_counters])
    mossy_output[:,trj] =  np.array([cell[0].as_numpy() for cell in nw.populations[1].ap_counters])
    basket_output[:,trj] =  np.array([cell[0].as_numpy() for cell in nw.populations[2].ap_counters])
    hipp_output[:,trj] =  np.array([cell[0].as_numpy() for cell in nw.populations[3].ap_counters])



    # tuned_save_file_name = (str(nw) + "_data_paradigm_grid-pattern" +
    #                         "-separation_par-traj_scale_seed1-seed2-seed3_ncells_dur_" +
    #                         str(par_idc_cm[trj]).zfill(3) + '_' +
    #                         str(input_scale).zfill(3) + '_' + 
    #                         str(seed_1)+'-'+str(seed_2)+'-'+str(seed_3)+'_'+
    #                         str(n_grid) + '_' +
    #                         str(dur_ms))
    # nw.shelve_network(savedir, tuned_save_file_name)

    # fig = nw.plot_aps(time=dur_ms)
    # tuned_fig_file_name = (str(nw) + "_spike-plot_paradigm_grid-pattern" +
    #                        "-separation_par-traj_scale_seed1-seed2-seed3_ncells_dur_" +
    #                        str(par_idc_cm[trj]).zfill(3) + '_' +
    #                        str(input_scale).zfill(3) + '_' + 
    #                        str(seed_1)+'-'+str(seed_2)+'-'+str(seed_3)+'_'+
    #                        str(n_grid) + '_' +
    #                        str(dur_ms))
    # nw.save_ap_fig(fig, savedir, tuned_fig_file_name)
file_name = ('grid_pattern_seperation_par-traj_seed1-seed2-seed3_n-grids_dur'+
            str(seed_1)+'-'+str(seed_2)+'-'+str(seed_3)+'_'+
            str(n_grid) + '_' +
            str(dur_ms))
np.savez(file_name, 
         input_grid_out=input_grid_out,
         granule_output=granule_output,
         mossy_output = mossy_output,
         basket_output=basket_output,
         hipp_output=hipp_output,
         seeds=seeds,
         par_traj_cm = par_idc_cm,
         n_grid=n_grid,
         dur_ms = dur_ms,
         input_scale=input_scale)
         
stop = time.time()
print(stop-start)
time_min = (stop-start)/60
time_hour = time_min/60
print(time_min)
print(time_hour)