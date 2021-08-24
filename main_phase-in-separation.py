#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:04:25 2021

@author: baris
"""


from grid_model import grid_simulate
import dentate_simulate
import neural_coding
import numpy as np



# shuffle = True
shuffle = False


#Seeds
grid_seeds = 520

# poiss_seeds = np.arange(360,365,1)
# poiss_seeds = np.arange(380,385,1)
# poiss_seeds = np.arange(400,405,1)
poiss_seeds = np.arange(420,425,1)
poiss_seeds_2 = poiss_seeds + 10
perc_seeds = grid_seeds-100


tune = 'full'
if shuffle==True:
    tune = 'shuffled_'+tune
else:
    tune = 'nonshuffled_'+tune
    

n_poiss = poiss_seeds.shape[0]
n_network = 1 #perc_seeds.shape[0]


#Intialize zeros arrays&lists to fill with data
sample_size = 2*poiss_seeds.shape[0]
n_sampleset = 1 #perc_seeds.shape[0]

trajs = np.array([75, 74.5, 74, 70])
n_traj = trajs.shape[0]

trajs = 75

'SIMULATION'

grid_spikes = grid_simulate(trajs, 2000, 400, poiss_seeds, False)

gra_spikes = dentate_simulate.granule_simulate(grid_spikes, grid_seed, poiss_seed, tune, dur_ms, trajs, pp_weight=9e-4)
grid_phases, gra_phases = neural_coding.spikes_n_phases(grid_spikes, gra_spikes, dur_ms, grid_seed, poiss_seeds, pp_weight, tune, shuffle)
    

'Updated spike counter'
#grid_spike_cts
#gra_spike_cts


grid_rate_code, grid_phase_code, grid_polar_code = neural_coding.code_maker(grid_spike_cts, grid_phases)
gra_rate_code, gra_phase_code, gra_polar_code = neural_coding.code_maker(gra_spike_cts, gra_phases)