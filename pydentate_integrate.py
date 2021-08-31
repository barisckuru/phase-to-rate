#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 10:20:34 2021

@author: baris
"""
from pydentate import net_tunedrev, neuron_tools
import numpy as np
import scipy.stats as stats

def granule_simulate(grid_spikes, poisson_seeds, network_type, grid_seed, pp_weight, input_scale=1000):
    granule_spikes = {}
    granule_spike_list = []
    for poisson_seed in poisson_seeds:
        for curr_grid_spikes in grid_spikes[poisson_seed]:
            np.random.seed(grid_seed)
    
            # Randomly choose target cells for the PP lines
            gauss_gc = stats.norm(loc=1000, scale=input_scale)
            gauss_bc = stats.norm(loc=12, scale=(input_scale/2000.0)*24)
            pdf_gc = gauss_gc.pdf(np.arange(2000))
            pdf_gc = pdf_gc/pdf_gc.sum()
            pdf_bc = gauss_bc.pdf(np.arange(24))
            pdf_bc = pdf_bc/pdf_bc.sum()
            GC_indices = np.arange(2000)
            start_idc = np.random.randint(0, 1999, size=400)
            
            PP_to_GCs = []
            for x in start_idc:
                curr_idc = np.concatenate((GC_indices[x:2000], GC_indices[0:x]))
                PP_to_GCs.append(np.random.choice(curr_idc, size=100, replace=False,
                                                  p=pdf_gc))
            
            PP_to_GCs = np.array(PP_to_GCs)
            PP_to_GCs = PP_to_GCs[0:24]
            
            BC_indices = np.arange(24)
            start_idc = np.array(((start_idc/2000.0)*24), dtype=int)
            
            PP_to_BCs = []
            for x in start_idc:
                curr_idc = np.concatenate((BC_indices[x:24], BC_indices[0:x]))
                PP_to_BCs.append(np.random.choice(curr_idc, size=1, replace=False,
                                                  p=pdf_bc))
            
            PP_to_BCs = np.array(PP_to_BCs)
            PP_to_BCs = PP_to_BCs[0:24]
            
            nw = net_tunedrev.TunedNetwork(None, curr_grid_spikes,
                              PP_to_GCs,
                              PP_to_BCs,
                              pp_weight=pp_weight)
            
            # Handle the different cases of inhibition
            if network_type == 'no-feedback':
                # Set GC to MC weight to 0
                for syn in nw.populations[0].connections[24].netcons: syn[0].weight[0] = 0.0
                # Set GC to BC weight to 0
                for syn in nw.populations[0].connections[25].synapses: syn[0].weight[0] = 0.0
            elif network_type == 'no-feedforward':
                # Set PP to BC weight to 0
                for pp_conns in nw.populations[2].connections[0:24]:
                    for syn in pp_conns.netcons: syn.weight[0] = 0.0
            elif network_type == 'disinhibited':
                # Set GC to MC weight to 0
                for syn in nw.populations[0].connections[24].netcons: syn[0].weight[0] = 0.0
                # Set GC to BC weight to 0
                for syn in nw.populations[0].connections[25].synapses: syn[0].weight[0] = 0.0
                # Set PP to BC weight to 0
                for pp_conns in nw.populations[2].connections[0:24]:
                    for syn in pp_conns.netcons: syn.weight[0] = 0.0
            elif network_type != 'tuned':
                raise ValueError("network_type must be 'tuned', 'no-feedback', 'no-feedforward' or 'disinhibited'")
        
            neuron_tools.run_neuron_simulator()
            granule_spike_list.append(nw.populations[0].get_timestamps())
        granule_spikes[poisson_seed] = granule_spike_list
    return granule_spikes
