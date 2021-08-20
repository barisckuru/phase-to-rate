#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 11:11:11 2021

@author: baris
"""

from neuron import h, gui  # gui necessary for some parameters to h namespace
import net_tunedrev




def granule_simulate(grid_spikes, grid_seed, poiss_seed, tune, dur_ms, trajs, pp_weight=9e-4):
    n_poiss = len(grid_spikes)
    n_traj = grid_spikes[0][1]
    for i in range(n_poiss):
        curr_grid_spikes = grid_spikes[i]
        curr_gra_spikes = pyDentate(curr_grid_spikes, trajs, grid_seed, poiss_seed, n_traj, dur_ms, pp_weight, tune, shuffle)[0]
        gra_spikes[idx] = copy.deepcopy(curr_gra_spikes)
    for i in :

        
def load_dll():
# Where to search for nrnmech.dll file. Must be adjusted for your machine. For pyDentate
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
                  ("/home/baris/grid_cell/mechs_7-6_linux/"
                    "x86_64/.libs/libnrnmech.so")]
    for x in dll_files:
        if os.path.isfile(x):
            dll_dir = x
    print("DLL loaded from: " + dll_dir)
    return dll_dir
        
def pyDentate(input_grid_out, trajs, grid_seed, poiss_seed, n_traj, dur_ms, pp_weight, tune, shuffle):
    savedir = os.getcwd()
    input_scale = 1000
    dent_seed = grid_seed+150 #dent_seed for network generation & simulation
    #number of cells
    n_grid = 200 
    n_granule = 2000
    n_mossy = 60
    n_basket = 24
    n_hipp = 24
    np.random.seed(dent_seed) # dent_seed for connections in the network
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
    np.random.seed(dent_seed) #dent_seed again for network generation & simulation
    granule_output = np.zeros((n_granule, n_traj), dtype = np.ndarray)
    mossy_output = np.zeros((n_mossy, n_traj), dtype = np.ndarray)
    basket_output = np.zeros((n_basket, n_traj), dtype = np.ndarray)
    hipp_output = np.zeros((n_hipp, n_traj), dtype = np.ndarray)  
    for trj in range(n_traj):
        nw = net_tunedrev.TunedNetwork(dent_seed, input_grid_out[:,trj], PP_to_GCs, PP_to_BCs, tune, pp_weight=pp_weight)
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
        granule_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[0].ap_counters], dtype=object))
        # mossy_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[1].ap_counters], dtype=object))
        # basket_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[2].ap_counters], dtype=object))
        # hipp_output[:,trj] =  copy.deepcopy(np.array([cell[0].as_numpy() for cell in nw.populations[3].ap_counters], dtype=object))
        
        tune_n_shuffle = tune + '_' + str(shuffle)
        fig = nw.plot_aps(time=dur_ms)
        tuned_fig_file_name = (str(nw) + "spike_plot_rate_"+tune_n_shuffle+"_n_phase_gra_out_for_perceptron_seed3_dur_weight_"+
                    str(dent_seed)+ '_' + str(dur_ms) +'_' +str(pp_weight)+'_'+ str(trj))
        nw.save_ap_fig(fig, savedir, tuned_fig_file_name)
    path = '/home/baris/results/pyDentate/'
    fname = 'pyDentate_'+tune_n_shuffle+'_out_traj_'+str(trajs[0])+'-'+str(trajs[1])+'_dur_'+str(dur_ms)+'_weight_'+str(pp_weight)+'_seed1_'+str(grid_seed)+'_seed2_'+str(poiss_seed)
    np.savez(path+fname, 
    granule_output = granule_output,
    mossy_output = mossy_output,
    basket_output = basket_output,
    hipp_output = hipp_output, allow_pickle=True)
    
    # tuned_save_file_name = (str(nw) + "data_rate_n_phase_gra_out_for_perceptron_seed3_dur_" +
    #                 '-'+str(dent_seed)+ '_' +
    #                 str(dur_ms))
    # nw.shelve_network(savedir, tuned_save_file_name)
    
    return granule_output, mossy_output, basket_output, hipp_output