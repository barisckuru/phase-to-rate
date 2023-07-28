from phase_to_rate.neural_coding import load_spikes_DMK, rate_n_phase, load_spikes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import sys
from copy import deepcopy

# parameters
phase_bin = 360
time_bin = 250
dur_ms = 2000
cycle_ms = 100

phase_bin_pi = phase_bin/180
spatial_bin = (time_bin/1000)*20

threshold = int((dur_ms/time_bin)*(360/phase_bin))

trajectories = [75]
n_samples = 20
grid_seeds = np.arange(1,11,1)
grid_seeds_idx = range(0,10)
# tunes = ['full', 'no-feedback']
tunes = ['full']

# =============================================================================
# skaggs info for rate-phase, mean of cells, mean of spatial bins, aggregated
# =============================================================================

def skaggs_information(spike_times, dur_ms, time_bin_size,
                        phase_bin_size=360, theta_bin_size=100):

    n_cell = len(spike_times)
    dur_s = int(dur_ms/1000)
    time_bin_s = time_bin_size/1000
    n_time_bins = int(dur_ms/time_bin_size)
    theta_bin_size_s = theta_bin_size/1000
    skaggs_all = np.zeros(n_cell)
    if phase_bin_size == 360:
        rates = np.zeros((n_cell, n_time_bins))
        for cell in range(n_cell):
            spikes = np.array(spike_times[cell])
            phases = [[] for _ in range(n_time_bins)]
            skaggs = np.zeros((n_time_bins))
            times = np.arange(0, dur_ms+time_bin_size, time_bin_size)
            for j, time in enumerate(times):
                if j == times.shape[0]-1:
                    break
                count = np.logical_and(spikes > time, spikes < times[j+1]).sum()
                rates[cell, j] = count/time_bin_s
            mean_rate = np.mean(rates[cell, :])
    
            for j, time in enumerate(times):
                if j == times.shape[0]-1:
                    break
                rate = rates[cell, j]
                info = (rate/mean_rate)*(np.log2(rate/mean_rate))
                if info == info: 
                    skaggs[j] = info
            skaggs_all[cell] = (1/(n_time_bins))*np.sum(skaggs)
        skaggs_info = np.mean(skaggs_all)
    else:
        n_phase_bins = int(360/phase_bin_size)
        rates = np.zeros((n_cell, n_phase_bins, n_time_bins))
        for cell in range(n_cell):
            spikes = np.array(spike_times[cell])
            phases = [[] for _ in range(n_time_bins)]
            skaggs = np.zeros((n_phase_bins, n_time_bins))
            times = np.arange(0, dur_ms+time_bin_size, time_bin_size)
            for j, time in enumerate(times):
                if j == times.shape[0]-1:
                    break
                curr_train = spikes[np.logical_and(spikes > time,
                                                  spikes < times[j+1])]
                if curr_train.size > 0:
                    phases[j] = list(curr_train % (theta_bin_size) / (theta_bin_size)*360)
            for i in range(n_phase_bins):
                for j, phases_in_time in enumerate(phases):
                    phases_in_time = np.array(phases_in_time)
                    count = ((phase_bin_size*(i) < phases_in_time) &
                                    (phases_in_time < phase_bin_size*(i+1))).sum()
                    rate = count*((1/theta_bin_size_s)*n_phase_bins)
                    rates[cell, i, j] = rate
            for j, phases_in_time in enumerate(phases):
                mean_rate = np.mean(rates[cell, :, j])
                for i in range(n_phase_bins):
                    rate = rates[cell, i, j]
                    info = (rate/mean_rate)*(np.log2(rate/mean_rate))
                    if info == info: 
                        skaggs[i, j] = info
            skaggs_all[cell] = (1/(n_phase_bins*n_time_bins))*np.sum(skaggs)
        # skaggs_info = np.sum(skaggs_all)
        skaggs_info = np.mean(skaggs_all)
    return skaggs_info

# =============================================================================
# aggraegate spikes from poisson seeds
# =============================================================================

def aggr (all_spikes, shuffling, cell):
    grid_seeds = range(1,11)
    poisson_seeds = range(0,20)
    agg_spikes = []
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for grid in grid_seeds:
        spikes = [[] for _ in range(n_cell)]
        for poiss in poisson_seeds:
            for c in range(n_cell):
                spikes[c]+= list(all_spikes[grid][shuffling][cell][75][poiss][c])
                spikes[c].sort()
        agg_spikes.append(spikes)
    return agg_spikes


# =============================================================================
# filter insufficient cells
# =============================================================================

def filter_inact_granule(agg_spikes, threshold):
    filtered_cells = []
    n_cell = len(agg_spikes[0])
    n_grid = len(agg_spikes)
    for grid in range(n_grid):
        cells = []
        for cell in range(n_cell):
            # print(len(agg_spikes[grid][cell]))
            if len(agg_spikes[grid][cell])>threshold:
                cells.append(agg_spikes[grid][cell])
        filtered_cells.append(cells)
    return filtered_cells


# =============================================================================
# load data
# =============================================================================
nonshuffled_granule_spikes = []
shuffled_granule_spikes = []

for tuning in tunes:
    all_spikes = {}
    for grid_seed in grid_seeds:

        path = r'C:\Users\Daniel\repos\phase-to-rate\data\noise\grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_pp-weight_noise-scale_'

        pp_strength = 0.0009
        if tuning == 'no-feedback':
            pp_strength = 0.0007
        else:
            pp_strength = 0.0009
        ns_path = (path + str(grid_seed) + "_[75]_100-119_2000_non-shuffled_"+str(tuning)+f"_{pp_strength}_0.25")
        grid_spikes = load_spikes_DMK(ns_path, "grid", trajectories, n_samples)
        granule_spikes = load_spikes_DMK(ns_path, "granule", trajectories, n_samples)

        # shuffled
        s_path = (path + str(grid_seed) + "_[75]_100-119_2000_shuffled_"+str(tuning)+f"_{pp_strength}_0.25")
        s_grid_spikes = load_spikes_DMK(s_path, "grid", trajectories, n_samples)
        s_granule_spikes = load_spikes_DMK(s_path, "granule", trajectories, n_samples)
        
        print('shuffled path ok')

        all_spikes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
        all_spikes[grid_seed]["shuffled"] = {"grid": s_grid_spikes, "granule": s_granule_spikes}
        all_spikes[grid_seed]["non-shuffled"] = {"grid": grid_spikes, "granule": granule_spikes}
        
        nonshuffled_granule_spikes.append(granule_spikes)
        shuffled_granule_spikes.append(s_granule_spikes)

    all_ns_grid = aggr(all_spikes, 'non-shuffled', 'grid')
    all_ns_grid = filter_inact_granule(all_ns_grid, threshold)
    all_s_grid = aggr(all_spikes, 'shuffled', 'grid')
    all_s_grid = filter_inact_granule(all_s_grid, threshold)
    all_ns_granule = aggr(all_spikes, 'non-shuffled', 'granule')
    all_ns_granule = filter_inact_granule(all_ns_granule, threshold)
    all_s_granule = aggr(all_spikes, 'shuffled', 'granule')
    all_s_granule = filter_inact_granule(all_s_granule, threshold)
    
    ns_grid_skaggs = []
    s_grid_skaggs = []
    ns_granule_skaggs = []
    s_granule_skaggs = []
    
    for grid in grid_seeds_idx:
        ns_grid = all_ns_grid[grid]
        s_grid = all_s_grid[grid]
        ns_granule = all_ns_granule[grid]
        s_granule = all_s_granule[grid]
        
        ns_grid_skaggs.append(skaggs_information(ns_grid, dur_ms, time_bin,
                                                 phase_bin_size=phase_bin))
        s_grid_skaggs.append(skaggs_information(s_grid, dur_ms, time_bin,
                                                phase_bin_size=phase_bin))
        ns_granule_skaggs.append(skaggs_information(
            ns_granule, dur_ms, time_bin, phase_bin_size=phase_bin))
        s_granule_skaggs.append(skaggs_information(
            s_granule, dur_ms, time_bin, phase_bin_size=phase_bin))
        print(f'grid seed {grid}')

    all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                                ns_granule_skaggs, s_granule_skaggs))
    cell = 20*[tuning +' grid']+20*[tuning + ' granule']
    shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
    all_skaggs = np.concatenate((ns_grid_skaggs, s_grid_skaggs,
                                ns_granule_skaggs, s_granule_skaggs))
    skaggs_info_all = np.stack((all_skaggs, cell, shuffling), axis=1)

    if tuning == 'full':
        skaggs = skaggs_info_all
    else:
        skaggs = np.concatenate((skaggs, skaggs_info_all[20:, :]), axis=0)

nonshuffled_granule_phases = deepcopy(nonshuffled_granule_spikes)
shuffled_granule_phases = deepcopy(shuffled_granule_spikes)

nonshuffled_mean_modulation = []
shuffled_mean_modulation = []

for idx_s, seed in enumerate(nonshuffled_granule_spikes):
    for idx, y in enumerate(nonshuffled_granule_spikes[idx_s][75]):
        for idx_z, y in enumerate(nonshuffled_granule_spikes[idx_s][75][idx]):
            for idx_q, q in enumerate(nonshuffled_granule_spikes[idx_s][75][idx][idx_z]):
                nonshuffled_granule_phases[idx_s][75][idx][idx_z][idx_q] = ((nonshuffled_granule_spikes[idx_s][75][idx][idx_z][idx_q] % cycle_ms) / cycle_ms) * 2 * np.pi

    curr_phases = np.array(nonshuffled_granule_phases[idx_s][75], dtype=object).reshape((2000, 20))
    curr_phases_flat = np.array([np.array([x for tr in traj for x in tr]) for traj in curr_phases], dtype=object)
    
    modulation = np.array([np.array([np.sqrt(np.cos(aps).sum()**2 + np.sin(aps).sum()**2)]) / len(aps) for aps in curr_phases_flat])
    
    nonshuffled_mean_modulation.append(modulation)
    

for idx_s, seed in enumerate(shuffled_granule_spikes):
    for idx, y in enumerate(shuffled_granule_spikes[idx_s][75]):
        for idx_z, y in enumerate(shuffled_granule_spikes[idx_s][75][idx]):
            for idx_q, q in enumerate(shuffled_granule_spikes[idx_s][75][idx][idx_z]):
                shuffled_granule_phases[idx_s][75][idx][idx_z][idx_q] = ((shuffled_granule_spikes[idx_s][75][idx][idx_z][idx_q] % cycle_ms) / cycle_ms) * 2 * np.pi

    curr_phases = np.array(shuffled_granule_phases[idx_s][75], dtype=object).reshape((2000, 20))
    curr_phases_flat = np.array([np.array([x for tr in traj for x in tr]) for traj in curr_phases], dtype=object)
    
    modulation = np.array([np.array([np.sqrt(np.cos(aps).sum()**2 + np.sin(aps).sum()**2)]) / len(aps) for aps in curr_phases_flat])
    
    shuffled_mean_modulation.append(modulation)

