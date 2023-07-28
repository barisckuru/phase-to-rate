from neural_coding import load_spikes, rate_n_phase
#from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import shelve
import matplotlib as mpl
from scipy.stats import pearsonr
import scipy.signal as signal
import scipy.stats
import copy
import glob
import pdb

trajectories = [75]

'''
Code to run positional information analysis according to Tingley & Buzsaki, 2018
'''

# =============================================================================
# return spike rate vectors for pos. Info analysis
# =============================================================================

def _spike_counter(spike_times_sample, bin_size_ms=100, dur_ms=2000):
    
    ''' 
    returns counts=[cell,bin] array with the number of spikes in each bin
    requires spike_times_sample=list (cell) of lists (spike time values in ms)
    '''
    
    n_bins = int(dur_ms / bin_size_ms)
    n_cells = len(spike_times_sample)
    counts = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, value in enumerate(spike_times_sample):
            curr_ct = (
                (bin_size_ms * (i) < value) & (value < bin_size_ms * (i + 1))
            ).sum()
            counts[idx, i] = curr_ct
    return counts

# =============================================================================
# return phase vectors for pos. Info analysis
# =============================================================================

def _phase_definer(spike_times_sample, nan_fill=False, bin_size_ms=100, dur_ms=2000):
    
    ''' 
    returns mean_phases=[cell,bin] array with mean phase in each bin
    requires spike_times=list (cell) of lists (spike time values in ms)
    '''
    
    n_bins = int(dur_ms / bin_size_ms)
    n_cells = len(spike_times_sample)
    phases = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, val in enumerate(spike_times_sample):
            curr_train = val[
                ((bin_size_ms * (i) < val) & (val < bin_size_ms * (i + 1)))
            ]
            if curr_train.size != 0:
                phases[idx, i] = scipy.stats.circmean(
                    curr_train % (bin_size_ms) / (bin_size_ms) * 2 * np.pi,
                    nan_policy='omit'
                )
    if nan_fill is True:
        mean_phases = np.mean(phases[phases != 0])
        phases[phases == 0] = mean_phases
    return phases

# =============================================================================
# convert spike trains into rate and phase vectors for pos. Info analysis
# =============================================================================

def _phase_n_rate_Tingley(spike_times,n_samples=20,bin_size_ms=100,dur_ms=2000):
    
    ''' 
    returns binwise AP count or phase in the form [cell,bin,Poisson_seed]
    requires spike_times=list(Poisson seed) of lists(cell) 
    '''
    
    n_bins = int(dur_ms / bin_size_ms)
    n_cell = len(spike_times[0])
    counts = np.empty((n_cell, n_bins, n_samples))
    phases = np.empty((n_cell, n_bins, n_samples))
    rate_code = np.empty((n_cell, n_bins, n_samples))
    phase_code = np.empty((n_cell, n_bins, n_samples))
    
    for sample_idx in range(n_samples):
        spike_times_sample = spike_times[sample_idx]
        single_count = _spike_counter(
            spike_times_sample, bin_size_ms=bin_size_ms, dur_ms=dur_ms
        )
        single_phase = _phase_definer(
            spike_times_sample, bin_size_ms=bin_size_ms, dur_ms=dur_ms
        )
        counts[:, :, sample_idx] = single_count
        phases[:, :, sample_idx] = single_phase
    return counts, phases

# =============================================================================
# filter insufficiently active cells
# =============================================================================

def _filter_cells(counts,phases, threshold=8):
    
    ''' counts,phases are rate and phase code arrays, respectively, 
    each of the form: [cell,bin,Poisson_seed]
    smoothing is the number of bins to smooth for boxcar
    
    returns counts and phases with cells that fire total spikes below threshold deleted'''
    
    cells = np.size(counts,axis=0)
    for cell in range(cells):
        cells_to_delete = []
        if counts[cell,:,:].sum() < threshold:
            cells_to_delete.append(cell)
    counts = np.delete(counts, cells_to_delete, axis=0)
    phases = np.delete(phases, cells_to_delete, axis=0)

    return counts, phases

# =============================================================================
# Boxcar filter over timebins
# =============================================================================

def _smooth_cells(counts,phases,smoothing):
    
    ''' counts,phases are rate and phase code arrays, respectively, 
    each of the form: [cell,bin,Poisson_seed]
    smoothing is the number of bins to smooth for boxcar'''
    
    if smoothing ==1:
        return counts,phases
    else: 
        temp_count = np.copy(counts)
        temp_phase = np.copy(phases)
        for i in range(np.size(counts,axis=1)-smoothing+1):
            count_box = counts[:,i:(i+smoothing),:]
            #print(np.nanmean(count_box,axis=1))
            temp_count[:,i,:]=np.nanmean(count_box,axis=1)
            #print(temp_count)
            
            phases_box = phases[:,i:(i+smoothing),:]
            temp_phase[:,i,:] = scipy.stats.circmean(phases_box,axis=1,nan_policy='omit')
        counts = temp_count[:,:-smoothing+1,:]
        phases = temp_phase[:,:-smoothing+1,:]
        return counts,phases

# =============================================================================
# Positional Information analysis
# =============================================================================
   
def pos_information(spike_times, discretization=7,
                    bin_size=100, smoothing=1):
    counts, phases = _phase_n_rate_Tingley(spike_times,n_samples=20,bin_size_ms=bin_size,dur_ms=2000)
    
    ''' counts,phases are rate and phase code arrays, respectively, 
    each of the form: [cell,bin,Poisson_seed]
    
    returns pos info for each [cell,pos] for both rate and phase codes
    '''
    counts, phases = _filter_cells(counts,phases)
    counts, phases = _smooth_cells(counts,phases,smoothing)

    n_cell =  np.size(counts,axis=0)
    n_pos =  np.size(counts,axis=1)
    Poisson_seeds = np.size(counts,axis=2)
    rate_results = np.zeros((n_cell,n_pos))
    phase_results = np.zeros((n_cell,n_pos))
    
    for cell in range(n_cell):
        #print(cell)
        max_rate = 0
        ''' e.g. pos_rate_local denotes the probability of occurence of a certain
        spike count at a specific position, i.e. the frequency of that spike count over Poisson seeds'''
        pos_rate_local = np.zeros(discretization)
        pos_rate_total = np.zeros(discretization)
        pos_phase_local = np.zeros(discretization)
        pos_phase_total = np.zeros(discretization)
        
        # assess max rate for data discretization
        for pos in range(n_pos):
            local_rates = counts[cell,pos,:]
            if max(local_rates)>max_rate:
                max_rate = max(local_rates)
        max_phase = 2*np.pi
        
        # assess mean occurences over all position bins of specific values (P_k)
        for pos in range(n_pos):
            local_rates = counts[cell,pos,:]
            local_phases = phases[cell,pos,:]
            #print(local_rates)
            for k in range(discretization):
                # rate
                pos_rate_local[k] = ((local_rates >= k*max_rate/discretization) & (local_rates <= (k+1)*max_rate/discretization)).sum()
                pos_phase_local[k] = ((local_phases >= k*max_phase/discretization) & (local_phases <= (k+1)*max_phase/discretization)).sum()
                pos_rate_local[k] = pos_rate_local[k]/Poisson_seeds
                pos_phase_local[k] = pos_phase_local[k]/Poisson_seeds
            pos_rate_total = np.add(pos_rate_total,pos_rate_local)
            pos_phase_total = np.add(pos_phase_total,pos_phase_local)
            #print(pos_rate_local)
        pos_rate_mean = pos_rate_total/n_pos
        pos_phase_mean = pos_phase_total/n_pos
        #print(pos_rate_mean)
                       
        # assess positional information
        for pos in range(n_pos):
            pos_rate_info = 0
            pos_phase_info = 0
            local_rates = counts[cell,pos,:]
            local_phases = phases[cell,pos,:]
            for k in range(discretization):
                # rate
                pos_rate_local[k] = ((local_rates >= k*max_rate/discretization) & (local_rates <= (k+1)*max_rate/discretization)).sum()
                pos_phase_local[k] = ((local_phases >= k*max_phase/discretization) & (local_phases <= (k+1)*max_phase/discretization)).sum()
                pos_rate_local[k] = pos_rate_local[k]/Poisson_seeds
                pos_phase_local[k] = pos_phase_local[k]/Poisson_seeds
                P_k_pos_rate = pos_rate_local[k]
                P_k_rate = pos_rate_mean[k]
                P_k_pos_phase = pos_phase_local[k]
                P_k_phase = pos_phase_mean[k]
                #print('local p(rate k): {}'.format(P_k_pos))
                #print('global p(rate k): {}'.format(P_k))
                if P_k_pos_rate > 0:
                    pos_rate_info += P_k_pos_rate * np.log(P_k_pos_rate/P_k_rate)
                    pos_phase_info +=  P_k_pos_phase * np.log(P_k_pos_phase/P_k_phase)
            rate_results[cell,pos]=pos_rate_info
            phase_results[cell,pos]=pos_phase_info
            
        #pos. info per bin
        rate_mean = np.nanmean(rate_results)
        phase_mean = np.nanmean(phase_results)
        
        # pos. info per spike
        rate_mean = rate_mean/ np.mean(counts)
        phase_mean = phase_mean/ np.mean(counts)
        
               
    return rate_mean,phase_mean


n_samples = 20
grid_seeds = np.arange(1,11,1)
grid_seeds_idx = range(0,10)
tunes = ['full','no-feedforward', 'no-feedback', 'disinhibited']


# =============================================================================
# reshape data to list(Poisson seed) of lists(cell)
# =============================================================================

def reshape_for_pos_info(all_spikes, shuffling, cell_type):
    grid_seeds = range(1,11)
    poisson_seeds = range(0,20)
    if cell_type == 'grid':
        n_cell = 200
    elif cell_type == 'granule':
        n_cell = 2000
    spike_times = {}
    for grid in grid_seeds:
        spike_times[grid]=[]
        for poiss in poisson_seeds:
            spike_times[grid].append([])
            for c in range(n_cell):
                spike_times[grid][poiss].append(all_spikes[grid][shuffling][cell_type][75][poiss][c])
    return spike_times


# =============================================================================
# load data and run analysis
# =============================================================================

# =============================================================================
# run over smoothing windows
# =============================================================================

bin_size=100
for tuning in tunes:
    print(tuning)
    all_spikes ={}
    for smoothing in range(1,20):
        print('smoothing: {}'.format(smoothing))
        for grid_seed in grid_seeds:
            path = "C:/Phase2RateDataFull/data/data/main/{}/collective/grid-seed_duration_shuffling_tuning_".format(tuning)
            # non-shuffled
            ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
            grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
            granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
            
            
            # shuffled
            s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
            s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
            s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
            
            #print('shuffled path ok')
        
            all_spikes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
            all_spikes[grid_seed]["shuffled"] = {"grid": s_grid_spikes, "granule": s_granule_spikes}
            all_spikes[grid_seed]["non-shuffled"] = {"grid": grid_spikes, "granule": granule_spikes}
        
        all_ns_grid = reshape_for_pos_info(all_spikes, 'non-shuffled', 'grid')
        all_s_grid = reshape_for_pos_info(all_spikes, 'shuffled', 'grid')
        all_ns_granule = reshape_for_pos_info(all_spikes, 'non-shuffled', 'granule')
        all_s_granule = reshape_for_pos_info(all_spikes, 'shuffled', 'granule')
        
        ns_grid_pos_info = []
        s_grid_pos_info = []
        ns_granule_pos_info = []
        s_granule_pos_info = []
        
        for grid in grid_seeds:
            ns_grid = all_ns_grid[grid]
            s_grid = all_s_grid[grid]
            ns_granule = all_ns_granule[grid]
            s_granule = all_s_granule[grid]
            
            ns_grid_pos_info.append(pos_information(ns_grid,bin_size=bin_size,smoothing=smoothing))
            s_grid_pos_info.append(pos_information(s_grid,bin_size=bin_size,smoothing=smoothing))
            ns_granule_pos_info.append(pos_information(ns_granule,bin_size=bin_size,smoothing=smoothing))
            s_granule_pos_info.append(pos_information(s_granule,bin_size=bin_size,smoothing=smoothing))
            print(f'grid seed {grid}')
           
        all_pos_info = np.concatenate((ns_grid_pos_info, s_grid_pos_info,
                                    ns_granule_pos_info, s_granule_pos_info))
        
        smooth_str = 40*[str(smoothing)]
        cell = 20*['grid']+20*['granule']
        shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
        
        # rate 
        all_pos_info_rate = all_pos_info[:,0]
        #cell = 20*[str(smoothing) +' grid']+20*[str(smoothing) + ' granule']

        pos_info_rate_all = np.stack((all_pos_info_rate, smooth_str, cell, shuffling), axis=1)
    
        if smoothing == 1:
            pos_info_rate = pos_info_rate_all
        else:
            pos_info_rate = np.concatenate((pos_info_rate, pos_info_rate_all[:, :]), axis=0)
        
        # phase
        all_pos_info_phase = all_pos_info[:,1]
        #shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
        pos_info_phase_all = np.stack((all_pos_info_phase, smooth_str, cell, shuffling), axis=1)
    
        if smoothing == 1:
            pos_info_phase = pos_info_phase_all
        else:
            pos_info_phase = np.concatenate((pos_info_phase, pos_info_phase_all[:, :]), axis=0)
    
    phase_bin_pi = 2
    threshold=8
    spatial_bin = 2
    
    # save rate data
    df_pos_info_rate = pd.DataFrame(pos_info_rate, columns=['info', 'smoothing','cell', 'shuffling'])
    df_pos_info_rate['info'] = df_pos_info_rate['info'].astype('float')
    plt.close('all')
    sns.barplot(data=df_pos_info_rate, x='smoothing', y='info', hue='shuffling', 
                errorbar='sd', capsize=0.2, errwidth=(2))
    plt.title(f'positional rate Information - Average of Population'
              +f'\n cells firing less than {threshold} spikes are filtered out'
              +f'\n 10 grid seeds, 20 poisson seeds aggregated,\n'
              +f'spatial bin = {spatial_bin} cm')
     
    #df_pos_info_rate.to_pickle('pos_info_rate_non-adjusted.pkl')
    #df_pos_info_rate.to_csv('pos_info_rate_non-adjusted.csv')
    df_pos_info_rate.to_excel('pos_info_rate_non-adjusted_smoothing_{}.xlsx'.format(tuning))
    
    # save phase data
    df_pos_info_phase = pd.DataFrame(pos_info_phase, columns=['info', 'smoothing','cell', 'shuffling'])
    df_pos_info_phase['info'] = df_pos_info_phase['info'].astype('float')
    plt.close('all')
    sns.barplot(data=df_pos_info_phase, x='smoothing', y='info', hue='shuffling', 
                errorbar='sd', capsize=0.2, errwidth=(2))
    plt.title(f'positional phase Information - Average of Population'
              +f'\n cells firing less than {threshold} spikes are filtered out'
              +f'\n 10 grid seeds, 20 poisson seeds aggregated,\n'
              +f'spatial bin = {spatial_bin} cm')
     
    #df_pos_info_phase.to_pickle('pos_info_phase_non-adjusted.pkl')
    #df_pos_info_phase.to_csv('pos_info_phase_non-adjusted.csv')
    df_pos_info_phase.to_excel('pos_info_phase_non-adjusted_smoothing_{}.xlsx'.format(tuning))

# =============================================================================        
# run over network conditions
# =============================================================================

smoothing = 3       # number of smoothing bins
print(smoothing)
for tuning in tunes:
    print(tuning)
    all_spikes = {}
    for grid_seed in grid_seeds:
        #path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
        path = "C:/Phase2RateDataFull/data/data/main/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
        
        # non-shuffled
        ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
        grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
        granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
        
        
        # shuffled
        s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
        s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
        s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)       
        #print('shuffled path ok')
    
        all_spikes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
        all_spikes[grid_seed]["shuffled"] = {"grid": s_grid_spikes, "granule": s_granule_spikes}
        all_spikes[grid_seed]["non-shuffled"] = {"grid": grid_spikes, "granule": granule_spikes}

    all_ns_grid = reshape_for_pos_info(all_spikes, 'non-shuffled', 'grid')
    all_s_grid = reshape_for_pos_info(all_spikes, 'shuffled', 'grid')
    all_ns_granule = reshape_for_pos_info(all_spikes, 'non-shuffled', 'granule')
    all_s_granule = reshape_for_pos_info(all_spikes, 'shuffled', 'granule')
    
    ns_grid_pos_info = []
    s_grid_pos_info = []
    ns_granule_pos_info = []
    s_granule_pos_info = []
    
    for grid in grid_seeds:
        ns_grid = all_ns_grid[grid]
        s_grid = all_s_grid[grid]
        ns_granule = all_ns_granule[grid]
        s_granule = all_s_granule[grid]
        
        ns_grid_pos_info.append(pos_information(ns_grid,smoothing=smoothing))
        s_grid_pos_info.append(pos_information(s_grid,smoothing=smoothing))
        ns_granule_pos_info.append(pos_information(ns_granule,smoothing=smoothing))
        s_granule_pos_info.append(pos_information(s_granule,smoothing=smoothing))
        print(f'grid seed {grid}')
       
    all_pos_info = np.concatenate((ns_grid_pos_info, s_grid_pos_info,
                                ns_granule_pos_info, s_granule_pos_info))
    # rate 
    all_pos_info_rate = all_pos_info[:,0]
    cell = 20*[tuning +' grid']+20*[tuning + ' granule']
    shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
    gridseed = 4*[g for g in range(1,11)]
    pos_info_rate_all = np.stack((all_pos_info_rate, cell, shuffling, gridseed), axis=1)

    if tuning == 'full':
        pos_info_rate = pos_info_rate_all
    else:
        pos_info_rate = np.concatenate((pos_info_rate, pos_info_rate_all[20:, :]), axis=0)
    
    # phase
    all_pos_info_phase = all_pos_info[:,1]
    cell = 20*[tuning +' grid']+20*[tuning + ' granule']
    shuffling = 2*(10*['non-shuffled']+10*['shuffled'])
    gridseed = 4*[g for g in range(1,11)]
    pos_info_phase_all = np.stack((all_pos_info_phase, cell, shuffling, gridseed), axis=1)

    if tuning == 'full':
        pos_info_phase = pos_info_phase_all
    else:
        pos_info_phase = np.concatenate((pos_info_phase, pos_info_phase_all[20:, :]), axis=0)

phase_bin_pi = 2
threshold=8
spatial_bin = 2

# save rate data
df_pos_info_rate = pd.DataFrame(pos_info_rate, columns=['info', 'cell', 'shuffling','gridseed'])
df_pos_info_rate['info'] = df_pos_info_rate['info'].astype('float')
plt.close('all')
sns.barplot(data=df_pos_info_rate, x='cell', y='info', hue='shuffling', 
            errorbar='sd', capsize=0.2, errwidth=(2))
plt.title(f'positional rate Information - Average of Population'
          +f'\n cells firing less than {threshold} spikes are filtered out'
          +f'\n 10 grid seeds, 20 poisson seeds aggregated,\n'
          +f'spatial bin = {spatial_bin} cm')


#df_pos_info_rate.to_pickle('pos_info_rate_non-adjusted.pkl')
#df_pos_info_rate.to_csv('pos_info_rate_non-adjusted.csv')
df_pos_info_rate.to_excel('pos_info_rate_non-adjusted.xlsx')
#isolated effects

# save phase data 
df_pos_info_phase = pd.DataFrame(pos_info_phase, columns=['info', 'cell', 'shuffling','gridseed'])
df_pos_info_phase['info'] = df_pos_info_phase['info'].astype('float')
plt.close('all')
sns.barplot(data=df_pos_info_phase, x='cell', y='info', hue='shuffling', 
            errorbar='sd', capsize=0.2, errwidth=(2))
plt.title(f'positional rate Information - Average of Population'
          +f'\n cells firing less than {threshold} spikes are filtered out'
          +f'\n 10 grid seeds, 20 poisson seeds aggregated,\n'
          +f'spatial bin = {spatial_bin} cm')


#df_pos_info_phase.to_pickle('pos_info_phase_non-adjusted.pkl')
#df_pos_info_phase.to_csv('pos_info_phase_non-adjusted.csv')
df_pos_info_phase.to_excel('pos_info_phase_non-adjusted.xlsx')
#isolated effects

