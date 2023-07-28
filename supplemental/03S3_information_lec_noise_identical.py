from phase_to_rate.neural_coding import load_spikes_DMK, rate_n_phase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import shelve

# parameters
phase_bin = 360
time_bin = 250
dur_ms = 2000

phase_bin_pi = phase_bin/180
spatial_bin = (time_bin/1000)*20

threshold = int((dur_ms/time_bin)*(360/phase_bin))

trajectories = [75]
n_samples = 20
grid_seeds = np.arange(340,350,1)
grid_seeds_idx = range(0,10)
tunes = ['full', 'no-feedback']

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
    grid_seeds = range(340,350)
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
for tuning in tunes:
    all_spikes = {}
    for grid_seed in grid_seeds:
        # path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
        path = r'C:\Users\Daniel\repos\phase-to-rate\data\noise_lec_identical\old\grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_pp-weight_noise-scale_'

        pp_strength = 0.0009
        if tuning == 'no-feedback':
            pp_strength = 0.0007
        else:
            pp_strength = 0.0009
        ns_path = (path + str(grid_seed) + "_[75]_100-119_2000_non-shuffled_"+str(tuning)+f"_{pp_strength}_200")
        grid_spikes = load_spikes_DMK(ns_path, "grid", trajectories, n_samples)
        granule_spikes = load_spikes_DMK(ns_path, "granule", trajectories, n_samples)

        # shuffled
        s_path = (path + str(grid_seed) + "_[75]_100-119_2000_shuffled_"+str(tuning)+f"_{pp_strength}_200")
        s_grid_spikes = load_spikes_DMK(s_path, "grid", trajectories, n_samples)
        s_granule_spikes = load_spikes_DMK(s_path, "granule", trajectories, n_samples)
        
        print('shuffled path ok')
    
        all_spikes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
        all_spikes[grid_seed]["shuffled"] = {"grid": s_grid_spikes, "granule": s_granule_spikes}
        all_spikes[grid_seed]["non-shuffled"] = {"grid": grid_spikes, "granule": granule_spikes}

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

phase_bin_pi = phase_bin/180

if int(phase_bin_pi) == 2:
    phase_bin_pi = ''
else:
    phase_bin_pi = ', phase bin = ' + str(phase_bin_pi) + 'pi'

df_skaggs = pd.DataFrame(skaggs, columns=['info', 'cell', 'shuffling'])
df_skaggs['info'] = df_skaggs['info'].astype('float')

#isolated effects
full_ns = ((df_skaggs.loc[(df_skaggs['cell'] == 'full granule') & 
                                 (df_skaggs['shuffling'] == 'non-shuffled')]
            ['info']).reset_index(drop=True))
full_s = ((df_skaggs.loc[(df_skaggs['cell'] == 'full granule') & 
                                 (df_skaggs['shuffling'] == 'shuffled')]
            ['info']).reset_index(drop=True))

nofb_ns = ((df_skaggs.loc[(df_skaggs['cell'] == 'no-feedback granule') & 
                                 (df_skaggs['shuffling'] == 'non-shuffled')]
            ['info']).reset_index(drop=True))
nofb_s = ((df_skaggs.loc[(df_skaggs['cell'] == 'no-feedback granule') & 
                                 (df_skaggs['shuffling'] == 'shuffled')]
            ['info']).reset_index(drop=True))

info = pd.concat((full_ns/full_s, nofb_ns/nofb_s),
                 axis=0).reset_index()
info = info.rename(columns={'index': 'grid_seed'})
isolated = (10*['isolated full'] + 10*['isolated nofb'])
shuffling = 1*(10*['non-shuffled']+10*['shuffled'])
info['isolated'] = isolated
info['shuffling'] = shuffling

"""PLOTTING"""
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

f1, ([ax1, ax2], [ax3, ax4]) = plt.subplots(2, 2, figsize=(9, 9))
f1.subplots_adjust(hspace=.6, wspace=.6, left=0.1, right=0.9)
sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)
plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=8) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10) #fontsize of the 
plt.rcParams["svg.fonttype"] = "none"

my_pal = {'grid': '#716969', 'granule': '#09316c'}
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
grid_pal = {'non-shuffled': '#716969', 'shuffled': '#a09573'}

loc = np.array([0.4]*10)

sns.boxplot(data=df_skaggs, x='cell', y='info', hue='shuffling', order=['full grid'], zorder=1, palette=grid_pal, linewidth=0.5, fliersize=1)

fig, ax = plt.subplots(1)
sns.boxplot(data=df_skaggs, x='cell', y='info', hue='shuffling', order=['full granule', 'no-feedback granule'], zorder=1, palette=granule_pal, linewidth=0.5, fliersize=1, ax=ax)

ns_full = df_skaggs[(df_skaggs['shuffling'] == 'non-shuffled') & (df_skaggs['cell'] == 'full granule')]['info']
s_full = df_skaggs[(df_skaggs['shuffling'] == 'shuffled') & (df_skaggs['cell'] == 'full granule')]['info']

ns_nofb = df_skaggs[(df_skaggs['shuffling'] == 'non-shuffled') & (df_skaggs['cell'] == 'no-feedback granule')]['info']
s_nofb = df_skaggs[(df_skaggs['shuffling'] == 'shuffled') & (df_skaggs['cell'] == 'no-feedback granule')]['info']

ax.scatter([-loc/2, loc/2], 
             [ns_full, s_full], color='k')
ax.plot([-loc/2, loc/2], 
             [ns_full, s_full], color='k')

ax.scatter([-loc/2+1, loc/2+1], 
             [ns_nofb, s_nofb], color='k')
ax.plot([-loc/2+1, loc/2+1], 
             [ns_nofb, s_nofb], color='k')

color='#2c423f'

loc = np.array([0.]*10)

fig, ax = plt.subplots()
sns.boxplot(x='isolated', y='info', data=info, zorder=1, linewidth=0.5, fliersize=1, color='#0a9396')
isolated_full = info[(info['isolated'] == 'isolated full')]['info']
isolated_nofb = info[(info['isolated'] == 'isolated nofb')]['info']
ax.scatter([loc, loc+1], 
             [isolated_full, isolated_nofb], color='k')

"""Spike Histogram Plotting"""
syns_off = shelve.open(r'C:\Users\Daniel\repos\phase-to-rate\data\main\full\collective\full\grid-seed_duration_shuffling_tuning_1_2000_non-shuffled_full')
syns_100 = shelve.open(r'C:\Users\Daniel\repos\phase-to-rate\data\lec_spike_histogram_data_100_syns')
syns_150 = shelve.open(r'C:\Users\Daniel\repos\phase-to-rate\data\lec_spike_histogram_data_150_syns')
syns_200 = shelve.open(r'C:\Users\Daniel\repos\phase-to-rate\data\lec_spike_histogram_data_200_syns')

syns_100_flattened = {}
syns_100_flattened['lec_spikes'] = [((spike % 100) / 100) * 2 * np.pi for seed in syns_100['lec_spikes'] for cell in seed for spike in cell]
syns_100_flattened['granule_spikes'] = [((spike % 100) / 100) * 2 * np.pi for seed in syns_100['granule_spikes'][75] for cell in syns_100['granule_spikes'][75][seed] for spike in cell]
syns_100_flattened['grid_spikes'] = [((spike % 100) / 100) * 2 * np.pi for seed in syns_100['grid_spikes'][75] for cell in syns_100['grid_spikes'][75][seed] for spike in cell]

alpha = 0.6

my_pal = {'grid': '#1b9e77', 'granule': '#d95f02', 'lec': '#e7298a',}
plt.figure()
plt.hist(syns_100_flattened['grid_spikes'], bins=90, color=my_pal['grid'], alpha = alpha)
plt.hist(syns_100_flattened['lec_spikes'], bins=90, color=my_pal['lec'], alpha = alpha)
plt.xlim((0, 2*np.pi))
plt.xlabel("Phase (pi)")
plt.ylabel("AP Count")
plt.legend(("Grid Cells", "LEC Cells"))

gcs_flat_syns_off = [((spike % 100) / 100) * 2 * np.pi for seed in syns_off['75']['granule_spikes'] for cell in syns_off['75']['granule_spikes'][seed] for spike in cell]
gcs_flat_syns_100 = [((spike % 100) / 100) * 2 * np.pi for seed in syns_100['granule_spikes'][75] for cell in syns_100['granule_spikes'][75][seed] for spike in cell]
gcs_flat_syns_150 = [((spike % 100) / 100) * 2 * np.pi for seed in syns_150['granule_spikes'][75] for cell in syns_150['granule_spikes'][75][seed] for spike in cell]
gcs_flat_syns_200 = [((spike % 100) / 100) * 2 * np.pi for seed in syns_200['granule_spikes'][75] for cell in syns_200['granule_spikes'][75][seed] for spike in cell]

my_pal = {'lec_off': '#2b83ba', '100': '#abdda4', '150': '#fdae61','200': '#d7191c',}
plt.figure()
plt.hist(gcs_flat_syns_off, bins=90, color=my_pal['lec_off'], alpha = alpha)
plt.hist(gcs_flat_syns_100, bins=90, color=my_pal['100'], alpha = alpha)
plt.xlim((0, 2*np.pi))
plt.ylim((0, 1300))
plt.xlabel("Phase (pi)")
plt.ylabel("GC AP Count")
plt.legend(("LEC off", "100 LEC Synapses / Cell"))

plt.figure()
plt.hist(gcs_flat_syns_off, bins=90, color=my_pal['lec_off'], alpha = alpha)
plt.hist(gcs_flat_syns_150, bins=90, color=my_pal['150'], alpha = alpha)
plt.xlim((0, 2*np.pi))
plt.ylim((0, 1300))
plt.xlabel("Phase (pi)")
plt.ylabel("GC AP Count")
plt.legend(("LEC off", "150 LEC Synapses / Cell"))

plt.figure()
plt.hist(gcs_flat_syns_off, bins=90, color=my_pal['lec_off'], alpha = alpha)
plt.hist(gcs_flat_syns_200, bins=90, color=my_pal['200'], alpha = alpha)
plt.xlim((0, 2*np.pi))
plt.ylim((0, 1300))
plt.xlabel("Phase (pi)")
plt.ylabel("GC AP Count")
plt.legend(("LEC off", "200 LEC Synapses / Cell"))

plt.figure()
plt.hist(gcs_flat_syns_off, bins=90, color=my_pal['lec_off'], alpha = alpha, density=True)
plt.hist(gcs_flat_syns_150, bins=90, color=my_pal['150'], alpha = alpha, density=True)
plt.xlim((0, 2*np.pi))
#plt.ylim((0, 1300))
plt.xlabel("Phase (pi)")
plt.ylabel("GC AP Density")
plt.legend(("LEC off", "150 LEC Synapses / Cell"))





