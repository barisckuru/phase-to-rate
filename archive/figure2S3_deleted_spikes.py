# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:38:23 2022

@author: Daniel
"""

from phase_to_rate.neural_coding import load_spikes, rate_n_phase, load_spikes_DMK
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import matplotlib as mpl
import pdb
import seaborn as sns
from elephant.spike_train_generation import homogeneous_poisson_process
from quantities import Hz, s, ms


dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')

trajectories = [75]

n_samples = 20
grid_seeds = np.arange(1,2,1)

tuning = 'full'

ns_grid_phases = np.empty(0)
ns_granule_phases = np.empty(0)
s_grid_phases = np.empty(0)
s_granule_phases = np.empty(0)

start_time_one = 600
end_time_one = 700
start_time_two = 1600
end_time_two = 1700

n_cells = 2000

data_bin_one = [[] for x in range(n_cells)]
data_bin_two = [[] for x in range(n_cells)]
total_spikes_bin_one = np.zeros(2000)
total_spikes_bin_two = np.zeros(2000)

for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    granule_spikes_full = granule_spikes
    cycle_ms = end_time_one - start_time_one
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    if spike >= start_time_one and spike < end_time_one:
                        total_spikes_bin_one[idx] += 1
                        data_bin_one[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)
                    elif spike >= start_time_two and spike < end_time_two:
                        total_spikes_bin_two[idx] += 1
                        data_bin_two[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)

mean_phase_bin_one = np.array([np.nanmean(np.array(cell)) for cell in data_bin_one])
mean_phase_bin_two = np.array([np.nanmean(np.array(cell)) for cell in data_bin_two])
most_active = np.argpartition(total_spikes_bin_one + total_spikes_bin_two, -100)[-100:]

mean_phase_bin_one_full = mean_phase_bin_one.copy()
mean_phase_bin_two_full = mean_phase_bin_two.copy()
total_spikes_bin_one_full = total_spikes_bin_one.copy()
total_spikes_bin_two_full = total_spikes_bin_two.copy()
info_bin_one_full = total_spikes_bin_one_full / (total_spikes_bin_one_full + total_spikes_bin_two_full)
info_bin_two_full = total_spikes_bin_two_full / (total_spikes_bin_one_full + total_spikes_bin_two_full)

data_bin_one_full = data_bin_one.copy()
data_bin_two_full = data_bin_two.copy()


"""NO FEEDBACK"""
tuning = 'no-feedback'
data_bin_one = [[] for x in range(n_cells)]
data_bin_two = [[] for x in range(n_cells)]
total_spikes_bin_one = np.zeros(2000)
total_spikes_bin_two = np.zeros(2000)


for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    granule_spikes_nofb = granule_spikes
    cycle_ms = end_time_one - start_time_one
    total_spikes_per_cell = np.zeros(2000)
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    if spike >= start_time_one and spike < end_time_one:
                        total_spikes_bin_one[idx] += 1
                        data_bin_one[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)
                    elif spike >= start_time_two and spike < end_time_two:
                        total_spikes_bin_two[idx] += 1
                        data_bin_two[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)

mean_phase_bin_one = np.array([np.nanmean(np.array(cell)) for cell in data_bin_one])
mean_phase_bin_two = np.array([np.nanmean(np.array(cell)) for cell in data_bin_two])

mean_phase_bin_one_nofb = mean_phase_bin_one.copy()
mean_phase_bin_two_nofb = mean_phase_bin_two.copy()
total_spikes_bin_one_nofb = total_spikes_bin_one.copy()
total_spikes_bin_two_nofb = total_spikes_bin_two.copy()
info_bin_one_nofb = total_spikes_bin_one_nofb / (total_spikes_bin_one_nofb + total_spikes_bin_two_nofb)
info_bin_two_nofb = total_spikes_bin_two_nofb / (total_spikes_bin_one_nofb + total_spikes_bin_two_nofb)

data_bin_one_nofb = data_bin_one.copy()
data_bin_two_nofb = data_bin_two.copy()

"""TWO EXAMPLE CELLS"""
cell_one_idx = most_active[-1]
cell_two_idx = most_active[-2]

spikes_full = [[] for n in range(2000)]
spikes_cell_one_full = []
spikes_cell_two_full = []
spikes_cell_one_nofb = []
spikes_cell_two_nofb = []

tuning = 'full'
for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    granule_spikes_nofb = granule_spikes
    cycle_ms = end_time_one - start_time_one
    total_spikes_per_cell = np.zeros(2000)
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for spike in poisson_sample[cell_one_idx]:
                spikes_cell_one_full.append(spike)
            for spike in poisson_sample[cell_two_idx]:
                spikes_cell_two_full.append(spike)
            
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    spikes_full[idx].append(spike)
                
spikes_nofb = [[] for n in range(2000)]
tuning = 'no-feedback'
for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    granule_spikes_nofb = granule_spikes
    cycle_ms = end_time_one - start_time_one
    total_spikes_per_cell = np.zeros(2000)
    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for spike in poisson_sample[cell_one_idx]:
                spikes_cell_one_nofb.append(spike)
            for spike in poisson_sample[cell_two_idx]:
                spikes_cell_two_nofb.append(spike)
                
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    spikes_nofb[idx].append(spike)

start_times = np.arange(0, 2000, 100)
end_times = np.arange(100, 2100, 100)

spikes_cell_one_full = np.array(spikes_cell_one_full)
spikes_cell_two_full = np.array(spikes_cell_two_full)
spikes_cell_one_nofb = np.array(spikes_cell_one_nofb)
spikes_cell_two_nofb = np.array(spikes_cell_two_nofb)

mean_phase_cell_one_full = []
mean_phase_cell_two_full = []
mean_phase_cell_one_nofb = []
mean_phase_cell_two_nofb = []

for binn in range(20):
    curr_spikes_cell_one_full = spikes_cell_one_full[(spikes_cell_one_full > start_times[binn]) & (spikes_cell_one_full < end_times[binn])]
    curr_spikes_cell_two_full = spikes_cell_two_full[(spikes_cell_two_full > start_times[binn]) & (spikes_cell_two_full < end_times[binn])]
    curr_spikes_cell_one_nofb = spikes_cell_one_nofb[(spikes_cell_one_nofb > start_times[binn]) & (spikes_cell_one_nofb < end_times[binn])]
    curr_spikes_cell_two_nofb = spikes_cell_two_nofb[(spikes_cell_two_nofb > start_times[binn]) & (spikes_cell_two_nofb < end_times[binn])]
    # pdb.set_trace()
    mean_phase_cell_one_full.append(((curr_spikes_cell_one_full.mean() % cycle_ms) / cycle_ms) * 2 * np.pi)
    mean_phase_cell_two_full.append(((curr_spikes_cell_two_full.mean() % cycle_ms) / cycle_ms) * 2 * np.pi)
    mean_phase_cell_one_nofb.append(((curr_spikes_cell_one_nofb.mean() % cycle_ms) / cycle_ms) * 2 * np.pi)
    mean_phase_cell_two_nofb.append(((curr_spikes_cell_two_nofb.mean() % cycle_ms) / cycle_ms) * 2 * np.pi)

# data_bin_one[idx].append(((spike % cycle_ms) / cycle_ms) * 2 * np.pi)

"""CALCULATE ROC CURVE"""
x = np.concatenate([mean_phase_bin_one_full, mean_phase_bin_two_full, mean_phase_bin_one_nofb, mean_phase_bin_two_nofb])
y = np.concatenate([np.zeros(mean_phase_bin_one_full.size + mean_phase_bin_two_full.size), np.ones(mean_phase_bin_one_nofb.size + mean_phase_bin_two_nofb.size)])

not_nan = np.argwhere(~np.isnan(x))[:,0]

x = x[not_nan]
y = y[not_nan]

parameters = np.arange(0, np.pi * 2, 0.01)

tpr_list = []
fpr_list = []
# sys.exit()
for param in parameters:
    classification = x > param
    tp = np.sum(np.logical_and(classification == 1, y == 1))
    fn = np.sum(np.logical_and(classification == 0, y == 1))
    fp = np.sum(np.logical_and(classification == 1, y == 0))
    tn = np.sum(np.logical_and(classification == 0, y == 0))
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    tpr_list.append(tpr)
    fpr_list.append(fpr)

tpr_list = np.array(tpr_list)
fpr_list = np.array(fpr_list)

optimal_tradeoff = tpr_list - fpr_list

optimal_threshold = parameters[optimal_tradeoff.argmax()]
    
"""PLOTTING"""
plt.rcParams["svg.fonttype"] = "none"
data_bin_one_array = np.array(data_bin_one)
phase_pref_most_active = data_bin_one_array[most_active]
mean_phase_bin_one_array = np.array([np.array(x).mean() for x in phase_pref_most_active])
mean_phase_sorted = np.argsort(mean_phase_bin_one_array)

phase_pref_most_active = data_bin_one_array[most_active]
phase_preference_bin_one_nofb = np.array(data_bin_one)[np.argsort(mean_phase_bin_one_array)]
plt.figure()
plt.eventplot(phase_pref_most_active[mean_phase_sorted])

plt.figure()
plt.scatter(mean_phase_bin_one_full[most_active], mean_phase_bin_two_full[most_active])
plt.xlim((0,2*np.pi))
plt.ylim((0,2*np.pi))
plt.hlines(optimal_threshold,xmin=0, xmax=2*np.pi)
plt.vlines(optimal_threshold,ymin=0, ymax=2*np.pi)
plt.xlabel("Mean Phase Bin # 7")
plt.ylabel("Mean Phase Bin # 17")

plt.figure()
plt.scatter(mean_phase_bin_one_nofb[most_active], mean_phase_bin_two_nofb[most_active])
plt.xlim((0,2*np.pi))
plt.ylim((0,2*np.pi))
plt.hlines(optimal_threshold,xmin=0, xmax=2*np.pi)
plt.vlines(optimal_threshold,ymin=0, ymax=2*np.pi)
plt.xlabel("Mean Phase Bin # 7")
plt.ylabel("Mean Phase Bin # 17")

fig, ax = plt.subplots(2, 2)
ax[0,0].scatter(info_bin_one_full, mean_phase_bin_one_full)
ax[0,1].scatter(info_bin_two_full, mean_phase_bin_two_full)
ax[1,0].scatter(info_bin_one_nofb, mean_phase_bin_one_nofb)
ax[1,1].scatter(info_bin_two_nofb, mean_phase_bin_two_nofb)

for row in ax:
    row[0].set_xlabel("# Spikes Bin 7 / (# Spikes Both Bins)")
    row[1].set_xlabel("# Spikes Bin 17 / (# Spikes Both Bins)")
    row[0].set_ylabel("Mean Phase Bin 7")
    row[1].set_ylabel("Mean Phase bin 17")
    for axis in row:
        axis.set_xlim((-0.1, 1.1))
        axis.set_ylim((0, 2*np.pi))


binarized_bins_full = np.zeros((2000, 20))
for idx, cell in enumerate(granule_spikes_full[75][0]):
    spiked_bins = np.array(cell / 100).astype(int)
    for sp in spiked_bins:
        binarized_bins_full[idx, sp] = 1

binarized_bins_nofb = np.zeros((2000, 20))
for idx, cell in enumerate(granule_spikes_nofb[75][0]):
    spiked_bins = np.array(cell / 100).astype(int)
    for sp in spiked_bins:
        binarized_bins_nofb[idx, sp] = 1

deleted_bins = (binarized_bins_full - binarized_bins_nofb) == -1

deleted_spikes = []
other_spikes = []
for idx, cell in enumerate(granule_spikes_nofb[75][0]):
    for spike in cell:
        # print(idx, cell, spike)
        #pdb.set_trace()
        if int(spike/ 100) in np.argwhere(deleted_bins[idx]):
            deleted_spikes.append(spike)
        else:
            other_spikes.append(spike)
            
deleted_spikes_phase = ((np.array(deleted_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi
other_spikes_phase = ((np.array(other_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi

all_spikes = np.concatenate((deleted_spikes_phase, other_spikes_phase))
all_spikes_label = ['deleted'] * deleted_spikes_phase.size + ['non-deleted'] * other_spikes_phase.size


df_spikes = pd.DataFrame({'spike_phase': all_spikes, 'spike_label': all_spikes_label})

plt.figure()
sns.histplot(data=df_spikes, x='spike_phase', hue='spike_label')

"""NEW PLOTTING"""
random_spikes = [homogeneous_poisson_process(rate=(8.54/(2*np.pi))*Hz, t_start=0.0*s, t_stop=2*np.pi*s) for i in range(100)]

random_spikes_mean_phase = [x.times.mean().magnitude for x in random_spikes]
random_spikes_raw = np.array([x.times.magnitude for x in random_spikes])

mean_phase_sorted = np.argsort(random_spikes_mean_phase)
# plt.figure()
# plt.eventplot(random_spikes_raw[mean_phase_sorted])

# plt.figure()
# plt.eventplot([spikes_cell_one_full, spikes_cell_one_nofb, spikes_cell_two_full, spikes_cell_two_nofb])

bin_center = np.arange(50, 2000, 100)
# plt.plot(bin_center, np.array(mean_phase_cell_one_full), marker='o')
# plt.plot(bin_center, np.array(mean_phase_cell_one_nofb), marker='o')

fig = plt.figure()
gs = fig.add_gridspec(3, 4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2:])
ax4 = fig.add_subplot(gs[1, :2])
ax5 = fig.add_subplot(gs[2, :2])
ax6 = fig.add_subplot(gs[1, 2:])
ax7 = fig.add_subplot(gs[2, 2:])

plt.rcParams["svg.fonttype"] = "none"
data_bin_one_array = np.array(data_bin_one)
phase_pref_most_active = data_bin_one_array[most_active]
mean_phase_bin_one_array = np.array([np.array(x).mean() for x in phase_pref_most_active])
mean_phase_sorted = np.argsort(mean_phase_bin_one_array)

phase_pref_most_active = data_bin_one_array[most_active]
phase_preference_bin_one_nofb = np.array(data_bin_one)[np.argsort(mean_phase_bin_one_array)]
ax1.eventplot(phase_pref_most_active[mean_phase_sorted])

random_spikes = [homogeneous_poisson_process(rate=(8.54/(2*np.pi))*Hz, t_start=0.0*s, t_stop=2*np.pi*s) for i in range(100)]

random_spikes_mean_phase = [x.times.mean().magnitude for x in random_spikes]
random_spikes_raw = np.array([x.times.magnitude for x in random_spikes])

mean_phase_sorted = np.argsort(random_spikes_mean_phase)
ax2.eventplot(random_spikes_raw[mean_phase_sorted])

binarized_bins_nofb = np.zeros((2000, 20))
for idx, cell in enumerate(granule_spikes_nofb[75][0]):
    spiked_bins = np.array(cell / 100).astype(int)
    for sp in spiked_bins:
        binarized_bins_nofb[idx, sp] = 1

deleted_bins = (binarized_bins_full - binarized_bins_nofb) == -1

deleted_spikes = []
other_spikes = []
for idx, cell in enumerate(granule_spikes_nofb[75][0]):
    for spike in cell:
        # print(idx, cell, spike)
        #pdb.set_trace()
        if int(spike/ 100) in np.argwhere(deleted_bins[idx]):
            deleted_spikes.append(spike)
        else:
            other_spikes.append(spike)
            
deleted_spikes_phase = ((np.array(deleted_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi
other_spikes_phase = ((np.array(other_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi

all_spikes = np.concatenate((deleted_spikes_phase, other_spikes_phase))
all_spikes_label = ['deleted'] * deleted_spikes_phase.size + ['non-deleted'] * other_spikes_phase.size

df_spikes = pd.DataFrame({'spike_phase': all_spikes, 'spike_label': all_spikes_label})

sns.histplot(data=df_spikes, x='spike_phase', hue='spike_label', ax=ax3)

eventplot_collection =  ax4.eventplot([spikes_cell_one_full, spikes_cell_one_nofb], color='k')

ax5.plot(bin_center, np.array(mean_phase_cell_one_full), marker='o')
ax5.plot(bin_center, np.array(mean_phase_cell_one_nofb), marker='o')

ax1.set_ylabel("Granule Cell #")
ax1.set_xlabel("Phase")
ax2.set_xlabel("Phase")

ax4_labels = ax4.get_yticklabels()

ax5.set_xlabel("Time (ms)")
ax5.set_ylabel("Mean Phase")

ax5.legend(("Full", "No FB"))

"""PREFERRED PLACE"""
spikes_full = np.array(spikes_full)
spikes_full_preferred_place = np.array([np.array(x).mean() for x in spikes_full])
place_sorted_full = np.argsort(spikes_full_preferred_place[most_active])
spikes_full_phases = [((np.array(np.sort(x)) % cycle_ms) / cycle_ms) * 2 * np.pi for x in spikes_full]

spikes_nofb = np.array(spikes_nofb)
spikes_nofb_preferred_place = np.array([np.array(x).mean() for x in spikes_nofb])
place_sorted_nofb = np.argsort(spikes_nofb_preferred_place[most_active])
spikes_nofb_phases = [((np.array(np.sort(x)) % cycle_ms) / cycle_ms) * 2 * np.pi for x in spikes_nofb]

# spikes_cell_one_full[(spikes_cell_one_full > start_times[binn]) & (spikes_cell_one_full < end_times[binn])]

mean_phases_nofb = [[] for x in range(2000)]
for idx, spikes in enumerate(spikes_nofb):
    for binn in range(20):
        spikes = np.array(spikes)
        curr_spikes = spikes[(spikes > start_times[binn]) & (spikes < end_times[binn])]
        mean_phases_nofb[idx].append(((curr_spikes.mean()  % cycle_ms) / cycle_ms) * 2 * np.pi)
        
mean_phases_full = [[] for x in range(2000)]
for idx, spikes in enumerate(spikes_full):
    for binn in range(20):
        spikes = np.array(spikes)
        curr_spikes = spikes[(spikes > start_times[binn]) & (spikes < end_times[binn])]
        mean_phases_full[idx].append(((curr_spikes.mean()  % cycle_ms) / cycle_ms) * 2 * np.pi)

color_norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
phases_full_norm = np.array([color_norm(x) for x in spikes_full_phases])[most_active]
phases_nofb_norm = np.array([color_norm(x) for x in spikes_nofb_phases])[most_active]

viridis = mpl.colormaps['viridis']

nofb_collection = ax6.eventplot(spikes_nofb[most_active][place_sorted_nofb])

for idx, col in enumerate(nofb_collection):
    col.set_colors(viridis(phases_nofb_norm[place_sorted_nofb][idx]))

plt.colorbar(mpl.cm.ScalarMappable(norm=color_norm, cmap=viridis), ax=ax6)

full_collection = ax7.eventplot(spikes_full[most_active][place_sorted_nofb])

"""PREFERRED PLACE ALIGNMENT"""
spikes_nofb_preferred_bin = spikes_nofb_preferred_place // 100
aligned_phases_nofb = a = np.empty((2000, 41))
aligned_phases_nofb[:] = np.nan

central_bin = 20
for idx, pref_bin in enumerate(spikes_nofb_preferred_bin):
    if not np.isnan(pref_bin):
        first_bin = int(central_bin - pref_bin)
        aligned_phases_nofb[idx, first_bin:first_bin+20] = mean_phases_nofb[idx]

spikes_full_preferred_bin = spikes_full_preferred_place // 100
aligned_phases_full = a = np.empty((2000, 41))
aligned_phases_full[:] = np.nan

for idx, pref_bin in enumerate(spikes_nofb_preferred_bin):
    if not np.isnan(pref_bin):
        first_bin = int(central_bin - pref_bin)
        aligned_phases_full[idx, first_bin:first_bin+20] = mean_phases_full[idx]

aligned_mean_phase_nofb = np.nanmean(aligned_phases_nofb[most_active], axis=0)
aligned_std_phase_nofb = np.nanstd(aligned_phases_nofb[most_active], axis=0)
    
aligned_mean_phase_full = np.nanmean(aligned_phases_full[most_active], axis=0)
aligned_std_phase_full = np.nanstd(aligned_phases_full[most_active], axis=0)
    

"""COLORED EVENTPLOT"""
"""
color_norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
phases_cell_one_nofb = ((np.sort(spikes_cell_one_nofb)  % cycle_ms) / cycle_ms) * 2 * np.pi
phases_norm = color_norm(phases_cell_one_nofb)

viridis = mpl.colormaps['viridis']

phases_colors = viridis(phases_norm)

eventplot_collection[1].set_colors(phases_colors)
"""