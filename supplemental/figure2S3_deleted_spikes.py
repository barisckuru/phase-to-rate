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
grid_seed = 1

ns_grid_phases = np.empty(0)
ns_granule_phases = np.empty(0)
s_grid_phases = np.empty(0)
s_granule_phases = np.empty(0)

start_time_one = 600
end_time_one = 700
start_time_two = 1600
end_time_two = 1700

n_cells = 2000

"""Gather spikes from 20 poisson seeds"""
spikes_full = [[] for n in range(2000)]
tuning = 'full'
path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
granule_spikes_full = granule_spikes
cycle_ms = end_time_one - start_time_one
total_spikes_per_cell = np.zeros(2000)
for k1 in granule_spikes.keys():
    for poisson_sample in granule_spikes[k1]:
        for idx, cell in enumerate(poisson_sample):
            for spike in cell:
                spikes_full[idx].append(spike)

spikes_nofb = [[] for n in range(2000)]
tuning = 'no-feedback'
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
                spikes_nofb[idx].append(spike)

"""Sort spikes"""
spikes_full = np.array([np.sort(x) for x in spikes_full], dtype=object)
spikes_nofb = np.array([np.sort(x) for x in spikes_nofb], dtype=object)

"""Calculate the phases"""
spikes_full_phases = [((np.array(x) % cycle_ms) / cycle_ms) * 2 * np.pi for x in spikes_full]
spikes_nofb_phases = [((np.array(x) % cycle_ms) / cycle_ms) * 2 * np.pi for x in spikes_nofb]

"""Find the 100 most active cells across poisson seeds"""
total_spikes_nofb = np.array([len(spikes) for spikes in spikes_nofb])
most_active = np.argpartition(total_spikes_nofb, -100)[-100:]
# most_active = np.argpartition(total_spikes_nofb, -200)[100:]

cell_one_idx = most_active[-1]
cell_two_idx = most_active[-2]

start_times = np.arange(0, 2000, 100)
end_times = np.arange(100, 2100, 100)

"""Calculate the mean phase in 20 bins"""
mean_phase_full = np.empty((n_cells, 20))
mean_phase_nofb = np.empty((n_cells, 20))
mean_phase_full[:] = np.nan
mean_phase_nofb[:] = np.nan
n_spikes_full = np.zeros((n_cells, 20))
n_spikes_nofb = np.zeros((n_cells, 20))


for cell_idx in range(n_cells):
    for binn in range(20):
        curr_spikes_full_idc = (spikes_full[cell_idx] > start_times[binn]) & (spikes_full[cell_idx] < end_times[binn])
        curr_spikes_nofb_idc = (spikes_nofb[cell_idx] > start_times[binn]) & (spikes_nofb[cell_idx] < end_times[binn])
        mean_phase_full[cell_idx, binn] = spikes_full_phases[cell_idx][curr_spikes_full_idc].mean()
        mean_phase_nofb[cell_idx, binn] = spikes_nofb_phases[cell_idx][curr_spikes_nofb_idc].mean()
        n_spikes_full[cell_idx, binn] = spikes_full_phases[cell_idx][curr_spikes_full_idc].size
        n_spikes_nofb[cell_idx, binn] = spikes_nofb_phases[cell_idx][curr_spikes_nofb_idc].size

"""Distinguish deleted from non-deleted spikes"""

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
            



"""PLOTTING"""
"""SETUP THE FIGURE LAYOUT AND STYLE"""
plt.rcParams["svg.fonttype"] = "none"
fig = plt.figure()
gs = fig.add_gridspec(3, 4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2:])
ax4 = fig.add_subplot(gs[1, :2])
ax5 = fig.add_subplot(gs[2, :2])
ax6 = fig.add_subplot(gs[1, 2:])
ax7 = fig.add_subplot(gs[2, 2:])

color_norm_cycle = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
viridis = mpl.colormaps['viridis']


"""PHASE PREFERENCE WITHIN SINGLE BIN"""
"""Extract data for example bin"""
data_bin_one = [((cell[(cell > start_time_one) & (cell < end_time_one)] % cycle_ms) / cycle_ms) * 2 * np.pi for cell in spikes_nofb]

"""Sort by mean phase"""
data_bin_one_array = np.array(data_bin_one)
phase_pref_most_active = data_bin_one_array[most_active]
mean_phase_bin_one_array = np.array([np.array(x).mean() for x in phase_pref_most_active])
mean_phase_sorted = np.argsort(mean_phase_bin_one_array)

#phase_pref_most_active = data_bin_one_array[most_active]
# phase_preference_bin_one_nofb = np.array(data_bin_one)[np.argsort(mean_phase_bin_one_array)]
ax1.eventplot(phase_pref_most_active[mean_phase_sorted])

"""Create random event data with equal average rate for visual comparison"""
average_events = np.array([x.size for x in phase_pref_most_active[mean_phase_sorted]]).mean()
random_spikes = [homogeneous_poisson_process(rate=(average_events/(2*np.pi))*Hz, t_start=0.0*s, t_stop=2*np.pi*s) for i in range(100)]
random_spikes_mean_phase = [x.times.mean().magnitude for x in random_spikes]
random_spikes_raw = np.array([x.times.magnitude for x in random_spikes])

mean_phase_sorted = np.argsort(random_spikes_mean_phase)

bin_center = np.arange(50, 2000, 100)
ax2.eventplot(random_spikes_raw[mean_phase_sorted])

sns.histplot(data=df_spikes, x='spike_phase', hue='spike_label', ax=ax3)

"""Eventplot of single example cell and mean phase per bin"""
example_cell_idx = most_active[0]

eventplot_collection =  ax4.eventplot([spikes_full[example_cell_idx], spikes_nofb[example_cell_idx]], color='k')

example_norm_phases_nofb = color_norm_cycle(spikes_nofb_phases[example_cell_idx])
example_colors_nofb = viridis(example_norm_phases_nofb)

example_norm_phases_full = color_norm_cycle(spikes_full_phases[example_cell_idx])
example_colors_full = viridis(example_norm_phases_full)

eventplot_collection[0].set_colors(example_colors_full)
eventplot_collection[1].set_colors(example_colors_nofb)

ax5.plot(bin_center, np.array(mean_phase_full[example_cell_idx]), marker='o')
ax5.plot(bin_center, np.array(mean_phase_nofb[example_cell_idx]), marker='o')

ax1.set_ylabel("Granule Cell #")
ax1.set_xlabel("Phase")
ax2.set_xlabel("Phase")

ax4_labels = ax4.get_yticklabels()

ax5.set_xlabel("Time (ms)")
ax5.set_ylabel("Mean Phase")

ax5.legend(("Full", "No FB"))

"""Plot 100 most active cells sorted by preferred place."""
spikes_full_preferred_place = np.array([np.array(x).mean() for x in spikes_full])
place_sorted_full = np.argsort(spikes_full_preferred_place[most_active])

spikes_nofb_preferred_place = np.array([np.array(x).mean() for x in spikes_nofb])
place_sorted_nofb = np.argsort(spikes_nofb_preferred_place[most_active])

phases_full_norm = np.array([color_norm_cycle(x) for x in spikes_full_phases])[most_active]
phases_nofb_norm = np.array([color_norm_cycle(x) for x in spikes_nofb_phases])[most_active]

nofb_collection = ax6.eventplot(spikes_nofb[most_active][place_sorted_nofb])

for idx, col in enumerate(nofb_collection):
    col.set_colors(viridis(phases_nofb_norm[place_sorted_nofb][idx]))

full_collection = ax7.eventplot(spikes_full[most_active][place_sorted_nofb])

for idx, col in enumerate(full_collection):
    col.set_colors(viridis(phases_full_norm[place_sorted_nofb][idx]))

plt.colorbar(mpl.cm.ScalarMappable(norm=color_norm_cycle, cmap=viridis), ax=ax6, label="Spike Phase", fraction=0.05)
plt.colorbar(mpl.cm.ScalarMappable(norm=color_norm_cycle, cmap=viridis), ax=ax7, label="Spike Phase", fraction=0.05)

"""PREFERRED PLACE ALIGNMENT"""
spikes_nofb_preferred_bin = spikes_full_preferred_place // 100
aligned_phases_nofb = np.empty((2000, 41))
aligned_phases_nofb[:] = np.nan
aligned_rates_nofb = np.zeros((2000, 41))

central_bin = 20
for idx, pref_bin in enumerate(spikes_nofb_preferred_bin):
    if not np.isnan(pref_bin):
        first_bin = int(central_bin - pref_bin)
        aligned_phases_nofb[idx, first_bin:first_bin+20] = mean_phase_nofb[idx]
        aligned_rates_nofb[idx, first_bin:first_bin+20] = n_spikes_nofb[idx]

spikes_full_preferred_bin = spikes_full_preferred_place // 100
aligned_phases_full = a = np.empty((2000, 41))
aligned_phases_full[:] = np.nan
aligned_rates_full = np.zeros((2000, 41))

for idx, pref_bin in enumerate(spikes_nofb_preferred_bin):
    if not np.isnan(pref_bin):
        first_bin = int(central_bin - pref_bin)
        aligned_phases_full[idx, first_bin:first_bin+20] = mean_phase_full[idx]
        aligned_rates_full[idx, first_bin:first_bin+20] = n_spikes_full[idx]

aligned_mean_phase_nofb = np.nanmean(aligned_phases_nofb[most_active], axis=0)
aligned_std_phase_nofb = np.nanstd(aligned_phases_nofb[most_active], axis=0)
    
aligned_mean_phase_full = np.nanmean(aligned_phases_full[most_active], axis=0)
aligned_std_phase_full = np.nanstd(aligned_phases_full[most_active], axis=0)

aligned_mean_spikes_nofb = np.nanmean(aligned_rates_nofb[most_active], axis=0)
aligned_std_spikes_nofb = np.nanstd(aligned_rates_nofb[most_active], axis=0)
    
aligned_mean_spikes_full = np.nanmean(aligned_rates_full[most_active], axis=0)
aligned_std_spikes_full = np.nanstd(aligned_rates_full[most_active], axis=0)

n_nofb = (~np.isnan(aligned_phases_nofb[most_active])).sum(axis=0)
n_full = (~np.isnan(aligned_phases_full[most_active])).sum(axis=0)

fig = plt.figure()
gs = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

ax1.errorbar(x=range(-20,21), y=aligned_mean_phase_nofb, yerr=aligned_std_phase_nofb/np.sqrt(n_nofb))
ax1.errorbar(x=range(-20,21), y=aligned_mean_phase_full, yerr=aligned_std_phase_full/np.sqrt(n_full))
ax1.set_xlim((-20, 20))

ax2.errorbar(x=range(-20,21), y=aligned_mean_spikes_nofb, yerr=aligned_std_spikes_nofb/np.sqrt(n_nofb))
ax2.errorbar(x=range(-20,21), y=aligned_mean_spikes_full, yerr=aligned_std_spikes_full/np.sqrt(n_full))
ax2.set_xlim((-20, 20))

ax3.plot(range(-20,21), n_nofb)
ax3.plot(range(-20,21), n_full)
ax3.set_xlim((-20, 20))

ax1.legend(("no fb", 'full'))

ax1.set_ylabel("Mean Phase +- SEM")
ax2.set_ylabel("Mean Number of Spikes +- SEM")
ax3.set_ylabel("Number of datapoints")

ax3.set_xlabel("Bins, most active bin aligned at 0")

"""COLORED EVENTPLOT"""
"""
color_norm = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
phases_cell_one_nofb = ((np.sort(spikes_cell_one_nofb)  % cycle_ms) / cycle_ms) * 2 * np.pi
phases_norm = color_norm(phases_cell_one_nofb)

viridis = mpl.colormaps['viridis']

phases_colors = viridis(phases_norm)

eventplot_collection[1].set_colors(phases_colors)
"""