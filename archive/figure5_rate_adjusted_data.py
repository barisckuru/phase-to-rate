# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:38:23 2022

@author: Daniel
"""

from phase_to_rate.neural_coding import load_spikes, rate_n_phase, load_spikes_DMK
from phase_to_rate.information_measure import skaggs_information
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
import scipy.signal
import copy
import shelve


dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')

trajectories = [75]

n_samples = 20
grid_seed = 1

ns_grid_phases = np.empty(0)
ns_granule_phases = np.empty(0)
s_grid_phases = np.empty(0)
s_granule_phases = np.empty(0)

cycle_ms = 100
start_time_one = 600
end_time_one = 700
start_time_two = 1600
end_time_two = 1700

n_cells = 2000

"""Gather spikes from 20 poisson seeds"""
tunings = ['full', 'no-feedback', 'no-feedforward', 'disinhibited']
spikes_full = {}

for tuning in tunings:
    curr_spikes_full = [[] for n in range(2000)]
    if tuning == 'full':
        path = os.path.join(results_dir, 'adjusted_data', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
        ns_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
        granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    else:
        path = os.path.join(results_dir, 'adjusted_data', str(tuning),  'collective', 'grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_')
        ns_path = (path + str(grid_seed) + "_[75]_100-119_2000_shuffled_"+str(tuning))
        granule_spikes = load_spikes_DMK(ns_path, "granule", trajectories, n_samples)

    for k1 in granule_spikes.keys():
        for poisson_sample in granule_spikes[k1]:
            for idx, cell in enumerate(poisson_sample):
                for spike in cell:
                    curr_spikes_full[idx].append(spike)
    spikes_full[tuning] = curr_spikes_full

"""Sort spikes"""
for tuning in tunings:
    spikes_full[tuning] = np.array([np.sort(x) for x in spikes_full[tuning]], dtype=object)

"""Calculate binary spike trains"""
binary_spikes_full = {}
for tuning in tunings:
    curr_binary_spikes = np.zeros((spikes_full[tuning].shape[0], 20000))
    for idx, cell in enumerate(spikes_full[tuning]):
        spike_idc = np.array(cell * 10, dtype=int)
        curr_binary_spikes[idx, spike_idc - 1] = 1
    binary_spikes_full[tuning] = curr_binary_spikes



"""Apply a gaussian filter to the spike trains"""
gaussian_width = 10000
gaussian_sigma = 400
gaussian_kernel = scipy.signal.windows.gaussian(gaussian_width, gaussian_sigma)
binary_spikes_convolved = {}
for tuning in tunings:
    curr_spikes_convolved = np.zeros((spikes_full[tuning].shape[0], 20000))
    for idx, cell in enumerate(binary_spikes_full[tuning]):
        curr_conv = scipy.signal.convolve(cell, gaussian_kernel, mode='same')
        if not curr_conv.sum() == 0:
            curr_spikes_convolved[idx] = curr_conv / curr_conv.sum()
        else:
            curr_spikes_convolved[idx] = 0
    binary_spikes_convolved[tuning] = curr_spikes_convolved

"""Calculate density for the different conditions"""
spike_density = {}
spike_density_flattened = {}
for spike_condition in tunings:
    spike_density[spike_condition] = {}
    spike_density_flattened[spike_condition] = {}
    for density_condition in tunings:
        curr_spike_density = copy.deepcopy(spikes_full[spike_condition])
        for cell_idx, cell in enumerate(spikes_full[spike_condition]):
            for spike_idx, spike in enumerate(cell):
                spike_binary_index = int(spike * 10)
                curr_spike_density[cell_idx][spike_idx - 1] = binary_spikes_convolved[density_condition][cell_idx, spike_binary_index - 1]
        spike_density[spike_condition][density_condition] = curr_spike_density
        spike_density_flattened[spike_condition][density_condition] = np.array([x for cell in curr_spike_density for x in cell])


"""Calculate the phases"""
spikes_phases = {}
spikes_phases_flat = {}
for tuning in tunings:
    curr_phases = np.array([((np.array(x) % cycle_ms) / cycle_ms) * 2 * np.pi for x in spikes_full[tuning]])
    spikes_phases[tuning] = curr_phases
    spikes_phases_flat[tuning] = np.array([x for cell in curr_phases for x in cell])

"""Distinguish deleted from non-deleted spikes"""
binarized_bins_full = np.zeros((2000, 20))
for idx, cell in enumerate(spikes_full['full']):
    spiked_bins = np.array(cell / 100 - 1).astype(int)
    for sp in spiked_bins:
        binarized_bins_full[idx, sp] = 1

binarized_bins_nofb = np.zeros((2000, 20))
for idx, cell in enumerate(spikes_full['no-feedback']):
    spiked_bins = np.array(cell / 100 - 1).astype(int)
    for sp in spiked_bins:
        binarized_bins_nofb[idx, sp] = 1

deleted_bins = (binarized_bins_full - binarized_bins_nofb) == -1

deleted_spikes = []
other_spikes = []
for idx, cell in enumerate(spikes_full['no-feedback']):
    for spike in cell:
        if int(spike/ 100) in np.argwhere(deleted_bins[idx]):
            deleted_spikes.append(spike)
        else:
            other_spikes.append(spike)
            
deleted_spikes_phase = ((np.array(deleted_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi
other_spikes_phase = ((np.array(other_spikes) % cycle_ms) / cycle_ms) * 2 * np.pi

all_spikes = np.concatenate((deleted_spikes_phase, other_spikes_phase))
all_spikes_label = ['deleted'] * deleted_spikes_phase.size + ['other'] * other_spikes_phase.size

df_spikes = pd.DataFrame({'spike_phase': all_spikes, 'spike_label': all_spikes_label})

"""Find the 100 most active cells across poisson seeds"""
n_max_cells = 50
total_spikes_dis = np.array([len(spikes) for spikes in spikes_full['disinhibited']])
total_spikes_full = np.array([len(spikes) for spikes in spikes_full['full']])
most_active = np.argpartition(total_spikes_dis, -n_max_cells)[-n_max_cells:]
# most_active = np.argpartition(total_spikes_nofb, -200)[100:]

start_times = np.arange(0, 2000, 100)
end_times = np.arange(100, 2100, 100)

"""CALCULATE SKAGGS FOR EACH CELL"""
skaggs = {}
for tuning in tunings:
    skaggs[tuning] = skaggs_information(spikes_full[tuning], 2000, 250, agg=False)

full_nofb_change = skaggs['full'] - skaggs['no-feedback']
full_nofb_percent_change = (skaggs['full'] - skaggs['no-feedback']) * 100

skaggs_included_cells = total_spikes_full > 8

skaggiest = np.argpartition(full_nofb_percent_change[skaggs_included_cells], -n_max_cells)[-n_max_cells:]

"""PLOTTING"""
"""SETUP THE FIGURE LAYOUT AND STYLE"""
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams["svg.fonttype"] = "none"
fig = plt.figure()
gs = fig.add_gridspec(3, 4)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2:])
ax4 = fig.add_subplot(gs[1, :2])
ax5 = fig.add_subplot(gs[2, :2])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[2, 2])
ax8 = fig.add_subplot(gs[1, 3])
ax9 = fig.add_subplot(gs[2, 3])

color_norm_cycle = mpl.colors.Normalize(vmin=0, vmax=2*np.pi)
viridis = mpl.colormaps['viridis']

"""PHASE PREFERENCE WITHIN SINGLE BIN"""
"""Extract data for example bin"""
data_bin_one = [((cell[(cell > start_time_one) & (cell < end_time_one)] % cycle_ms) / cycle_ms) * 2 * np.pi for cell in spikes_full['disinhibited']]

"""Sort by mean phase"""
data_bin_one_array = np.array(data_bin_one)
# phase_pref_most_active = data_bin_one_array[most_active]
phase_pref_most_active = data_bin_one_array[skaggs_included_cells][skaggiest]
mean_phase_bin_one_array = np.array([np.array(x).mean() for x in phase_pref_most_active])
mean_phase_sorted = np.argsort(mean_phase_bin_one_array)

#phase_pref_most_active = data_bin_one_array[most_active]
# phase_preference_bin_one_nofb = np.array(data_bin_one)[np.argsort(mean_phase_bin_one_array)]
ax1.eventplot(phase_pref_most_active[mean_phase_sorted])

ax1.set_ylabel("Granule Cell #")
ax1.set_xlabel("Phase")
ax1.set_xlim((0, 2*np.pi))

"""Create random event data with equal average rate for visual comparison"""
average_events = np.array([x.size for x in phase_pref_most_active[mean_phase_sorted]]).mean()
random_spikes = [homogeneous_poisson_process(rate=(average_events/(2*np.pi))*Hz, t_start=0.0*s, t_stop=2*np.pi*s) for i in range(n_max_cells)]
random_spikes_mean_phase = [x.times.mean().magnitude for x in random_spikes]
random_spikes_raw = np.array([x.times.magnitude for x in random_spikes])

mean_phase_sorted = np.argsort(random_spikes_mean_phase)

bin_center = np.arange(50, 2000, 100)
ax2.eventplot(random_spikes_raw[mean_phase_sorted])

ax2.set_xlabel("Phase")
ax2.set_xlim((0, 2*np.pi))

"""DELETED_SPIKES"""
sns.histplot(data=df_spikes, x='spike_phase', hue='spike_label', ax=ax3)


"""Eventplot of single example cell and mean phase per bin"""
# example_cell_idx = most_active[10]
example_cell_idx = skaggiest[-1]

eventplot_collection =  ax4.eventplot([spikes_full['disinhibited'][skaggs_included_cells][example_cell_idx],
                                       spikes_full['no-feedback'][skaggs_included_cells][example_cell_idx],
                                       spikes_full['no-feedforward'][skaggs_included_cells][example_cell_idx],
                                       spikes_full['full'][skaggs_included_cells][example_cell_idx]],
                                       color='k')

plt.colorbar(mpl.cm.ScalarMappable(norm=color_norm_cycle, cmap=viridis), ax=ax4, label="Spike Phase", fraction=0.1, location='top')

ax4.set_yticks(np.arange(4))
ax4.set_yticklabels(['disinh.', 'no FB', 'no FF', 'Full'])
ax4.set_xlim([0, 2000])

example_cell_colors = {}
for t in tunings:
    curr_norm = color_norm_cycle(spikes_phases[t][skaggs_included_cells][example_cell_idx])
    example_cell_colors[t] = viridis(curr_norm)

eventplot_collection[0].set_colors(example_cell_colors['disinhibited'])
eventplot_collection[1].set_colors(example_cell_colors['no-feedback'])
eventplot_collection[2].set_colors(example_cell_colors['no-feedforward'])
eventplot_collection[3].set_colors(example_cell_colors['full'])

"""PLot densities for example cell"""
t=np.arange(0, 2000, 0.1)
ax5.plot(t, binary_spikes_convolved['full'][skaggs_included_cells][example_cell_idx])
ax5.plot(t, binary_spikes_convolved['no-feedforward'][skaggs_included_cells][example_cell_idx])
ax5.plot(t, binary_spikes_convolved['no-feedback'][skaggs_included_cells][example_cell_idx])
ax5.plot(t, binary_spikes_convolved['disinhibited'][skaggs_included_cells][example_cell_idx])
ax5.legend(("full", "no-feedforward", "no-feedback", "disinhibited"))
ax5.set_xlim([0, 2000])
ax5.set_xlabel("Time (ms)")

"""Plot 100 most active cells sorted by preferred place."""
spikes_full_preferred_place = np.array([np.array(x).mean() for x in spikes_full['full']])
place_sorted_full = np.argsort(spikes_full_preferred_place[skaggs_included_cells][skaggiest])

spikes_nofb_preferred_place = np.array([np.array(x).mean() for x in spikes_full['no-feedback']])
place_sorted_nofb = np.argsort(spikes_nofb_preferred_place[skaggs_included_cells][skaggiest])

phases_full_norm = np.array([color_norm_cycle(x) for x in spikes_phases['full']])[skaggs_included_cells][skaggiest]
phases_nofb_norm = np.array([color_norm_cycle(x) for x in spikes_phases['no-feedback']])[skaggs_included_cells][skaggiest]

nofb_collection = ax6.eventplot(spikes_full['no-feedback'][skaggs_included_cells][skaggiest][place_sorted_nofb])

ax6.set_ylabel("Place Sorted Cells no-feedback Network")

for idx, col in enumerate(nofb_collection):
    col.set_colors(viridis(phases_nofb_norm[place_sorted_nofb][idx]))

full_collection = ax7.eventplot(spikes_full['full'][skaggs_included_cells][skaggiest][place_sorted_nofb])
ax7.set_ylabel("Place Sorted Cells Full Network")

ax7.set_xlabel("Time (ms)")

for idx, col in enumerate(full_collection):
    col.set_colors(viridis(phases_full_norm[place_sorted_nofb][idx]))

nofb_collection = ax8.eventplot(spikes_full['no-feedback'][skaggs_included_cells][skaggiest][place_sorted_nofb])

ax8.set_xlim([1900, 2000])

for idx, col in enumerate(nofb_collection):
    col.set_colors(viridis(phases_nofb_norm[place_sorted_nofb][idx]))

full_collection = ax9.eventplot(spikes_full['full'][skaggs_included_cells][skaggiest][place_sorted_nofb])
ax9.set_ylabel("Place Sorted Cells Full Network")

for idx, col in enumerate(full_collection):
    col.set_colors(viridis(phases_full_norm[place_sorted_nofb][idx]))

ax9.set_xlim([1900, 2000])

ax9.set_xlabel("Time (ms)")

"""PREFERRED PLACE ALIGNMENT"""
spikes_nofb_preferred_bin = spikes_full['no-feedback'] // 100
aligned_phases_nofb = np.empty((2000, 41))
aligned_phases_nofb[:] = np.nan
aligned_rates_nofb = np.zeros((2000, 41))

central_bin = 20
for idx, pref_bin in enumerate(spikes_nofb_preferred_bin):
    if not np.isnan(pref_bin):
        first_bin = int(central_bin - pref_bin)
        aligned_phases_nofb[idx, first_bin:first_bin+20] = mean_phase_nofb[idx]
        aligned_rates_nofb[idx, first_bin:first_bin+20] = n_spikes_nofb[idx]

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

"""Phase Density Plot"""
plt.figure()
plt.scatter(spikes_nofb_phases_flat, spikes_nofb_full_density_flat)
plt.figure()
plt.scatter(spikes_nofb_phases_flat, spikes_nofb_nofb_density_flat)

fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.scatter(spikes_phases_flat['no-feedback'], spike_density_flattened['no-feedback']['full'], alpha=0.3)
ax2.scatter(spikes_phases_flat['no-feedback'], spike_density_flattened['no-feedback']['no-feedback'], alpha=0.3)

hist_full, xbins_full, ybins_full = np.histogram2d(x=spikes_phases_flat['no-feedback'], y=spike_density_flattened['no-feedback']['full'], bins=(np.arange(0, 2*np.pi, 0.1), np.arange(0, 0.0014, 0.00001)))
ax3.imshow(hist_full.T, cmap='jet', interpolation=None, origin='lower', extent=[0, 2*np.pi, 0, 0.0014], aspect='auto')

hist_self, xbins_self, ybins_self = np.histogram2d(x=spikes_phases_flat['no-feedback'], y=spike_density_flattened['no-feedback']['no-feedback'], bins=(np.arange(0, 2*np.pi, 0.1), np.arange(0, 0.0014, 0.00001)))
ax4.imshow(hist_full.T, cmap='jet', interpolation=None, origin='lower', extent=[0, 2*np.pi, 0, 0.0014], aspect='auto')

for cax in [ax1, ax2, ax3, ax4]:
    cax.set_xlabel("Spike Phase No-Feedback Spikes")

for cax in [ax1, ax2]:
    cax.set_ylim((0, 0.0014))
    cax.set_xlabel("Spike Phase No-Feedback Spikes")

for cax in [ax1, ax3]:
    cax.set_ylabel("Spike Density in Full Condition")

for cax in [ax2, ax4]:
    cax.set_ylabel("Spike Density in No-Feedback Condition")


fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax1.scatter(spikes_phases_flat['no-feedback'], spike_density_flattened['no-feedback']['no-feedback'] - spike_density_flattened['no-feedback']['full'])
ax1.set_xlabel("Pase No-Feedback")
ax1.set_ylabel("No-feedback minus full density")

"""
fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['no-feedback'])
ax2.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['no-feedforward'])
ax3.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['disinhibited'])
ax4.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedforward'])

fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['no-feedback'])
ax2.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['no-feedforward'])
ax3.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedback'])
ax4.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedforward'])

loss_no_feedback = spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedback']
loss_no_feedforward = spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedforward']

data_hope = pd.DataFrame(data={'nofb_loss': loss_no_feedback,
                               'noff_loss': loss_no_feedforward,
                               'phase': spikes_phases_flat['disinhibited']})

sns.scatterplot(data=data_hope, x='nofb_loss', y='noff_loss', hue='phase')

fig = plt.figure()
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

ax1.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['no-feedback'])
ax2.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['no-feedforward'])
ax3.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedback'])
ax4.scatter(spikes_phases_flat['disinhibited'], spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedforward'])

loss_no_feedback = spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedback']
loss_no_feedforward = spike_density_flattened['disinhibited']['disinhibited'] - spike_density_flattened['disinhibited']['no-feedforward']

data_hope = pd.DataFrame(data={'nofb_loss': loss_no_feedback,
                               'noff_loss': loss_no_feedforward,
                               'phase': spikes_phases_flat['disinhibited']})

sns.scatterplot(data=data_hope, x='nofb_loss', y='noff_loss', hue='phase', alpha=0.5)
"""





