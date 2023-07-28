#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:29:15 2022

@author: baris
"""
import pickle
import numpy as np
# import seaborn as sns
import pandas as pd
from phase_to_rate import grid_model
from phase_to_rate.figure_functions import (_make_cmap, _precession_spikes,
                              _adjust_box_widths)
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import SymLogNorm
import matplotlib.font_manager
from scipy.stats import pearsonr,  spearmanr
from scipy import stats
import copy
import os
import sys
import shelve
from copy import deepcopy


# file directory
dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data', 'multiple_celltypes_adjusted')
save_dir = dirname

# plotting settings
plt.rc('font', size=18) #controls default text size
plt.rcParams["svg.fonttype"] = "none"
cm=1/2.54

# =============================================================================
# Figure 2 A
# =============================================================================

tunings = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
shufflings = ['non-shuffled']

all_files = set(['.'.join(x.split('.')[:-1]) for x in os.listdir(results_dir)])

poisson_seeds = range(100, 110)
grid_seeds = range(1, 6)
cell_types = range(4)

all_spikes = {}
for t in tunings:
    all_spikes[t] = {}
    for s in shufflings:
        all_spikes[t][s] = {}
            
for f in all_files:
    curr_file = shelve.open(os.path.join(results_dir, f))
    split = f.split('_')
    grid_seed = int(split[8])
    shuffling = split[12]
    tuning = split[13]
    all_spikes[tuning][shuffling][grid_seed] = dict(curr_file)

all_phases = deepcopy(all_spikes)

# Calculate phases
for t in tunings:
    for s in shufflings:
        for g in grid_seeds:
            for ps in poisson_seeds:
                all_phases[t][s][g]['grid_spikes'][75][ps] = [((spikes % 100) / 100) * (np.pi * 2) for spikes in all_spikes[t][s][g]['grid_spikes'][75][ps]]
                
                all_phases[t][s][g]['all_spikes'][75][ps] = list(all_phases[t][s][g]['all_spikes'][75][ps])
                for ct in cell_types:
                    all_phases[t][s][g]['all_spikes'][75][ps][ct] = [((spikes % 100) / 100) * (np.pi * 2) for spikes in all_spikes[t][s][g]['all_spikes'][75][ps][ct]]

all_phases_flattened = {}
for t in tunings:
    all_phases_flattened[t] = {}
    for s in shufflings:
        all_phases_flattened[t][s] = {}
        for ct in ['grid', 'granule', 'mossy', 'basket', 'hipp']:
            all_phases_flattened[t][s][ct] = np.array([])

for t in tunings:
    for s in shufflings:
        for g in grid_seeds:
            for ps in poisson_seeds:
                for cell in all_phases[t][s][g]['grid_spikes'][75][ps]:
                    all_phases_flattened[t][s]['grid'] = np.concatenate((all_phases_flattened[t][s]['grid'], cell))
                for ct_idx, ct in enumerate(['granule', 'mossy', 'basket', 'hipp']):
                    for cell in all_phases[t][s][g]['all_spikes'][75][ps][ct_idx]:
                        all_phases_flattened[t][s][ct] = np.concatenate((all_phases_flattened[t][s][ct], cell))

all_phases_flattened_adjusted = deepcopy(all_phases_flattened)

"""With mossy cells"""
fig, ax = plt.subplots(2,2)
my_pal = {'grid': '#1b9e77', 'granule': '#d95f02', 'mossy': '#7570b3', 'basket': '#e7298a', 'hipp': '#66a61e'}
ax = ax.flatten()
alpha = 0.6
shuffling = 'non-shuffled'

density=False
for idx, t in enumerate(tunings):
    ax[idx].set_title(t)
    ax[idx].set_xlabel("Phase (pi)")
    ax[idx].set_ylabel("Probability Density of Spikes")
    ax[idx].hist(all_phases_flattened[t][shuffling]['grid'], bins=90, color=my_pal['grid'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['granule'], bins=90, color=my_pal['granule'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['mossy'], bins=90, color=my_pal['mossy'], alpha=alpha, density=density)
    if density:
        ax[idx].set_ylim((0,1))
    ax[idx].set_xlim((0, 2*np.pi))
    ax[idx].legend(("Grid", "Granule", "Mossy"))


fig, ax = plt.subplots(2,2)
ax = ax.flatten()

for idx, t in enumerate(tunings):
    ax[idx].set_title(t)
    ax[idx].set_xlabel("Phase (pi)")
    ax[idx].set_ylabel("Probability Density of Spikes")
    ax[idx].hist(all_phases_flattened[t][shuffling]['granule'], bins=90, color=my_pal['granule'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['basket'], bins=90, color=my_pal['basket'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['hipp'], bins=90, color=my_pal['hipp'], alpha=alpha, density=density)
    if density:
        ax[idx].set_ylim((0,1))
    ax[idx].set_xlim((0, 2*np.pi))
    ax[idx].legend(("Granule", "Basket", "HIPP"))
    
fig, ax = plt.subplots(2,2)
ax = ax.flatten()

for idx, t in enumerate(tunings):
    ax[idx].set_title(t)
    ax[idx].set_xlabel("Phase (pi)")
    ax[idx].set_ylabel("Probability Density of Spikes")
    ax[idx].hist(all_phases_flattened[t][shuffling]['grid'], bins=90, color=my_pal['grid'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['granule'], bins=90, color=my_pal['granule'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['mossy'], bins=90, color=my_pal['mossy'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['basket'], bins=90, color=my_pal['basket'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['hipp'], bins=90, color=my_pal['hipp'], alpha=alpha, density=density)
    if density:
        ax[idx].set_ylim((0,1))
    ax[idx].set_xlim((0, 2*np.pi))
    ax[idx].legend(("Grid", "Granule", "Mossy" "Basket", "HIPP"))

fig, ax = plt.subplots(1, 3)
ax = ax.flatten()

for idx, t in enumerate(['no-feedforward', 'no-feedback', 'disinhibited']):
    ax[idx].set_title(t)
    ax[idx].set_xlabel("Phase (pi)")
    ax[idx].set_ylabel("# of spikes")
    ax[idx].hist(all_phases_flattened['full'][shuffling]['granule'], bins=90, color=my_pal['granule'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened[t][shuffling]['granule'], bins=90, color=my_pal['basket'], alpha=alpha, density=density)
    if density:
        ax[idx].set_ylim((0,1))
    ax[idx].set_xlim((0, 2*np.pi))
    ax[idx].legend(("Granule Full", f"Granule {t} adjusted"))


# file directory
dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data', 'multiple_celltypes_non-adjusted')
save_dir = dirname

tunings = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
shufflings = ['shuffled', 'non-shuffled']

all_files = set(['.'.join(x.split('.')[:-1]) for x in os.listdir(results_dir)])

poisson_seeds = range(100, 110)
grid_seeds = range(1, 6)
cell_types = range(4)

all_spikes = {}
for t in tunings:
    all_spikes[t] = {}
    for s in shufflings:
        all_spikes[t][s] = {}
            
for f in all_files:
    curr_file = shelve.open(os.path.join(results_dir, f))
    split = f.split('_')
    grid_seed = int(split[8])
    shuffling = split[12]
    tuning = split[13]
    all_spikes[tuning][shuffling][grid_seed] = dict(curr_file)

all_phases = deepcopy(all_spikes)

# Calculate phases
for t in tunings:
    for s in shufflings:
        for g in grid_seeds:
            for ps in poisson_seeds:
                all_phases[t][s][g]['grid_spikes'][75][ps] = [((spikes % 100) / 100) * (np.pi * 2) for spikes in all_spikes[t][s][g]['grid_spikes'][75][ps]]
                
                all_phases[t][s][g]['all_spikes'][75][ps] = list(all_phases[t][s][g]['all_spikes'][75][ps])
                for ct in cell_types:
                    all_phases[t][s][g]['all_spikes'][75][ps][ct] = [((spikes % 100) / 100) * (np.pi * 2) for spikes in all_spikes[t][s][g]['all_spikes'][75][ps][ct]]

all_phases_flattened = {}
for t in tunings:
    all_phases_flattened[t] = {} 
    for s in shufflings:
        all_phases_flattened[t][s] = {}
        for ct in ['grid', 'granule', 'mossy', 'basket', 'hipp']:
            all_phases_flattened[t][s][ct] = np.array([])

for t in tunings:
    for s in shufflings:
        for g in grid_seeds:
            for ps in poisson_seeds:
                for cell in all_phases[t][s][g]['grid_spikes'][75][ps]:
                    all_phases_flattened[t][s]['grid'] = np.concatenate((all_phases_flattened[t][s]['grid'], cell))
                for ct_idx, ct in enumerate(['granule', 'mossy', 'basket', 'hipp']):
                    for cell in all_phases[t][s][g]['all_spikes'][75][ps][ct_idx]:
                        all_phases_flattened[t][s][ct] = np.concatenate((all_phases_flattened[t][s][ct], cell))

all_phases_flattened_nonadjusted = deepcopy(all_phases_flattened)

fig, ax = plt.subplots(2,2)
ax = ax.flatten()
for idx, t in enumerate(tunings):
    ax[idx].set_title(t)
    ax[idx].set_xlabel("Phase (pi)")
    ax[idx].set_ylabel("# of Spikes")
    ax[idx].hist(all_phases_flattened_nonadjusted [t][shuffling]['granule'], bins=90, color=my_pal['grid'], alpha=alpha, density=density)
    ax[idx].hist(all_phases_flattened_adjusted [t][shuffling]['granule'], bins=90, color=my_pal['granule'], alpha=alpha, density=density)
    if density:
        ax[idx].set_ylim((0,1))
    ax[idx].set_xlim((0, 2*np.pi))
    ax[idx].set_ylim((0, 8000))
    ax[idx].legend(("GCs non-adjusted", "GCs rate adjusted"))

sys.exit()

for i, tuning in enumerate(tunings):
    fname = os.path.join(results_dir, 'pickled', f'fig2_phase-dist_{tuning}.pkl')
    with open(fname, 'rb') as f:
        phase_df = pickle.load(f)
    phase_df = phase_df[phase_df['shuffling']=='non-shuffled']
    ax = axes.flatten()[i]
    fig = sns.histplot(data=phase_df, x='phase (pi)', palette=my_pal,
                       legend=False,
                       kde=True, hue='cell', binwidth=(2*np.pi/360), ax=ax)
    ax.set_title(f'{tuning}')
    ax.set_xlabel('phase ($\pi$)')
    fig.set(ylim=(0, 500000))
sns.despine(fig=f2a)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax.ticklabel_format(style='sci', scilimits=[0, 0])

f2a.savefig(f'{save_dir}figure02_A.svg', dpi=200)
f2a.savefig(f'{save_dir}figure02_A.png', dpi=200)

# =============================================================================
# Figure 2 B
# =============================================================================

f2b, axes = plt.subplots(2,2, figsize=(8*cm, 5*cm), sharey=True, sharex=True)
# sns.set(style='ticks', palette='deep', font='Arial',
#         font_scale=1.2, color_codes=True)
my_pal = {'grid': '#a09573', 'granule': '#127475'}

tunings = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
for i, tuning in enumerate(tunings):
    fname = os.path.join(results_dir, 'pickled', f'fig2_phase-dist_{tuning}.pkl')
    with open(fname, 'rb') as f:
        phase_df = pickle.load(f)
    phase_df = phase_df[phase_df['shuffling']=='shuffled']
    ax = axes.flatten()[i]
    fig = sns.histplot(data=phase_df, x='phase (pi)', palette=my_pal,
                       legend=False,
                       kde=True, hue='cell', binwidth=(2*np.pi/360), ax=ax)
    ax.set_title(f'{tuning}')
    ax.set_xlabel('phase ($\pi$)')
    fig.set(ylim=(0, 500000))
sns.despine(fig=f2b)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
ax.ticklabel_format(style='sci', scilimits=[0,0])
plt.rcParams["svg.fonttype"] = "none"
f2b.savefig(f'{save_dir}figure02_B.svg', dpi=200)
f2b.savefig(f'{save_dir}figure02_B.png', dpi=200)

# =============================================================================
# Figure 02 C
# =============================================================================

grid_pal = {'non-shuffled': '#716969', 'shuffled': '#a09573'}
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
legend_size = 8

fname = os.path.join(results_dir, 'excel', 'mean_firing_rates.xlsx')
all_mean_rates = pd.read_excel(fname, index_col=0)
grid_seeds = list(np.arange(1, 11, 1))
grid_mean = all_mean_rates[all_mean_rates['cell'] == 'grid']
granule_mean = all_mean_rates[all_mean_rates['cell'] == 'granule']

f2c, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 4]},
                               figsize=(8*cm, 5*cm))
sns.boxplot(x='cell', y='mean_rate', hue='shuffling', ax=ax1, zorder=1,
            data=grid_mean, palette=grid_pal, linewidth=0.5, fliersize=1)
loc = 0.4
tune_idx = 0
for grid in grid_seeds:
    ns_data = grid_mean.loc[(grid_mean['shuffling'] == 'non-shuffled')
                            &
                            (grid_mean['grid_seeds'] == grid)]
    s_data = grid_mean.loc[(grid_mean['shuffling'] == 'shuffled')
                           &
                           (grid_mean['grid_seeds'] == grid)]
    ns_data = np.array(ns_data['mean_rate'])[0]
    s_data = np.array(s_data['mean_rate'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[ns_data, s_data], color='k', linewidth=0.2)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[ns_data, s_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None,
           prop={'size': legend_size})
ax1.set_ylabel('Mean rate (Hz)')
# ax1.legend(loc='upper left', title=None)
sns.boxplot(x='cell & tuning', y='mean_rate', hue='shuffling', ax=ax2,
            zorder=1, data=granule_mean, palette=granule_pal, linewidth=0.3,
            fliersize=1)
# sns.stripplot(x='cell & tuning', y='mean_rate', ax=ax2, zorder=10,
#                 hue='shuffling', data=granule_mean, color='black',
#                 jitter=False, dodge=True, linewidth=0.1, size=1.5)
loc = 0.4
tunings = ['full', 'no-feedforward',
           'no-feedback', 'disinhibited']
for grid in grid_seeds:
    tune_idx = 0
    for tuning in tunings:
        ns_data = granule_mean.loc[(granule_mean['shuffling'] =='non-shuffled')
                                   &
                                   (granule_mean['tuning'] == tuning)
                                   &
                                   (granule_mean['grid_seeds'] == grid)]
        s_data = granule_mean.loc[(granule_mean['shuffling'] == 'shuffled')
                                  &
                                  (granule_mean['tuning'] == tuning)
                                  &
                                  (granule_mean['grid_seeds'] == grid)]
        ns_data = np.array(ns_data['mean_rate'])[0]
        s_data = np.array(s_data['mean_rate'])[0]
        sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax2,
                     y=[ns_data, s_data], color='k', linewidth=0.2)
        sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax2,
                        y=[ns_data, s_data], color='k', s=3)
        tune_idx += 1


ax2.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
handles2, labels2 = ax2.get_legend_handles_labels()
# plt.sca(ax2)
# l = ax2.legend(handles2[2:], labels2[2:], bbox_to_anchor=(1.05, 1),
#                borderaxespad=0., title=None)
ax2.legend(handles2[:2], labels2[:2], bbox_to_anchor=(0, 1.5),
           borderaxespad=0., loc='upper left', title=None,
           prop={'size': legend_size})
ax2.set_ylabel('Mean rate (Hz)')
f2c.subplots_adjust(bottom=0.3, wspace=1, top=0.75, left=0.2)
sns.despine(fig=f2c)
_adjust_box_widths(f2c, 0.7)
plt.rcParams["svg.fonttype"] = "none"
# save_dir = '/home/baris/paper/figures/figure02/'
f2c.savefig(f'{save_dir}figure02_C_boxplot.svg', dpi=200)
f2c.savefig(f'{save_dir}figure02_C_boxplot.png', dpi=200)

# =============================================================================
# Figure 02 D nonshuffled
# =============================================================================
plt.close('all')
spacing = 40
pos_peak = [100, 100]
orientation = 30
dur = 5
shuffle = False

grid_rate = grid_model._grid_maker(spacing,
                                   orientation, pos_peak).reshape(200,
                                                                  200, 1)
grid_rates = np.append(grid_rate, grid_rate, axis=2)
spacings = [spacing, spacing]
grid_dist = grid_model._rate2dist(grid_rates,
                                  spacings)[:, :, 0].reshape(200, 200, 1)
trajs = np.array([50])
dist_trajs = grid_model._draw_traj(grid_dist, 1, trajs, dur_ms=5000)
rate_trajs = grid_model._draw_traj(grid_rate, 1, trajs, dur_ms=5000)
rate_trajs, rate_t_arr = grid_model._interp(rate_trajs, 5, new_dt_s=0.002)

grid_overall = grid_model._overall(dist_trajs,
                                   rate_trajs, 240, 0.1, 1, 1, 5,
                                   20, 5)[0, :, 0]
trains, phases, phase_loc = _precession_spikes(grid_overall,
                                               shuffle=shuffle)

# plotting
rate = grid_rate.reshape(200, 200)
f2d, ax = plt.subplots(1, 1, sharex=True, figsize=(5*cm, 5*cm))
im = ax.imshow(phase_loc, aspect='auto',
                 cmap='RdYlBu_r', extent=[0, 100, 720, 0],
                 vmin=0, vmax=66)
ax.set_ylim((0, 720))
ax.set_xlabel('Location (cm)')
ax.set_ylabel('Theta phase (deg)')
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 1080, 360))
cax = f2d.add_axes([0.83, 0.4, 0.04, 0.35])
cbar = f2d.colorbar(im, cax=cax)
cbar.set_label('Hz', labelpad=15, rotation=270)
parameters = ('spacing_center_orientation_' +
              f'{spacing}_{pos_peak[0]}_{orientation}')
f2d.subplots_adjust(bottom=0.3, wspace=1, left=0.2, right=0.8)
plt.rcParams["svg.fonttype"] = "none"
f2d.savefig(f'{save_dir}figure02_D_nonshuffled.svg', dpi=200)
f2d.savefig(f'{save_dir}figure02_D_nonshuffled.png', dpi=200)

# =============================================================================
# Figure 02 D shuffled
# =============================================================================
shuffle = True
spacing = 40
pos_peak = [100, 100]
orientation = 30
dur = 5
grid_rate = grid_model._grid_maker(spacing,
                                   orientation, pos_peak).reshape(200,
                                                                  200, 1)
grid_rates = np.append(grid_rate, grid_rate, axis=2)
spacings = [spacing, spacing]
grid_dist = grid_model._rate2dist(grid_rates,
                                  spacings)[:, :, 0].reshape(200, 200, 1)
trajs = np.array([50])
dist_trajs = grid_model._draw_traj(grid_dist, 1, trajs, dur_ms=5000)
rate_trajs = grid_model._draw_traj(grid_rate, 1, trajs, dur_ms=5000)
rate_trajs, rate_t_arr = grid_model._interp(rate_trajs, 5, new_dt_s=0.002)

grid_overall = grid_model._overall(dist_trajs,
                                   rate_trajs, 240, 0.1, 1, 1, 5,
                                   20, 5)[0, :, 0]
trains, phases, phase_loc = _precession_spikes(grid_overall,
                                               shuffle=shuffle)
# plotting

rate = grid_rate.reshape(200, 200)
f2ds, ax = plt.subplots(1, 1, sharex=True, figsize=(5*cm, 5*cm))
im2 = ax.imshow(phase_loc, aspect='auto',
                 cmap='RdYlBu_r', extent=[0, 100, 720, 0],
                 vmin=0, vmax=66)
ax.set_ylim((0, 720))
ax.set_xlabel('Location (cm)')
ax.set_ylabel('Theta phase (deg)')
ax.set_xticks(np.arange(0, 120, 20))
ax.set_yticks(np.arange(0, 1080, 360))
cax = f2ds.add_axes([0.83, 0.4, 0.04, 0.35])
cbar = f2ds.colorbar(im2, cax=cax)
cbar.set_label('Hz', labelpad=15, rotation=270)
# plt.tight_layout()
parameters = ('spacing_center_orientation_' +
              f'{spacing}_{pos_peak[0]}_{orientation}')
f2ds.subplots_adjust(bottom=0.3, wspace=1, left=0.2, right=0.8)
plt.rcParams["svg.fonttype"] = "none"
f2ds.savefig(f'{save_dir}figure02_D_shuffled.svg', dpi=200)
f2ds.savefig(f'{save_dir}figure02_D_shuffled.png', dpi=200)


# =============================================================================
# Figure 2E and 2F
# =============================================================================
# pre - plotting
color_list_1 = ["0a2d27","13594e","1d8676","26b29d","30dfc4","59e5d0","83ecdc"]
color_list_2 = ["572800","ab5100","ff7900","ff9637","ffb570"]
my_cmap = _make_cmap(color_list_1)
my_cmap_2 = _make_cmap(color_list_2)

# load data, codes

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
n_samples = 20
grid_seeds = np.arange(1, 11, 1)
tuning = 'full'
fname = os.path.join(results_dir, 'pickled', 'neural_codes_full.pkl')
with open(fname, 'rb') as f:
    all_codes = pickle.load(f)

# 75 vs all in all time bins
# calculate pearson R
r_data = []
for grid_seed in all_codes:
    shuffling = 'shuffled'
    for cell in all_codes[grid_seed][shuffling]:
        for code in all_codes[grid_seed][shuffling][cell]:
            for traj_idx in range(len(trajectories)):
                r_input_code = np.array(all_codes[grid_seed]
                                        [shuffling][cell][code])
                f = int(r_input_code.shape[0]/8000)
                baseline_traj = r_input_code[:, 0, 0]
                for poisson in range(r_input_code.shape[1]):
                    compared_traj = r_input_code[:, poisson, traj_idx]
                    pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                    spearman_r = spearmanr(baseline_traj, compared_traj)[0]
                    traj = trajectories[traj_idx]
                    idx = 75 - traj
                    comp_trajectories = str(75)+'_'+str(traj)
                    r_data_sing = [idx, pearson_r, spearman_r, poisson,
                                   comp_trajectories, grid_seed, shuffling,
                                   cell, code]
                    r_data.append(copy.deepcopy(r_data_sing))


# preprocessing

df = pd.DataFrame(r_data,
                  columns=['distance', 'pearson_r', 'spearman_r',
                           'poisson_seed', 'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')

shuffled_grid_rate = (df[(df['cell_type'] == 'grid') &
                            (df['code_type'] == 'rate') &
                            (df['shuffling'] == 'shuffled')]
                         [['distance', 'grid_seed',
                           'pearson_r']].reset_index(drop=True))
shuffled_granule_rate = (df[(df['cell_type'] == 'granule') &
                               (df['code_type'] == 'rate') &
                               (df['shuffling'] == 'shuffled')]
                            ['pearson_r'].reset_index(drop=True))
shuffled_grid_phase = (df[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'phase') &
                             (df['shuffling'] == 'shuffled')]
                          ['pearson_r'].reset_index(drop=True))
shuffled_granule_phase = (df[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'phase') &
                                (df['shuffling'] == 'shuffled')]
                             ['pearson_r'].reset_index(drop=True))

pearson_r = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    shuffled_grid_phase, shuffled_granule_phase], axis=1)
pearson_r.columns = ['distance', 'grid_seed',
                     's_grid_rate', 's_granule_rate',
                     's_grid_phase', 's_granule_phase']


f2e, ax1 = plt.subplots(figsize=(4*cm, 4*cm))
f2f, ax3 = plt.subplots(figsize=(4*cm, 4*cm))

hue = list(pearson_r['distance'])
# rate
sns.scatterplot(ax=ax1,
                data=pearson_r, x="s_grid_rate", y="s_granule_rate",
                hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap, s=1,
                linewidth=0.1, alpha=0.8)
ax1.get_legend().remove()
ax1.plot(np.arange(-0.2, 1.1, 0.1), np.arange(-0.2, 1.1, 0.1),
         color='#2c423f', alpha=0.4, linewidth=1)
ax1.set_xlim(-0.1, 0.5)
ax1.set_ylim(-0.1, 0.5)

s_rate = stats.binned_statistic(pearson_r['s_grid_rate'],
                                 list((pearson_r['s_grid_rate'],
                                       pearson_r['s_granule_rate'])),
                                 'mean',
                                 bins=[0, 0.1, 0.2, 0.3, 0.4,
                                       0.5, 0.6, 0.7, 0.8])
ax1.plot(s_rate[0][0], s_rate[0][1], 'k',
         linestyle=(0, (6, 2)), linewidth=2)
ax1.set_ylabel('$R_{out}$')
ax1.set_xlabel('$R_{in}$')
norm = mpl.colors.SymLogNorm(vmin=0.5, vmax=65, linthresh=0.1)
# f2e.colorbar(matplotlib.cm.ScalarMappable(cmap=my_cmap, norm=norm), ax=ax1)

# phase
sns.scatterplot(ax=ax3,
                data=pearson_r, x="s_grid_phase", y="s_granule_phase",
                hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap_2, s=1,
                linewidth=0.1, alpha=0.8)
ax3.get_legend().remove()
ax3.plot(np.arange(-0.2, 1.1, 0.1), np.arange(-0.2, 1.1, 0.1),
         color='#2c423f', alpha=0.4, linewidth=1)
ax3.set_xlim(-0.1, 0.5)
ax3.set_ylim(-0.1, 0.5)

s_phase = stats.binned_statistic(pearson_r['s_grid_phase'],
                                  list((pearson_r['s_grid_phase'],
                                        pearson_r['s_granule_phase'])),
                                  'mean', bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax3.plot(s_phase[0][0], s_phase[0][1], 'k',
         linestyle=(0, (6, 2)), linewidth=2)
ax3.set_ylabel('$R_{out}$')
ax3.set_xlabel('$R_{in}$')
norm = mpl.colors.SymLogNorm(vmin=0.5, vmax=65, linthresh=0.1)
# f2f.colorbar(matplotlib.cm.ScalarMappable(cmap=my_cmap_2, norm=norm), ax=ax3)
# =============================================================================
# mean delta R
# =============================================================================
ax2 = inset_axes(ax1,  "20%", "50%", loc="upper right", borderpad=0)
ax4 = inset_axes(ax3,  "20%", "50%", loc="upper right", borderpad=0)
grid_seeds = np.arange(1, 11, 1)
delta_s_rate = []
delta_s_phase = []
for seed in grid_seeds:
    grid_1 = pearson_r.loc[(pearson_r['grid_seed'] == seed)]
    s_rate = stats.binned_statistic(grid_1['s_grid_rate'],
                                     list((grid_1['s_grid_rate'],
                                           grid_1['s_granule_rate'])),
                                     'mean',
                                     bins=np.arange(0, 1, 0.1))
    s_phase = stats.binned_statistic(grid_1['s_grid_phase'],
                                      list((grid_1['s_grid_phase'],
                                            grid_1['s_granule_phase'])),
                                      'mean',
                                      bins=np.arange(0, 0.6, 0.1))

    delta_s_rate.append(np.mean((s_rate[0][0] - s_rate[0][1])
                                 [s_rate[0][0] == s_rate[0][0]]))
    delta_s_phase.append(np.mean((s_phase[0][0] - s_phase[0][1])
                                  [s_phase[0][0] == s_phase[0][0]]))


s_deltaR = np.concatenate((delta_s_rate, delta_s_phase))
shuffling = 20*['shuffled']
code = 10*['rate']+10*['phase']
seeds = 2*list(range(1, 11))
s_delta = np.stack((s_deltaR, shuffling, code, seeds), axis=1)
s_deltaR = pd.DataFrame(s_delta, columns=['mean deltaR',
                                            'shuffling',
                                            'code',
                                            'grid_seed'])
s_deltaR['mean deltaR'] = s_deltaR['mean deltaR'].astype('float')
s_deltaR['grid_seed'] = s_deltaR['grid_seed'].astype('float')

# plotting mean deltaR
my_pal1 = {'#26b29d'}
my_pal2 = {'#ff7900'}

sns.boxplot(x='code', y='mean deltaR', ax=ax2, linewidth=0.5, fliersize=0.5,
            data=s_deltaR[s_deltaR['code'] == 'rate'], palette=my_pal1, 
            width=0.5)
sns.scatterplot(x='code', y='mean deltaR', ax=ax2, s=3,
            data=s_deltaR[s_deltaR['code'] == 'rate'], color='black')

sns.boxplot(x='code', y='mean deltaR', ax=ax4, linewidth=0.5, fliersize=0.5,
            data=s_deltaR[s_deltaR['code'] == 'phase'], palette=my_pal2,
            width=0.5)
sns.scatterplot(x='code', y='mean deltaR', ax=ax4, s=3,
            data=s_deltaR[s_deltaR['code'] == 'phase'], color='black')

ax2.set_ylabel('mean $\u0394R$')
ax2.set(xticklabels=[])  # remove the tick labels
ax4.set_ylabel('mean $\u0394R$')
ax4.set_xlabel('')
ax2.yaxis.set_label_position("right")
ax4.yaxis.set_label_position("right")
sns.despine(ax=ax1)
sns.despine(ax=ax2, right=False, left=True)
sns.despine(ax=ax3)
sns.despine(ax=ax4, right=False, left=True)
f2f.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.9, wspace=1)
f2e.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.9, wspace=1)
_adjust_box_widths(f2e, 0.7)
_adjust_box_widths(f2f, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f2e.savefig(f'{save_dir}figure02_E.svg', dpi=200)
f2e.savefig(f'{save_dir}figure02_E.png', dpi=200)
plt.rcParams["svg.fonttype"] = "none"
f2f.savefig(f'{save_dir}figure02_F.svg', dpi=200)
f2f.savefig(f'{save_dir}figure02_F.png', dpi=200)



# =============================================================================
# Figure 2G
# =============================================================================
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}

fname = os.path.join(results_dir, 'excel', 'mean_deltaR_2000ms_75vsall.xlsx')
all_mean_deltaR = pd.read_excel(fname, index_col=0)
f2g, ax = plt.subplots(1,1, figsize=(4*cm, 4*cm))

sns.boxplot(x='tuning', y='mean deltaR', hue='shuffling', ax=ax, zorder=1,
            data=all_mean_deltaR, linewidth=0.5, fliersize=1,
            palette=granule_pal)
loc = 0.4
tunings = ['full', 'no-feedforward',
           'no-feedback', 'disinhibited']
for grid in grid_seeds:
    tune_idx = 0
    for tuning in tunings:
        ns_data = all_mean_deltaR.loc[(all_mean_deltaR['shuffling']=='non-shuffled')
                                      &
                                      (all_mean_deltaR['tuning']==tuning)
                                      &
                                      (all_mean_deltaR['grid_seeds']==grid)]
        s_data = all_mean_deltaR.loc[(all_mean_deltaR['shuffling']=='shuffled')
                                      &
                                      (all_mean_deltaR['tuning']==tuning)
                                      &
                                      (all_mean_deltaR['grid_seeds']==grid)]
        if (np.array(ns_data['mean deltaR']).size > 0 and
            np.array(s_data['mean deltaR']).size > 0):
            ns_info = np.array(ns_data['mean deltaR'])[0]
            s_info = np.array(s_data['mean deltaR'])[0]   
            sns.lineplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax,
                         y = [ns_info, s_info], color='k', linewidth = 0.2)
            sns.scatterplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax,
                          y = [ns_info, s_info], color='k', s = 3)
        tune_idx+=1

ax.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.4),
          loc='upper left', borderaxespad=0., title=None, prop={'size': 8})
ax.set_ylabel('Mean Delta R')
f2g.subplots_adjust(left=0.35, bottom=0.3, right=0.9, top=0.8)
sns.despine(fig=f2g)
_adjust_box_widths(f2g, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f2g.savefig(f'{save_dir}figure02_G.svg', dpi=200)
f2g.savefig(f'{save_dir}figure02_G.png', dpi=200)

# =============================================================================
# Figure 2H
# =============================================================================

'''Skaggs Information - Average of Population
cells firing less than 8 spikes are filtered out
10 grid seeds, 20 poisson seeds aggregated,
spatial bin = 5cm'''

grid_pal = {'non-shuffled': '#716969', 'shuffled': '#a09573'}
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
legend_size = 8

fname = os.path.join(results_dir, 'excel', 'figure_2I_skaggs_non-adjusted.xlsx')
df_skaggs = pd.read_excel(fname, index_col=0)
grid_seeds = list(np.arange(1,11,1))
grid_info = df_skaggs[df_skaggs['cell']=='full grid']
grid_info['grid_seed'] = 2*grid_seeds
granule_info = df_skaggs[df_skaggs['cell']!='full grid']
granule_info['grid_seed'] = 8*grid_seeds

f2h, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]},
                               figsize=(8*cm, 5*cm))

sns.boxplot(x='cell', y='info', hue='shuffling', ax=ax1, zorder=1,
            data=grid_info, linewidth=0.4, fliersize=1, palette=grid_pal)
loc = 0.4
tune_idx = 0
tunings = ['full granule', 'no-feedforward granule',
           'no-feedback granule', 'disinhibited granule']
for grid in grid_seeds:
    ns_data = grid_info.loc[(grid_info['shuffling']=='non-shuffled')
                                  &
                                  (grid_info['grid_seed']==grid)]
    s_data = grid_info.loc[(grid_info['shuffling']=='shuffled')
                                  &
                                  (grid_info['grid_seed']==grid)]
    if (np.array(ns_data['info']).size > 0 and
        np.array(s_data['info']).size > 0):
        ns_info = np.array(ns_data['info'])[0]
        s_info = np.array(s_data['info'])[0]   
        sns.lineplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                     y = [ns_info, s_info], color='k', linewidth = 0.2)
        sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                        y=[ns_info, s_info], color='k', s=3)
ax1.set_xticklabels(['grid'], rotation = 60)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1.5, 1.5),
           loc='upper left', borderaxespad=0., title=None,
           prop={'size': legend_size})
ax1.set_ylabel('info (bits/AP)')

sns.boxplot(x='cell', y='info', hue='shuffling', ax=ax2, zorder=1,
            data=granule_info, linewidth=0.4, fliersize=1, palette=granule_pal)
loc = 0.4
tunings = ['full granule', 'no-feedforward granule',
           'no-feedback granule', 'disinhibited granule']
for grid in grid_seeds:
    tune_idx = 0
    for tuning in tunings:
        ns_data = granule_info.loc[(granule_info['shuffling']=='non-shuffled')
                                      &
                                      (granule_info['cell']==tuning)
                                      &
                                      (granule_info['grid_seed']==grid)]
        s_data = granule_info.loc[(granule_info['shuffling']=='shuffled')
                                      &
                                      (granule_info['cell']==tuning)
                                      &
                                      (granule_info['grid_seed']==grid)]
        if (np.array(ns_data['info']).size > 0 and
            np.array(s_data['info']).size > 0):
            ns_info = np.array(ns_data['info'])[0]
            s_info = np.array(s_data['info'])[0]   
            sns.lineplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax2,
                         y = [ns_info, s_info], color='k', linewidth = 0.2)
            sns.scatterplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax2,
                            y=[ns_info, s_info], color='k', s=3)
        tune_idx+=1

ax2.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.5),
           loc='upper left', borderaxespad=0., title=None,
           prop={'size': legend_size})
ax2.set_ylabel('info (bits/AP)')

f2h.subplots_adjust(bottom=0.3, left=0.21, wspace=1, top=0.75)
sns.despine(fig=f2h)
_adjust_box_widths(f2h, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f2h.savefig(f'{save_dir}figure02_H.svg', dpi=200)
f2h.savefig(f'{save_dir}figure02_H.png', dpi=200)


# =============================================================================
# Figure 2I
# =============================================================================

# Info in differently tuned networks with mean between 0.2-0.3
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
legend_size = 8

fname = results_dir + 'excel/fig2j_adjusted-filtered-info.xlsx'
fname = os.path.join(results_dir, 'excel', 'fig2j_adjusted-filtered-info.xlsx')
dfa = pd.read_excel(fname, index_col=0)
realistic_means = copy.deepcopy(dfa)
f2i, ax = plt.subplots(1,1, figsize=(5*cm, 5*cm))

sns.boxplot(x='tuning', y='info (bits/AP)', hue='shuffling', ax=ax, zorder=1,
            data=realistic_means, linewidth=0.5, fliersize=1,
            palette=granule_pal)
# sns.stripplot(x='tuning', y='info (bits/AP)', ax=ax, zorder=2, hue='shuffling',
#                 data=realistic_means, color='black', jitter=False, dodge=True,
#                 linewidth=0.1, size=1)
loc = 0.4
tunings = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
for grid in grid_seeds:
    tune_idx = 0
    for tuning in tunings:
        ns_data = realistic_means.loc[(realistic_means['shuffling']=='non-shuffled')
                                      &
                                      (realistic_means['tuning']==tuning)
                                      &
                                      (realistic_means['grid_seed']==grid)]
        s_data = realistic_means.loc[(realistic_means['shuffling']=='shuffled')
                                      &
                                      (realistic_means['tuning']==tuning)
                                      &
                                      (realistic_means['grid_seed']==grid)]
        
        
        if (np.array(ns_data['info (bits/AP)']).size > 0 and
            np.array(s_data['info (bits/AP)']).size > 0):
            ns_info = np.array(ns_data['info (bits/AP)'])[0]
            s_info = np.array(s_data['info (bits/AP)'])[0]   
            sns.lineplot(x= [-loc/2+tune_idx, loc/2+tune_idx],
                         y = [ns_info, s_info], color='k', linewidth = 0.2)
            sns.scatterplot(x= [-loc/2+tune_idx, loc/2+tune_idx],
                          y = [ns_info, s_info], color='k', s = 3)
        tune_idx+=1
ax.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.5),
          borderaxespad=0., loc='upper left', title=None,
          prop={'size': legend_size})
ax.set_ylabel('info (bits/AP)')
f2i.subplots_adjust(bottom=0.3, left=0.21, top=0.75)
sns.despine(fig=f2i)
_adjust_box_widths(f2i, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f2i.savefig(f'{save_dir}figure02_I.svg', dpi=200)
f2i.savefig(f'{save_dir}figure02_I.png', dpi=200)


# =============================================================================
# Figure 2J
# =============================================================================

scaled_pal = {'nonshuffled/shuffled': '#267282'}

fname = results_dir + 'excel/fig2j_adjusted-filtered-info.xlsx'
fname = os.path.join(results_dir, 'excel', 'fig2j_adjusted-filtered-info.xlsx')
dfa = pd.read_excel(fname, index_col=0)
dfa2 = copy.deepcopy(dfa)
dfa2_ns = dfa2[dfa2['shuffling']=='non-shuffled'].reset_index(drop=True)
dfa2_s = dfa2[dfa2['shuffling']=='shuffled'].reset_index(drop=True)
dfa2_div = dfa2_ns['info (bits/AP)']/dfa2_s['info (bits/AP)']
dfa3 = copy.deepcopy(dfa2_ns)
dfa3['info (bits/AP)'] = dfa2_div
dfa3.rename(columns={'info (bits/AP)':'scaled value (non-shuffled/shuffled)'},
            inplace=True)
dfa3['shuffling'] = 'nonshuffled/shuffled'

# output_name = '/home/baris/results/excel/figure2I_adjusted_scaled_skaggs-info.xlsx'

# with pd.ExcelWriter(output_name) as writer:
#     dfa3.to_excel(writer, sheet_name='scaled_info')
    
plt.close('all')
f2j, ax = plt.subplots(1,1, figsize=(5*cm, 4*cm))
sns.boxplot(x='tuning', y='scaled value (non-shuffled/shuffled)', ax=ax,
            data=dfa3, palette=scaled_pal, zorder=1, linewidth=0.5, fliersize=1,
            hue='shuffling')
# sns.scatterplot(x='tuning', y='scaled value (non-shuffled/shuffled)', ax=ax,
#                 data=dfa3, color='black', alpha=0.8)
sns.stripplot(x='tuning', y='scaled value (non-shuffled/shuffled)', ax=ax,
              zorder=2, data=dfa3,
              color='black', jitter=False, dodge=True, linewidth=0.1, size=2)
sns.lineplot(x='tuning', y='scaled value (non-shuffled/shuffled)', ax=ax,
             zorder=1, legend=False,
             data=dfa3, linewidth=1, hue='grid_seed',
             palette= sns.color_palette(10*['black']), alpha=0.3, ci=None)
ax.set_ylim([1,1.4])
ax.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
ax.set_title('non-shuffled/shuffled')
ax.set_ylabel('scaled value')
ax.get_legend().remove()
f2j.subplots_adjust(left=0.3, bottom=0.3)
sns.despine(fig=f2j)
_adjust_box_widths(f2j, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f2j.savefig(f'{save_dir}figure02_J.svg', dpi=200)
f2j.savefig(f'{save_dir}figure02_J.png', dpi=200)









# =============================================================================
# Figure 1C barplot
# =============================================================================
fname = results_dir + 'excel/mean_firing_rates.xlsx'
fname = os.path.join(results_dir, 'excel', 'mean_firing_rates.xlsx')
all_mean_rates = pd.read_excel(fname, index_col=0)

grid_mean = all_mean_rates[all_mean_rates['cell']=='grid']
granule_mean = all_mean_rates[all_mean_rates['cell']=='granule']

f2c, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]},
                               figsize=(8*cm, 5*cm))
grid_pal = {'non-shuffled': '#716969', 'shuffled': '#a09573'}
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
sns.barplot(x='cell', y='mean_rate', hue='shuffling', ax=ax1, zorder=1,
            data=grid_mean, palette=grid_pal, ci='sd', capsize=.3,
            errwidth=1)
sns.stripplot(x='cell', y='mean_rate', ax=ax1, zorder=10, hue='shuffling',
                data=grid_mean, color='black', jitter=False, dodge=True,
                linewidth=0.1, size=2.5)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 60)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[2:], labels[2:], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None, prop={'size': 8})
ax1.set_ylabel('Mean rate (Hz)')
# ax1.legend(loc='upper left', title=None)
sns.barplot(x='cell & tuning', y='mean_rate', hue='shuffling', ax=ax2,
            zorder=1, data=granule_mean, palette=granule_pal,
            ci='sd', capsize=.3, errwidth=1)
sns.stripplot(x='cell & tuning', y='mean_rate', ax=ax2, zorder=10,
                hue='shuffling', data=granule_mean, color='black',
                jitter=False, dodge=True, linewidth=0.1, size=2.5)
ax2.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles2[2:], labels2[2:], bbox_to_anchor=(0, 1.5),
           borderaxespad=0., loc='upper left', title=None, prop={'size': 8})
ax2.set_ylabel('Mean rate (Hz)')
f2c.subplots_adjust(bottom=0.3, wspace=1, top=0.75, left=0.2)
sns.despine(fig=f2c)
plt.rcParams["svg.fonttype"] = "none"
f2c.savefig(f'{save_dir}figure02_C_barplot.svg', dpi=200)
f2c.savefig(f'{save_dir}figure02_C_barplot.png', dpi=200)


