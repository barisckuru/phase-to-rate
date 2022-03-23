#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:44:28 2022

@author: baris
"""

import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import grid_model
from figure_functions import _make_cmap, _precession_spikes, _adjust_box_widths
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import SymLogNorm
import matplotlib.font_manager
from scipy.stats import pearsonr,  spearmanr
from scipy import stats
import copy


# file directory
results_dir = '/home/baris/results/'
save_dir = '/home/baris/paper/figures/figure03/'

# plotting settings
sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)
plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=8) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
plt.rc('ytick', labelsize=8) #fontsize of the y tick labels
plt.rc('legend', fontsize=8) #fontsize of the legend
cm=1/2.54



# =============================================================================
# Figure 3 C
# =============================================================================
plt.close('all')

grid_phase_pal = {'non-shuffled': '#8f1010', 'shuffled': '#ba4104'}

grid_rate_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}

full_dir = 'pickled/75-15_full_perceptron_speed.pkl'
fname = results_dir + full_dir
with open(fname, 'rb') as f:
    full_load = pickle.load(f)
    
    
f3c, (ax1, ax2) = plt.subplots(2, 1, figsize=(6*cm, 5*cm), sharex=True)
    
full_perceptron = pd.DataFrame(full_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
full_grid_rate = full_perceptron.loc[(full_perceptron['code']=='rate') &
                                     (full_perceptron['cell']=='grid')]
full_grid_phase = full_perceptron.loc[(full_perceptron['code']=='phase') &
                                      (full_perceptron['cell']=='grid')]

sns.lineplot(data=full_grid_rate, x='distance', y='speed', hue='shuffling',
             palette=grid_rate_pal, ci='sd', ax = ax1, linewidth=1.2)
sns.lineplot(data=full_grid_phase, x='distance', y='speed', hue='shuffling',
             palette=grid_phase_pal, ci='sd', ax=ax2, linewidth=1.2)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1[::-1], labels1[::-1],
           bbox_to_anchor=(1, 1.05), title=None, handlelength=1,
           borderaxespad=0., frameon=False)
ax2.legend(handles2[::-1], labels2[::-1],
           bbox_to_anchor=(1, 1.05), frameon=False, handlelength=1,
           borderaxespad=0., title=None)

ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
f3c.subplots_adjust(bottom=0.3, hspace=0.5, top=0.9, left=0.25, right=0.6)
plt.rcParams["svg.fonttype"] = "none"
f3c.savefig(f'{save_dir}figure03_C.svg', dpi=200)
f3c.savefig(f'{save_dir}figure03_C.png', dpi=200)



# =============================================================================
# Figure 3 D
# =============================================================================
plt.close('all')

grid_phase_pal = {'non-shuffled': '#8f1010', 'shuffled': '#ba4104'}

grid_rate_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}

full_dir = 'pickled/75-15_full_perceptron_speed.pkl'
fname = results_dir + full_dir
with open(fname, 'rb') as f:
    full_load = pickle.load(f)
    
    
f3d, (ax1, ax2) = plt.subplots(2, 1, figsize=(5*cm, 5*cm), sharex=True)
    
full_perceptron = pd.DataFrame(full_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
full_grid_rate = full_perceptron.loc[(full_perceptron['code']=='rate') &
                                     (full_perceptron['cell']=='granule')]
full_grid_phase = full_perceptron.loc[(full_perceptron['code']=='phase') &
                                      (full_perceptron['cell']=='granule')]

sns.lineplot(data=full_grid_rate, x='distance', y='speed', hue='shuffling',
             palette=grid_rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2)
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=full_grid_phase, x='distance', y='speed', hue='shuffling',
             palette=grid_phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2)
f3d.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
plt.rcParams["svg.fonttype"] = "none"
f3d.savefig(f'{save_dir}figure03_D.svg', dpi=200)
f3d.savefig(f'{save_dir}figure03_D.png', dpi=200)

# =============================================================================
# Figure 3 E
# =============================================================================
plt.close('all')

grid_phase_pal = {'non-shuffled': '#8f1010', 'shuffled': '#ba4104'}

grid_rate_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}

noff_dir = 'pickled/75-15_no-feedforward_perceptron_speed_polar_inc.pkl'
fname = results_dir + noff_dir
with open(fname, 'rb') as f:
    noff_load = pickle.load(f)
    
    
f3e, (ax1, ax2) = plt.subplots(2, 1, figsize=(5*cm, 5*cm), sharex=True)
    
noff_perceptron = pd.DataFrame(noff_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
noff_grid_rate = noff_perceptron.loc[(noff_perceptron['code']=='rate') &
                                     (noff_perceptron['cell']=='granule')]
noff_grid_phase = noff_perceptron.loc[(noff_perceptron['code']=='phase') &
                                      (noff_perceptron['cell']=='granule')]

sns.lineplot(data=noff_grid_rate, x='distance', y='speed', hue='shuffling',
             palette=grid_rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2)
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=noff_grid_phase, x='distance', y='speed', hue='shuffling',
             palette=grid_phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2)
f3e.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
plt.rcParams["svg.fonttype"] = "none"
f3e.savefig(f'{save_dir}figure03_E.svg', dpi=200)
f3e.savefig(f'{save_dir}figure03_E.png', dpi=200)


# =============================================================================
# Figure 3 F
# =============================================================================
plt.close('all')

grid_phase_pal = {'non-shuffled': '#8f1010', 'shuffled': '#ba4104'}

grid_rate_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}

nofb_dir = 'pickled/75-15_no-feedback_perceptron_speed_polar_inc.pkl'
fname = results_dir + nofb_dir
with open(fname, 'rb') as f:
    nofb_load = pickle.load(f)
    
    
f3f, (ax1, ax2) = plt.subplots(2, 1, figsize=(5*cm, 5*cm), sharex=True)
    
nofb_perceptron = pd.DataFrame(nofb_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
nofb_grid_rate = nofb_perceptron.loc[(nofb_perceptron['code']=='rate') &
                                     (nofb_perceptron['cell']=='granule')]
nofb_grid_phase = nofb_perceptron.loc[(nofb_perceptron['code']=='phase') &
                                      (nofb_perceptron['cell']=='granule')]

sns.lineplot(data=nofb_grid_rate, x='distance', y='speed', hue='shuffling',
             palette=grid_rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2)
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=nofb_grid_phase, x='distance', y='speed', hue='shuffling',
             palette=grid_phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2)
f3f.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
plt.rcParams["svg.fonttype"] = "none"
f3f.savefig(f'{save_dir}figure03_F.svg', dpi=200)
f3f.savefig(f'{save_dir}figure03_F.png', dpi=200)


# =============================================================================
# Figure 3 G
# =============================================================================
plt.close('all')

grid_phase_pal = {'non-shuffled': '#8f1010', 'shuffled': '#ba4104'}

grid_rate_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}

disinh_dir = 'pickled/75-15_disinhibited_perceptron_speed_polar_inc.pkl'
fname = results_dir + disinh_dir
with open(fname, 'rb') as f:
    disinh_load = pickle.load(f)
    
    
f3g, (ax1, ax2) = plt.subplots(2, 1, figsize=(5*cm, 5*cm), sharex=True)
    
disinh_perceptron = pd.DataFrame(disinh_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
disinh_grid_rate = disinh_perceptron.loc[(disinh_perceptron['code']=='rate') &
                                     (disinh_perceptron['cell']=='granule')]
disinh_grid_phase = disinh_perceptron.loc[(disinh_perceptron['code']=='phase') &
                                      (disinh_perceptron['cell']=='granule')]

sns.lineplot(data=disinh_grid_rate, x='distance', y='speed', hue='shuffling',
             palette=grid_rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2)
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=disinh_grid_phase, x='distance', y='speed', hue='shuffling',
             palette=grid_phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2)
f3g.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
plt.rcParams["svg.fonttype"] = "none"
f3g.savefig(f'{save_dir}figure03_G.svg', dpi=200)
f3g.savefig(f'{save_dir}figure03_G.png', dpi=200)
