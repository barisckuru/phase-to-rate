#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:13:50 2022

@author: baris
"""

import pickle
import numpy as np
import seaborn as sns
import pandas as pd
from phase_to_rate.figure_functions import (_make_cmap, _precession_spikes,
                              _adjust_box_widths, f5_load_data)
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import SymLogNorm
import matplotlib.font_manager
from scipy.stats import pearsonr,  spearmanr
from scipy import stats
import copy
import os
import pickle


dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')
save_dir = dirname

# plotting settings
sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)
plt.rc('font', size=8) #controls default text size
plt.rc('axes', titlesize=8) #fontsize of the title
plt.rc('axes', labelsize=8) #fontsize of the x and y labels
plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
plt.rc('ytick', labelsize=8) #fontsize of the y tick labels
plt.rc('legend', fontsize=8) #fontsize of the legend
cm=1/2.54

# =============================================================================
# load data for B, C and J     ## this should be B,C, F, 
# =============================================================================
fname = 'figure05_B-C-J_condition_dict.pkl'
curr_dir = os.path.join(results_dir, 'pickled')
rates_df, weights_df = f5_load_data(fname, curr_dir)



# =============================================================================
# Figure 5B - barplot
# =============================================================================

granule_pal = {'full': '#09316c', 'noFB': '#2280bf'}
granule_rates_df = rates_df.loc[rates_df['cell']=='granule']
f5b, ax1 = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm))
sns.boxplot(x='tuning', y='mean rate', ax=ax1, zorder=1,
            data=granule_rates_df,
            palette=granule_pal, linewidth=0.5, fliersize=1)
loc = 1
tune_idx = 0.5

grid_seeds = range(1, 31)

for grid in grid_seeds:
    full_data = granule_rates_df.loc[(granule_rates_df['tuning'] == 'full')
                                     &
                                     (granule_rates_df['grid seed'] == grid)]
    noFB_data = granule_rates_df.loc[(granule_rates_df['tuning'] == 'noFB')
                                                                          &
                                     (granule_rates_df['grid seed'] == grid)]

    full_data = np.array(full_data['mean rate'])[0]
    noFB_data = np.array(noFB_data['mean rate'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[full_data, noFB_data], color='k', linewidth=0.1)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[full_data, noFB_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('Mean rate (Hz)')
f5b.subplots_adjust(bottom=0.3, left=0.5, top=0.9)
sns.despine(fig=f5b)
_adjust_box_widths(f5b, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f5b.savefig(f'{save_dir}figure05_B_bar.svg', dpi=200)
f5b.savefig(f'{save_dir}figure05_B_bar.png', dpi=200)


# =============================================================================
# Figure 05C - barplots
# =============================================================================

ca3_pal = {'full': '#09422d', 'noFB': '#66a253'}
ca3_rates_df = rates_df.loc[rates_df['cell']=='ca3']
f5c, ax1 = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm))
sns.boxplot(x='tuning', y='mean rate', ax=ax1, zorder=1,
            data=ca3_rates_df,
            palette=ca3_pal, linewidth=0.5, fliersize=1)
loc = 1
tune_idx = 0.5
for grid in grid_seeds:
    full_data = ca3_rates_df.loc[(ca3_rates_df['tuning'] == 'full')
                                     &
                                     (ca3_rates_df['grid seed'] == grid)]
    noFB_data = ca3_rates_df.loc[(ca3_rates_df['tuning'] == 'noFB')
                                                                          &
                                     (ca3_rates_df['grid seed'] == grid)]

    full_data = np.array(full_data['mean rate'])[0]
    noFB_data = np.array(noFB_data['mean rate'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[full_data, noFB_data], color='k', linewidth=0.05)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[full_data, noFB_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('Mean rate (Hz)')
f5c.subplots_adjust(bottom=0.3, left=0.5, top=0.9)
sns.despine(fig=f5c)
_adjust_box_widths(f5c, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f5c.savefig(f'{save_dir}figure05_C_ca3_bar.svg', dpi=200)
f5c.savefig(f'{save_dir}figure05_C_ca3_bar.png', dpi=200)


# interneuron
# plt.close('all')
intn_pal = {'full': '#495057', 'noFB': '#6c757d'}
intn_rates_df = rates_df.loc[rates_df['cell']=='interneuron']
f5c2, ax1 = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm))
sns.boxplot(x='tuning', y='mean rate', ax=ax1, zorder=1,
            data=intn_rates_df,
            palette=intn_pal, linewidth=0.5, fliersize=1)
loc = 1
tune_idx = 0.5
for grid in grid_seeds:
    full_data = intn_rates_df.loc[(intn_rates_df['tuning'] == 'full')
                                     &
                                     (intn_rates_df['grid seed'] == grid)]
    noFB_data = intn_rates_df.loc[(intn_rates_df['tuning'] == 'noFB')
                                                                          &
                                     (intn_rates_df['grid seed'] == grid)]

    full_data = np.array(full_data['mean rate'])[0]
    noFB_data = np.array(noFB_data['mean rate'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[full_data, noFB_data], color='k', linewidth=0.05)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[full_data, noFB_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('Mean rate (Hz)')
f5c2.subplots_adjust(bottom=0.3, left=0.5, top=0.9)
sns.despine(fig=f5c2)
_adjust_box_widths(f5c2, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f5c2.savefig(f'{save_dir}figure05_C_intn_bar.svg', dpi=200)
f5c2.savefig(f'{save_dir}figure05_C_intn_bar.png', dpi=200)

# =============================================================================
# Figure 5F
# =============================================================================

ca3_pal = {'full': '#09422d', 'noFB': '#66a253'}
f5j, ax1 = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm))
sns.boxplot(x='tuning', y='mean weight', ax=ax1, zorder=1,
            data=weights_df,
            palette=ca3_pal, linewidth=0.5, fliersize=1)
loc = 1
tune_idx = 0.5
for grid in grid_seeds:
    full_data = weights_df.loc[(weights_df['tuning'] == 'full')
                                     &
                                     (weights_df['grid seed'] == grid)]
    noFB_data = weights_df.loc[(weights_df['tuning'] == 'noFB')
                                                                          &
                                     (weights_df['grid seed'] == grid)]

    full_data = np.array(full_data['mean weight'])[0]
    noFB_data = np.array(noFB_data['mean weight'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[full_data, noFB_data], color='k', linewidth=0.05)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[full_data, noFB_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('Mean weight (Hz)')
f5j.subplots_adjust(bottom=0.3, left=0.5, top=0.9)
sns.despine(fig=f5j)
_adjust_box_widths(f5j, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f5j.savefig(f'{save_dir}figure05_J_bar.svg', dpi=200)
f5j.savefig(f'{save_dir}figure05_J_bar.png', dpi=200)

# =============================================================================
# Figure 5H        ### here weights_df still conteins the 'symmetric STDP' data
# =============================================================================

ca3_pal = {'full': '#09422d', 'noFB': '#66a253'}
f5l, ax1 = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm))

df = weights_df['weights/rates'] = weights_df['mean weight']/ca3_rates_df['mean rate']

sns.boxplot(x='tuning', y='weights/rates', ax=ax1, zorder=1,
            data=weights_df,
            palette=ca3_pal, linewidth=0.5, fliersize=1)
loc = 1
tune_idx = 0.5
for grid in grid_seeds:
    full_data = weights_df.loc[(weights_df['tuning'] == 'full')
                                     &
                                     (weights_df['grid seed'] == grid)]
    noFB_data = weights_df.loc[(weights_df['tuning'] == 'noFB')
                                                                          &
                                     (weights_df['grid seed'] == grid)]

    full_data = np.array(full_data['mean weight'])[0]
    noFB_data = np.array(noFB_data['mean weight'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[full_data, noFB_data], color='k', linewidth=0.05)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[full_data, noFB_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('Mean weight (Hz)')
f5l.subplots_adjust(bottom=0.3, left=0.5, top=0.9)
sns.despine(fig=f5l)
_adjust_box_widths(f5l, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f5l.savefig(f'{save_dir}figure05_L_bar.svg', dpi=200)
f5l.savefig(f'{save_dir}figure05_L_bar.png', dpi=200)


# =============================================================================
# Figure 5J
# =============================================================================

fname = 'figure05_F_condition_dict.pkl'
curr_dir = os.path.join(results_dir, 'pickled')
rates_df, weights_df = f5_load_data(fname, curr_dir)


ca3_pal = {'full': '#09422d', 'noFB': '#66a253'}
f5f, ax1 = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm))
sns.boxplot(x='tuning', y='mean weight', ax=ax1, zorder=1,
            data=weights_df,
            palette=ca3_pal, linewidth=0.5, fliersize=1)
loc = 1
tune_idx = 0.5
for grid in grid_seeds:
    full_data = weights_df.loc[(weights_df['tuning'] == 'full')
                                     &
                                     (weights_df['grid seed'] == grid)]
    noFB_data = weights_df.loc[(weights_df['tuning'] == 'noFB')
                                                                          &
                                     (weights_df['grid seed'] == grid)]

    full_data = np.array(full_data['mean weight'])[0]
    noFB_data = np.array(noFB_data['mean weight'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[full_data, noFB_data], color='k', linewidth=0.05)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[full_data, noFB_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('Mean weight (Hz)')
f5f.subplots_adjust(bottom=0.3, left=0.5, top=0.9)
sns.despine(fig=f5f)
_adjust_box_widths(f5f, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f5f.savefig(f'{save_dir}figure05_F_bar.svg', dpi=200)
f5f.savefig(f'{save_dir}figure05_F_bar.png', dpi=200)


# =============================================================================
# Figure 5L     ## here weights_df should still contain the 'assymmetric STDP' data
# =============================================================================

df = weights_df['weights/rates'] = weights_df['mean weight']/ca3_rates_df['mean rate']

ca3_pal = {'full': '#09422d', 'noFB': '#66a253'}
f5h, ax1 = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm))
sns.boxplot(x='tuning', y='mean weight', ax=ax1, zorder=1,
            data=weights_df,
            palette=ca3_pal, linewidth=0.5, fliersize=1)
loc = 1
tune_idx = 0.5
for grid in grid_seeds:
    full_data = weights_df.loc[(weights_df['tuning'] == 'full')
                                     &
                                     (weights_df['grid seed'] == grid)]
    noFB_data = weights_df.loc[(weights_df['tuning'] == 'noFB')
                                                                          &
                                     (weights_df['grid seed'] == grid)]

    full_data = np.array(full_data['mean weight'])[0]
    noFB_data = np.array(noFB_data['mean weight'])[0]
    sns.lineplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                 y=[full_data, noFB_data], color='k', linewidth=0.05)
    sns.scatterplot(x=[-loc/2+tune_idx, loc/2+tune_idx], ax=ax1,
                    y=[full_data, noFB_data], color='k', s=3)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=60)
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(-1, 1.5),
           loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('Mean weight (Hz)')
f5h.subplots_adjust(bottom=0.3, left=0.5, top=0.9)
sns.despine(fig=f5h)
_adjust_box_widths(f5h, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f5h.savefig(f'{save_dir}figure05_H_bar.svg', dpi=200)
f5h.savefig(f'{save_dir}figure05_H_bar.png', dpi=200)
