#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:29:15 2022

@author: baris
"""
import pickle
import numpy as np
import seaborn as sns
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


# file directory
dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')
save_dir = dirname
save_dir = 'c://phase-to-rate/'

# plotting settings
sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)
plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=8) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10) #fontsize of the legend
cm=1/2.54

# =============================================================================
# Pos_Info
# =============================================================================

'''Positional Information (Tingley et al., 2018)- Average of Population
cells firing less than 8 spikes are filtered out
10 grid seeds, 20 poisson seeds aggregated,
spatial bin = 2cm'''

grid_pal = {'non-shuffled': '#716969', 'shuffled': '#a09573'}
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
legend_size = 8

fname = 'C:/phase-to-rate/pos_info_rate_results.xlsx'
df_skaggs = pd.read_excel(fname, index_col=0)
grid_seeds = list(np.arange(1,11,1))
grid_info = df_skaggs[df_skaggs['cell']=='full grid']
grid_info['grid_seed'] = 2*grid_seeds
granule_info = df_skaggs[df_skaggs['cell']!='full grid']
granule_info['grid_seed'] = 8*grid_seeds

f1, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]},
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

ax2.set_xticklabels(['EC','full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.5),
           loc='upper left', borderaxespad=0., title=None,
           prop={'size': legend_size})
ax2.set_ylabel('info (bits/AP)')

f1.subplots_adjust(bottom=0.3, left=0.21, wspace=1, top=0.75)
sns.despine(fig=f1)
_adjust_box_widths(f1, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f1.savefig(f'{save_dir}figure02_H.svg', dpi=200)
f1.savefig(f'{save_dir}figure02_H.png', dpi=200)



# =============================================================================
# Positional Info (nonshuffled/shuffled)
# =============================================================================

scaled_pal = {'nonshuffled/shuffled': '#267282'}

fname = results_dir + 'excel/fig2j_adjusted-filtered-info.xlsx'
fname = 'C:/phase-to-rate/figure_2I_skaggs_non-adjusted.xlsx'
dfa = pd.read_excel(fname, index_col=0)
dfa2 = copy.deepcopy(dfa)
dfa2_ns = dfa2[dfa2['shuffling']=='non-shuffled'].reset_index(drop=True)
dfa2_s = dfa2[dfa2['shuffling']=='shuffled'].reset_index(drop=True)
dfa2_div = dfa2_ns['info']/dfa2_s['info']
dfa3 = copy.deepcopy(dfa2_ns)
dfa3['info (bits/AP)'] = dfa2_div
dfa3.rename(columns={'info':'scaled value (non-shuffled/shuffled)'},
            inplace=True)
dfa3['shuffling'] = 'nonshuffled/shuffled'

# output_name = '/home/baris/results/excel/figure2I_adjusted_scaled_skaggs-info.xlsx'

# with pd.ExcelWriter(output_name) as writer:
#     dfa3.to_excel(writer, sheet_name='scaled_info')
    
plt.close('all')
f2, ax = plt.subplots(1,1, figsize=(5*cm, 4*cm))
sns.boxplot(x='cell', y='scaled value (non-shuffled/shuffled)', ax=ax,
            data=dfa3, palette=scaled_pal, zorder=1, linewidth=0.5, fliersize=1,
            hue='shuffling')
# sns.scatterplot(x='tuning', y='scaled value (non-shuffled/shuffled)', ax=ax,
#                 data=dfa3, color='black', alpha=0.8)
sns.stripplot(x='cell', y='scaled value (non-shuffled/shuffled)', ax=ax,
              zorder=2, data=dfa3,
              color='black', jitter=False, dodge=True, linewidth=0.1, size=2)
sns.lineplot(x='cell', y='scaled value (non-shuffled/shuffled)', ax=ax,
             zorder=1, legend=False,
             data=dfa3, linewidth=1,
             palette= sns.color_palette(10*['black']), alpha=0.3, errorbar=None)
#ax.set_ylim([1,1.4])
ax.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
ax.set_title('non-shuffled/shuffled')
ax.set_ylabel('scaled value')
ax.get_legend().remove()
f2.subplots_adjust(left=0.3, bottom=0.3)
sns.despine(fig=f2)
_adjust_box_widths(f2, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f2.savefig(f'{save_dir}figure_pos_info_rate.svg', dpi=200)
f2.savefig(f'{save_dir}figure_pos_info_rate.png', dpi=200)

