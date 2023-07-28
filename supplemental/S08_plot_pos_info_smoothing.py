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

tuning = 'full'
#tuning = 'no-feedforward'
tuning = 'no-feedback'
#tuning = 'disinhibited'

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
# Pos_Info Rate
# =============================================================================

'''Positional Information (Tingley et al., 2018)- Average of Population
sm_cells firing less than 8 spikes are filtered out
10 grid seeds, 20 poisson seeds aggregated,
spatial bin = 2cm'''

grid_pal = {'non-shuffled': '#716969', 'shuffled': '#a09573'}
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
legend_size = 8


# =============================================================================
# Positional Info across smoothing bins
# =============================================================================


fname = 'C:/phase-to-rate/pos_info_rate_non-adjusted_smoothing_{}.xlsx'.format(tuning)
dfa = pd.read_excel(fname, index_col=0)
dfa_grid = copy.deepcopy(dfa)
dfa_grid = dfa_grid[dfa_grid['cell']=='grid'].reset_index(drop=True)
dfa_granule = copy.deepcopy(dfa)
dfa_granule = dfa_granule[dfa_granule['cell']=='granule'].reset_index(drop=True)


f3, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(12*cm, 8*cm))
sns.lineplot(x='smoothing', y='info', ax=ax1,
            data=dfa_grid, palette=grid_pal, zorder=1, linewidth=0.5,
            hue='shuffling')
sns.lineplot(x='smoothing', y='info', ax=ax2,
            data=dfa_granule, palette=granule_pal, zorder=1, linewidth=0.5,
            hue='shuffling')

#ax.set_ylim([1,1.4])
#ax1.set_xticklabels(range(1,40))
ax1.set_title('grid rate-code ')
#ax1.set_ylabel('scaled value')
#ax2.set_xticklabels(range(1,40))
ax2.set_title('granule rate-code')
#ax2.set_ylabel('scaled value')
ax1.legend([],[], frameon=False)
ax2.legend([],[], frameon=False)

fname = 'C:/phase-to-rate/pos_info_phase_non-adjusted_smoothing_{}.xlsx'.format(tuning)
dfa = pd.read_excel(fname, index_col=0)
dfa_grid = copy.deepcopy(dfa)
dfa_grid = dfa_grid[dfa_grid['cell']=='grid'].reset_index(drop=True)
dfa_granule = copy.deepcopy(dfa)
dfa_granule = dfa_granule[dfa_granule['cell']=='granule'].reset_index(drop=True)

sns.lineplot(x='smoothing', y='info', ax=ax3,
            data=dfa_grid, palette=grid_pal, zorder=1, linewidth=0.5,
            hue='shuffling')
sns.lineplot(x='smoothing', y='info', ax=ax4,
            data=dfa_granule, palette=granule_pal, zorder=1, linewidth=0.5,
            hue='shuffling')

#ax.set_ylim([1,1.4])
#ax3.set_xticklabels(range(1,40))
ax3.set_title('grid phase-code')
#ax3.set_ylabel('scaled value')
#ax4.set_xticklabels(range(1,40))
ax4.set_title('granule phase-code')
#ax4.set_ylabel('scaled value')
ax3.legend([],[], frameon=False)
ax4.legend([],[], frameon=False)

#f2.subplots_adjust(left=0.3, bottom=0.3)
sns.despine(fig=f3)
#_adjust_box_widths(f2, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f3.savefig('pos_info_smooth_{}.svg'.format(tuning), dpi=200)
#f2.savefig(f'{save_dir}figure_pos_info_rate.png', dpi=200)

