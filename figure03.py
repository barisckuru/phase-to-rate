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
plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10) #fontsize of the legend
cm=1/2.54



# =============================================================================
# Figure 3 C
# =============================================================================

full_dir = 'pickled/75-15_full_perceptron_speed.pkl'
fname = results_dir + full_dir
with open(fname, 'rb') as f:
    full_perceptron = pickle.load(f)