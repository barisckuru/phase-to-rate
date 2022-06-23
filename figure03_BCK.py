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
from neural_coding import load_spikes, rate_n_phase
from perceptron import run_perceptron
import os


# file directory
results_dir = '/home/baris/results/'
save_dir = '/home/baris/paper/figures/figure03/'

# plotting settings
sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)
plt.rc('font', size=8) #controls default text size
plt.rc('axes', titlesize=8) #fontsize of the title
plt.rc('axes', labelsize=8) #fontsize of the x and y labels
plt.rc('xtick', labelsize=8) #fontsize of the x tick labels
plt.rc('ytick', labelsize=8) #fontsize of the y tick labels
plt.rc('legend', fontsize=8) #fontsize of the legend
cm=1/2.54

phase_pal = {'non-shuffled': '#8f1010', 'shuffled': '#ff7900'}
rate_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}

# =============================================================================
# Figure 3 B
# =============================================================================
           
fname = 'figure03B_threshold_n_loss.pkl'
save_pickle = '/home/baris/results/pickled/'
with open(save_pickle+fname, 'wb') as f:
    pickle.dump(th_dict, f)

th_dir = 'pickled/figure03B_threshold_n_loss.pkl'
# keys in this file are 'loss' and 'th_cross'
fname = results_dir + th_dir
with open(fname, 'rb') as f:
    thresholds = pickle.load(f)
    
    
f3b, ax1 = plt.subplots(figsize=(2.5*cm, 4*cm))
ax1.plot(thresholds['loss']['nearby'], 'b-')
ax1.plot(thresholds['loss']['intermediate'], 'b--')
ax1.plot(thresholds['loss']['distant'], linestyle=(0, (1, 1)) )
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss")
handles, labels = ax1.get_legend_handles_labels()

ax1.legend(("2cm", "5cm", "10cm"), bbox_to_anchor=(0.2, 1.2),
           loc='upper left', borderaxespad=0., title=None, ncol=1,
           frameon=False, handlelength=1.5, columnspacing=0.8,
           handletextpad=0.5)
ax1.hlines(0.2, xmin=0, xmax=10000, color='k', alpha=0.8, linewidth=1.2)
ax1.vlines(thresholds['th_cross']['nearby'], ymin=0, ymax=0.35,
           color='k', alpha=0.8, linewidth=1.2)
ax1.set_xlim(0, 10000)
ax1.set_ylim(0, 0.55)
sns.despine(fig=f3b)
f3b.subplots_adjust(bottom=0.3, wspace=1, top=0.9, left=0.3)
plt.rcParams["svg.fonttype"] = "none"
f3b.savefig(f'{save_dir}figure03_B.svg', dpi=200)
f3b.savefig(f'{save_dir}figure03_B.png', dpi=200)

# =============================================================================
# Figure 3 C
# =============================================================================
plt.close('all')

full_dir = 'pickled/75-15_full_perceptron_speed.pkl'
fname = results_dir + full_dir
with open(fname, 'rb') as f:
    full_load = pickle.load(f)
    
    
f3c, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*cm, 5*cm), sharex=True)
    
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
             palette=rate_pal, ci='sd', ax = ax1, linewidth=1.2,
             err_kws={'linewidth':0})
sns.lineplot(data=full_grid_phase, x='distance', y='speed', hue='shuffling',
             palette=phase_pal, ci='sd', ax=ax2, linewidth=1.2,
             err_kws={'linewidth':0})

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1[::-1], labels1[::-1],
           bbox_to_anchor=(1, 1.05), title=None, handlelength=0.5,
           borderaxespad=0., frameon=False, columnspacing=0.8)
ax2.legend(handles2[::-1], labels2[::-1],
           bbox_to_anchor=(1, 1.05), frameon=False, handlelength=0.5,
           borderaxespad=0., title=None, columnspacing=0.8)

ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
ax1.set_ylabel('Speed')
ax2.set_ylabel('Speed')
ax2.set_xlabel('Distance (cm)')
sns.despine(ax=ax1)
sns.despine(ax=ax2)
f3c.subplots_adjust(bottom=0.3, hspace=1, top=0.9, left=0.25, right=0.6)
plt.rcParams["svg.fonttype"] = "none"
f3c.savefig(f'{save_dir}figure03_C.svg', dpi=200)
f3c.savefig(f'{save_dir}figure03_C.png', dpi=200)

f3c2, ax = plt.subplots(figsize=(3*cm, 4*cm))
ax.set_ylim(0,2.5)

palette = sns.color_palette(['#267282', '#d63515'])
df = full_perceptron.loc[(full_perceptron['cell']=='grid')&
                                 (full_perceptron['distance']==15)]
ns_df = df.loc[df['shuffling']=='non-shuffled']
s_df = df.loc[df['shuffling']=='shuffled']
grid_all = ns_df.copy()
ns_speed = ns_df['speed'].reset_index(drop=True)
s_speed = s_df['speed'].reset_index(drop=True)
div = ns_speed/s_speed
grid_all.reset_index(inplace=True, drop=True)
grid_all['division'] = div
sns.boxplot(x='code', y='division', data=grid_all, palette=palette,
            linewidth=0.7)
f3c2.subplots_adjust(bottom=0.2, hspace=0.5, top=0.95, left=0.25, right=0.9)
sns.despine(ax=ax)
plt.rcParams["svg.fonttype"] = "none"
f3c2.savefig(f'{save_dir}figure03_C2.svg', dpi=200)
f3c2.savefig(f'{save_dir}figure03_C2.png', dpi=200)
# =============================================================================
# Figure 3 D
# =============================================================================
plt.close('all')

full_dir = 'pickled/75-15_full_perceptron_speed.pkl'
fname = results_dir + full_dir
with open(fname, 'rb') as f:
    full_load = pickle.load(f)
    
f3d, (ax1, ax2) = plt.subplots(2, 1, figsize=(3*cm, 5*cm), sharex=True)
 
full_perceptron = pd.DataFrame(full_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
full_gra_rate = full_perceptron.loc[(full_perceptron['code']=='rate') &
                                     (full_perceptron['cell']=='granule')]
full_gra_phase = full_perceptron.loc[(full_perceptron['code']=='phase') &
                                      (full_perceptron['cell']=='granule')]

sns.lineplot(data=full_gra_rate, x='distance', y='speed', hue='shuffling',
             palette=rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=full_gra_phase, x='distance', y='speed', hue='shuffling',
             palette=phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
f3d.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
ax1.set_ylabel('Speed')
ax2.set_ylabel('Speed')
ax2.set_xlabel('Distance (cm)')
plt.rcParams["svg.fonttype"] = "none"
f3d.savefig(f'{save_dir}figure03_D.svg', dpi=200)
f3d.savefig(f'{save_dir}figure03_D.png', dpi=200)

# =============================================================================
# Figure 3 E
# =============================================================================
plt.close('all')

noff_dir = 'pickled/75-15_no-feedforward_perceptron_speed_polar_inc.pkl'
fname = results_dir + noff_dir
with open(fname, 'rb') as f:
    noff_load = pickle.load(f)

f3e, (ax1, ax2) = plt.subplots(2, 1, figsize=(3*cm, 5*cm), sharex=True)
    
noff_perceptron = pd.DataFrame(noff_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
noff_gra_rate = noff_perceptron.loc[(noff_perceptron['code']=='rate') &
                                     (noff_perceptron['cell']=='granule')]
noff_gra_phase = noff_perceptron.loc[(noff_perceptron['code']=='phase') &
                                      (noff_perceptron['cell']=='granule')]

sns.lineplot(data=noff_gra_rate, x='distance', y='speed', hue='shuffling',
             palette=rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=noff_gra_phase, x='distance', y='speed', hue='shuffling',
             palette=phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
f3e.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
ax1.set_ylabel('Speed')
ax2.set_ylabel('Speed')
ax2.set_xlabel('Distance (cm)')
plt.rcParams["svg.fonttype"] = "none"
f3e.savefig(f'{save_dir}figure03_E.svg', dpi=200)
f3e.savefig(f'{save_dir}figure03_E.png', dpi=200)


# =============================================================================
# Figure 3 F
# =============================================================================
plt.close('all')

nofb_dir = 'pickled/75-15_no-feedback_perceptron_speed_polar_inc.pkl'
fname = results_dir + nofb_dir
with open(fname, 'rb') as f:
    nofb_load = pickle.load(f)

f3f, (ax1, ax2) = plt.subplots(2, 1, figsize=(3*cm, 5*cm), sharex=True)
    
nofb_perceptron = pd.DataFrame(nofb_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
nofb_gra_rate = nofb_perceptron.loc[(nofb_perceptron['code']=='rate') &
                                     (nofb_perceptron['cell']=='granule')]
nofb_gra_phase = nofb_perceptron.loc[(nofb_perceptron['code']=='phase') &
                                      (nofb_perceptron['cell']=='granule')]

sns.lineplot(data=nofb_gra_rate, x='distance', y='speed', hue='shuffling',
             palette=rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=nofb_gra_phase, x='distance', y='speed', hue='shuffling',
             palette=phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
f3f.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
ax1.set_ylabel('Speed')
ax2.set_ylabel('Speed')
ax2.set_xlabel('Distance (cm)')
plt.rcParams["svg.fonttype"] = "none"
f3f.savefig(f'{save_dir}figure03_F.svg', dpi=200)
f3f.savefig(f'{save_dir}figure03_F.png', dpi=200)


# =============================================================================
# Figure 3 G
# =============================================================================
plt.close('all')

disinh_dir = 'pickled/75-15_disinhibited_perceptron_speed_polar_inc.pkl'
fname = results_dir + disinh_dir
with open(fname, 'rb') as f:
    disinh_load = pickle.load(f)

f3g, (ax1, ax2) = plt.subplots(2, 1, figsize=(3*cm, 5*cm), sharex=True)
    
disinh_perceptron = pd.DataFrame(disinh_load, columns=['distance', 'speed',
                                                         'th_cross',
                                                         'trajectories',
                                                         'grid_seed',
                                                         'shuffling', 'cell',
                                                         'code', 'lr'])
disinh_gra_rate = disinh_perceptron.loc[(disinh_perceptron['code']=='rate') &
                                     (disinh_perceptron['cell']=='granule')]
disinh_gra_phase = disinh_perceptron.loc[(disinh_perceptron['code']=='phase') &
                                      (disinh_perceptron['cell']=='granule')]

sns.lineplot(data=disinh_gra_rate, x='distance', y='speed', hue='shuffling',
             palette=rate_pal, ci='sd', ax = ax1, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
# ax1.set_title('rate code \n lr=0.0001')
# ax2.set_title('phase code \n lr=0.001')
ax1.ticklabel_format(style='sci', scilimits=[0, 2])
ax2.ticklabel_format(style='sci', scilimits=[0, 2])
sns.despine(ax=ax1)
sns.despine(ax=ax2)
sns.lineplot(data=disinh_gra_phase, x='distance', y='speed', hue='shuffling',
             palette=phase_pal, ci='sd', ax=ax2, legend=False,
             linewidth=1.2, err_kws={'linewidth':0})
f3g.subplots_adjust(bottom=0.22, hspace=0.5, top=0.90, left=0.3, right=0.95)
ax1.set_ylabel('Speed')
ax2.set_ylabel('Speed')
ax2.set_xlabel('Distance (cm)')
plt.rcParams["svg.fonttype"] = "none"
f3g.savefig(f'{save_dir}figure03_G.svg', dpi=200)
f3g.savefig(f'{save_dir}figure03_G.png', dpi=200)

# =============================================================================
# Figure 3 H and I 
# =============================================================================
plt.close('all')
full_gra_rate['tuning'] = 320*['full']
noff_gra_rate['tuning'] = 320*['noff']
nofb_gra_rate['tuning'] = 320*['nofb']
disinh_gra_rate['tuning'] = 320*['disinh']
full_gra_phase['tuning'] = 320*['full']
noff_gra_phase['tuning'] = 320*['noff']
nofb_gra_phase['tuning'] = 320*['nofb']
disinh_gra_phase['tuning'] = 320*['disinh']
rate_all = pd.concat([full_gra_rate, noff_gra_rate,
                      nofb_gra_rate, disinh_gra_rate], ignore_index=True)
rate_all_2 = rate_all.copy()
rate_all_3 = rate_all_2.loc[(rate_all_2['shuffling']=='non-shuffled')&
                            (rate_all_2['distance']==15)]
nonshuff_speed = rate_all_3['speed'].reset_index(drop=True)
shuff_speed = rate_all_2.loc[(rate_all_2['shuffling']=='shuffled')&
                             (rate_all_2['distance']==15)]
shuff_speed = shuff_speed['speed'].reset_index(drop=True)
div = nonshuff_speed/shuff_speed
rate_all_3.reset_index(drop=True, inplace=True)
rate_all_3['division'] = div

phase_all = pd.concat([full_gra_phase, noff_gra_phase,
                      nofb_gra_phase, disinh_gra_phase], ignore_index=True)

phase_all_2 = phase_all.copy()
phase_all_3 = phase_all_2.loc[(phase_all_2['shuffling']=='non-shuffled')&
                             (rate_all_2['distance']==15)]
nonshuff_speed = phase_all_3['speed'].reset_index(drop=True)
shuff_speed = phase_all_2.loc[(phase_all_2['shuffling']=='shuffled')&
                             (rate_all_2['distance']==15)]
shuff_speed = shuff_speed['speed'].reset_index(drop=True)
div = nonshuff_speed/shuff_speed
phase_all_3.reset_index(drop=True, inplace=True)
phase_all_3['division'] = div

plt.close('all')
f3hi, (ax1, ax2) = plt.subplots(1,2, figsize=(7*cm, 3.5*cm), sharey=True)
# f3i, ax2 = plt.subplots(figsize=(3.5*cm, 3.5*cm), sharex=True)

sns.boxplot(x='tuning', y='division', ax=ax1, zorder=1,
            data=rate_all_3, linewidth=0.5, fliersize=1,
            palette= sns.color_palette(4*['#267282']))
sns.lineplot(x='tuning', y='division', ax=ax1, zorder=1,  legend=False,
            data=rate_all_3, linewidth=0.2, hue = 'grid_seed', color='black',
            palette= sns.color_palette(10*['black']), alpha=0.7)
sns.scatterplot(x='tuning', y='division', ax=ax1, zorder=1,  legend=False,
                data=rate_all_3, s=5, hue = 'grid_seed', color='black',
                palette= sns.color_palette(10*['black']))

sns.boxplot(x='tuning', y='division', ax=ax2, zorder=1,
            data=phase_all_3, linewidth=0.5, fliersize=1,
            palette= sns.color_palette(4*['#d63515']))
sns.lineplot(x='tuning', y='division', ax=ax2, zorder=1, legend=False,
            data=phase_all_3, linewidth=0.2, hue='grid_seed',
            palette= sns.color_palette(10*['black']), alpha=0.7)
sns.scatterplot(x='tuning', y='division', ax=ax2, zorder=1, legend=False,
                data=phase_all_3, s=5, hue='grid_seed',
                palette= sns.color_palette(10*['black']))
sns.despine(ax=ax1)
sns.despine(ax=ax2)
ax1.set_xlabel('')
ax1.set_ylabel('nonshuffled/shuffled')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
ax1.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
f3hi.subplots_adjust(bottom=0.3, top=0.95, left=0.2, right=0.95, wspace=0.7)
plt.rcParams["svg.fonttype"] = "none"
f3hi.savefig(f'{save_dir}figure03_H&I.svg', dpi=200)
f3hi.savefig(f'{save_dir}figure03_H&I.png', dpi=200)


# =============================================================================
# Figure 3J&K
# =============================================================================

rate_df = pd.DataFrame(columns=['grid_seed', 'speed', 'threshold_crossing',
                                'shuffling', 'code_type', 'learning_rate',
                                'tuning'])
phase_df = pd.DataFrame(columns=['grid_seed', 'speed', 'threshold_crossing',
                                'shuffling', 'code_type', 'learning_rate',
                                'tuning'])
tunes = ['full', 'noff', 'nofb', 'disinh']
for tuning in tunes:
    f_dir = f'pickled/{tuning}_adjusted_perceptron.pkl'
    fname = results_dir + f_dir
    with open(fname, 'rb') as f:
        loaded = pickle.load(f)
        df = loaded.loc[loaded['grid_seed']<21]
        df = df[df['grid_seed']!=14]
        df['tuning'] = 36*[tuning]
        rate_df = rate_df.append(df.groupby(['code_type']).get_group(('rate')),
                       ignore_index=True)
        phase_df= phase_df.append(df.groupby(['code_type']).get_group(('phase')),
                        ignore_index=True)

output_name = '/home/baris/results/excel/figure3J&K_adjusted_data-perceptron.xlsx'

with pd.ExcelWriter(output_name) as writer:
    rate_df.to_excel(writer, sheet_name='rate code')
    phase_df.to_excel(writer, sheet_name='phase code')
    
    
rate_all_2 = rate_df.copy()
rate_all_3 = rate_all_2.loc[(rate_all_2['shuffling']=='non-shuffled')]
nonshuff_speed = rate_all_3['speed'].reset_index(drop=True)
shuff_speed = rate_all_2.loc[(rate_all_2['shuffling']=='shuffled')]
shuff_speed = shuff_speed['speed'].reset_index(drop=True)
div = nonshuff_speed/shuff_speed
rate_all_3.reset_index(drop=True, inplace=True)
rate_all_3['division'] = div
        
phase_all_2 = phase_df.copy()
phase_all_3 = phase_all_2.loc[(phase_all_2['shuffling']=='non-shuffled')]
nonshuff_speed = phase_all_3['speed'].reset_index(drop=True)
shuff_speed = phase_all_2.loc[(phase_all_2['shuffling']=='shuffled')]
shuff_speed = shuff_speed['speed'].reset_index(drop=True)
div = nonshuff_speed/shuff_speed
phase_all_3.reset_index(drop=True, inplace=True)
phase_all_3['division'] = div


plt.close('all')
f3jk, (ax1, ax2) = plt.subplots(1,2, figsize=(7*cm, 3.5*cm), sharey=True)
# f3i, ax2 = plt.subplots(figsize=(3.5*cm, 3.5*cm), sharex=True)

sns.boxplot(x='tuning', y='division', ax=ax1, zorder=1,
            data=rate_all_3, linewidth=0.5, fliersize=1.2,
            palette= sns.color_palette(4*['#267282']))
sns.lineplot(x='tuning', y='division', ax=ax1, zorder=1,  legend=False,
            data=rate_all_3, linewidth=0.2, hue = 'grid_seed', color='black',
            palette= sns.color_palette(9*['black']), alpha=0.7)
sns.scatterplot(x='tuning', y='division', ax=ax1, zorder=1,  legend=False,
                data=rate_all_3, s=5, hue = 'grid_seed', color='black',
                palette= sns.color_palette(9*['black']))


sns.boxplot(x='tuning', y='division', ax=ax2, zorder=1,
            data=phase_all_3, linewidth=0.5, fliersize=1.2,
            palette= sns.color_palette(4*['#d63515']))
sns.lineplot(x='tuning', y='division', ax=ax2, zorder=1, legend=False,
            data=phase_all_3, linewidth=0.2, hue='grid_seed',
            palette= sns.color_palette(9*['black']), alpha=0.7)
sns.scatterplot(x='tuning', y='division', ax=ax2, zorder=1, legend=False,
                data=phase_all_3, s=5, hue='grid_seed',
                palette= sns.color_palette(9*['black']))
sns.despine(ax=ax1)
sns.despine(ax=ax2)
ax1.set_xlabel('')
ax1.set_ylabel('nonshuffled/shuffled')
ax2.set_ylabel('')
ax2.set_xlabel('')
ax2.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
ax1.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
f3jk.subplots_adjust(bottom=0.3, top=0.95, left=0.2, right=0.95, wspace=0.7)
plt.rcParams["svg.fonttype"] = "none"
f3jk.savefig(f'{save_dir}figure03_J&K.svg', dpi=200)
f3jk.savefig(f'{save_dir}figure03_J&K.png', dpi=200)


# # =============================================================================
# # 3J 
# # =============================================================================
# plt.close('all')
# grid_seeds = list(np.arange(11,21,1))
# f3j, ax = plt.subplots(1,1, figsize=(4*cm, 4*cm))

# sns.boxplot(x='tuning', y='speed', hue='shuffling', ax=ax, zorder=1,
#             data=rate_df, linewidth=0.5, fliersize=1,
#             palette=rate_pal, hue_order=['non-shuffled', 'shuffled'])
# loc = 0.4
# tunings = ['full', 'noff', 'nofb', 'disinh']
# for grid in grid_seeds:
#     tune_idx = 0
#     for tuning in tunings:
#         ns_data = rate_df.loc[(rate_df['shuffling']=='non-shuffled')
#                                       &
#                                       (rate_df['tuning']==tuning)
#                                       &
#                                       (rate_df['grid_seed']==grid)]
#         s_data = rate_df.loc[(rate_df['shuffling']=='shuffled')
#                                       &
#                                       (rate_df['tuning']==tuning)
#                                       &
#                                       (rate_df['grid_seed']==grid)]
#         if (np.array(ns_data['speed']).size > 0 and
#             np.array(s_data['speed']).size > 0):
#             ns_info = np.array(ns_data['speed'])[0]
#             s_info = np.array(s_data['speed'])[0]   
#             sns.lineplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax, alpha=0.7,
#                          y = [ns_info, s_info], color='k', linewidth = 0.2)
#             sns.scatterplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax,
#                           y = [ns_info, s_info], color='k', s = 3, alpha=0.7)
#             print(tuning)
#         tune_idx+=1
# ax.ticklabel_format(style='sci', scilimits=[0, 0], axis='y')
# ax.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
#                     , rotation=60)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.4),
#           loc='upper left', borderaxespad=0., title=None,frameon=False)
# # ax.set_title('adjusted rate codes')
# ax.set_ylabel('Speed')
# f3j.subplots_adjust(left=0.25, bottom=0.3, right=0.9, top=0.8)
# sns.despine(fig=f3j)
# _adjust_box_widths(f3j, 0.7)
# plt.rcParams["svg.fonttype"] = "none"
# f3j.savefig(f'{save_dir}figure03_J.svg', dpi=200)
# f3j.savefig(f'{save_dir}figure03_J.png', dpi=200)

# # =============================================================================
# # 3K
# # =============================================================================
# plt.close('all')
# grid_seeds = list(np.arange(11,21,1))
# f3k, ax = plt.subplots(1,1, figsize=(4*cm, 4*cm))

# sns.boxplot(x='tuning', y='speed', hue='shuffling', ax=ax, zorder=1,
#             data=phase_df, linewidth=0.5, fliersize=1,
#             palette=phase_pal, hue_order=['non-shuffled', 'shuffled'])
# loc = 0.4
# tunings = ['full', 'noff', 'nofb', 'disinh']
# for grid in grid_seeds:
#     tune_idx = 0
#     for tuning in tunings:
#         ns_data = phase_df.loc[(phase_df['shuffling']=='non-shuffled')
#                                       &
#                                       (phase_df['tuning']==tuning)
#                                       &
#                                       (phase_df['grid_seed']==grid)]
#         s_data = phase_df.loc[(phase_df['shuffling']=='shuffled')
#                                       &
#                                       (phase_df['tuning']==tuning)
#                                       &
#                                       (phase_df['grid_seed']==grid)]
#         if (np.array(ns_data['speed']).size > 0 and
#             np.array(s_data['speed']).size > 0):
#             ns_info = np.array(ns_data['speed'])[0]
#             s_info = np.array(s_data['speed'])[0]   
#             sns.lineplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax, alpha=0.7,
#                          y = [ns_info, s_info], color='k', linewidth = 0.2)
#             sns.scatterplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax,
#                           y = [ns_info, s_info], color='k', s = 3, alpha=0.7)
#             print(tuning)
#         tune_idx+=1
# ax.ticklabel_format(style='sci', scilimits=[0, 0], axis='y')
# ax.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
#                     , rotation=60)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.4),
#           loc='upper left', borderaxespad=0., title=None, frameon=False)
# # ax.set_title('adjusted phase codes')
# ax.set_ylabel('Speed')
# f3k.subplots_adjust(left=0.25,  bottom=0.3, right=0.9, top=0.8)
# sns.despine(fig=f3k)
# _adjust_box_widths(f3k, 0.7)
# plt.rcParams["svg.fonttype"] = "none"
# f3k.savefig(f'{save_dir}figure03_K.svg', dpi=200)
# f3k.savefig(f'{save_dir}figure03_K.png', dpi=200)
       
        
       