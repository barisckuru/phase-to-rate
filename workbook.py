#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:11:14 2022

@author: baris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import pearsonr,  spearmanr
import copy
from scipy import stats


# =============================================================================
# figure 2i old version
# =============================================================================
fname = results_dir + 'excel/figure_2I_skaggs_non-adjusted.xlsx'
df_skaggs = pd.read_excel(fname, index_col=0)

grid_info = df_skaggs[df_skaggs['cell']=='full grid']
granule_info = df_skaggs[df_skaggs['cell']!='full grid']

f2i, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]},
                               figsize=(10*cm, 7*cm))
grid_pal = {'non-shuffled': '#716969', 'shuffled': '#a09573'}
granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}


sns.boxplot(x='cell', y='info', hue='shuffling', ax=ax1, zorder=1,
            data=grid_info, linewidth=0.5, fliersize=1, palette=grid_pal)
sns.stripplot(x='cell', y='info', ax=ax1, zorder=2, hue='shuffling',
                data=grid_info, color='black', jitter=False, dodge=True,
                linewidth=0.1, size=1)

ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 60)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.15),
               loc='upper left', borderaxespad=0., title=None)
ax1.set_ylabel('information (bits/AP)')

sns.boxplot(x='cell', y='info', hue='shuffling', ax=ax2, zorder=1,
            data=granule_info, linewidth=0.5, fliersize=1, palette=granule_pal)
sns.stripplot(x='cell', y='info', ax=ax2, zorder=2, hue='shuffling',
                data=granule_info, color='black', jitter=False, dodge=True,
                linewidth=0.1, size=1)
ax2.set_xticklabels(['full', 'no ff', 'no fb', 'disinh']
                    , rotation=60)
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[:2], labels[:2], bbox_to_anchor=(0, 1.15),
               loc='upper left', borderaxespad=0., title=None)
ax2.set_ylabel('information (bits/AP)')

f2i.subplots_adjust(bottom=0.3, left=0.2, wspace=0.8)
sns.despine(fig=f2i)
plt.rcParams["svg.fonttype"] = "none"
f2i.savefig(f'{save_dir}figure02_I.svg', dpi=200)
f2i.savefig(f'{save_dir}figure02_I.png', dpi=200)





trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
n_samples = 20
grid_seeds = np.arange(1, 11, 1)
tuning = ['full', 'no-feedforward', 'no-feedback', 'disinhibited']
dfs = []
for tune in tuning:
    with open(f'neural_codes_{tune}.pkl', 'rb') as f:
        all_codes = pickle.load(f)
    # 75 vs all in all time bins
    # calculate pearson R
    r_data = []
    for grid_seed in all_codes:
        for shuffling in all_codes[grid_seed]:
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
                            
    
    df = pd.DataFrame(r_data,
                      columns=['distance', 'pearson_r', 'spearman_r',
                               'poisson_seed', 'trajectories', 'grid_seed',
                               'shuffling', 'cell_type', 'code_type'])
    
    df = df.drop(columns='trajectories')
    shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                 (df['code_type'] == 'rate') &
                                 (df['shuffling'] == 'shuffled')]
                                 [['distance', 'grid_seed',
                                   'pearson_r']].reset_index(drop=True))
    shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                    (df['code_type'] == 'rate') & 
                                    (df['shuffling'] == 'shuffled')]
                                    ['pearson_r'].reset_index(drop=True))
    nonshuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                    (df['code_type'] == 'rate') &
                                    (df['shuffling'] == 'non-shuffled')]
                                 ['pearson_r'].reset_index(drop=True))
    nonshuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                       (df['code_type'] == 'rate') &
                                       (df['shuffling'] == 'non-shuffled')]
                                        ['pearson_r'].reset_index(drop=True))
    
    pearson_r = pd.concat([
        shuffled_grid_rate, shuffled_granule_rate,
        nonshuffled_grid_rate, nonshuffled_granule_rate], axis=1)
    pearson_r.columns = ['distance', 'grid_seed', 
                                's_grid_rate', 's_granule_rate',
                                'ns_grid_rate', 'ns_granule_rate']
    delta_s_rate = []
    delta_ns_rate = []
    for seed in grid_seeds:
        grid_1 = pearson_r.loc[(pearson_r['grid_seed'] == seed)]
    
        s_rate = stats.binned_statistic(grid_1['s_grid_rate'], 
                                        list((grid_1['s_grid_rate'],
                                              grid_1['s_granule_rate'])),
                                        'mean',
                                        bins=[0,0.1,0.2,0.3,0.4,
                                              0.5,0.6,0.7,0.8,0.9])
        ns_rate = stats.binned_statistic(grid_1['ns_grid_rate'],
                                         list((grid_1['ns_grid_rate'],
                                               grid_1['ns_granule_rate'])),
                                         'mean',
                                         bins=[0,0.1,0.2,0.3,0.4,
                                               0.5,0.6,0.7,0.8,0.9])
        delta_s_rate.append(np.mean((s_rate[0][0] - s_rate[0][1])
                                    [s_rate[0][0]==s_rate[0][0]]))
        delta_ns_rate.append(np.mean((ns_rate[0][0] - ns_rate[0][1])
                                     [ns_rate[0][0]==ns_rate[0][0]]))

    deltaR = np.concatenate((delta_ns_rate, delta_s_rate))
    shuffling = (10*['non-shuffled'] + 10*['shuffled'])
    code = 20*['rate']
    tuning = 20*[tune]
    deltaR = np.stack((deltaR, shuffling, code, tuning), axis=1)
    df_deltaR = pd.DataFrame(deltaR, columns=['mean deltaR', 'shuffling',
                                              'code', 'tuning'])
    df_deltaR['mean deltaR'] = df_deltaR['mean deltaR'].astype('float')
    dfs.append(df_deltaR)

    

    # load data, codes

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
n_samples = 20
grid_seeds = np.arange(1, 11, 1)
tuning = 'full'

with open('all_codes.pkl', 'rb') as f:
    all_codes = pickle.load(f)

# 75 vs all in all time bins
# calculate pearson R
r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
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
                        
                        

df_deltaR.to_pickle(f'{tuning}_mean_deltaR')

df_deltaR_full = copy.deepcopy(df_deltaR)
df_deltaR_full['tuning'] = 40*['full']
df_deltaR_full['grid_seeds'] = 4*list(np.arange(1,11,1))

df_deltaR_nofb = pd.read_pickle('no-feedback_mean_deltaR')
df_deltaR_nofb['tuning'] = 40*['no-feedback']
df_deltaR_nofb['grid_seeds'] = 4*list(np.arange(1,11,1))
df_deltaR_noff = pd.read_pickle('no-feedforward_mean_deltaR')
df_deltaR_noff['tuning'] = 40*['no-feedforward']
df_deltaR_noff['grid_seeds'] = 4*list(np.arange(1,11,1))
df_deltaR_disinh = pd.read_pickle('disinhibited_mean_deltaR')
df_deltaR_disinh['tuning'] = 40*['disinhibited']
df_deltaR_disinh['grid_seeds'] = 4*list(np.arange(1,11,1))

frames = [df_deltaR_full, df_deltaR_noff, df_deltaR_nofb, df_deltaR_disinh]
all_deltaR = pd.concat(frames)


# rate
deltaR_rate = all_deltaR[all_deltaR['code'] == 'rate']
ax = sns.catplot(data=deltaR_rate, kind='bar', x='tuning', y='mean deltaR',
                 hue='shuffling', ci='sd', capsize=0.2, errwidth=2)
ax = sns.swarmplot(data=deltaR_rate, x='tuning', y='mean deltaR',
                   hue='shuffling', color='black', dodge=True)
ax.get_legend().set_visible(False)
ax.set_title('Rate code mean delta R for different tuned networks')

# phase
deltaR_phase = all_deltaR[all_deltaR['code'] == 'phase']
ax1 = sns.catplot(data=deltaR_phase, kind='bar', x='tuning', y='mean deltaR',
                 hue='shuffling', ci='sd', capsize=0.2, errwidth=2)
ax1 = sns.swarmplot(data=deltaR_phase, x='tuning', y='mean deltaR',
                   hue='shuffling', color='black', dodge=True)
ax1.get_legend().set_visible(False)
ax1.set_title('Phase code mean delta R for different tuned networks')



    
