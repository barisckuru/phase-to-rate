#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:55:20 2021

@author: baris
"""

from neural_coding import load_spikes, rate_n_phase
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import pearsonr,  spearmanr
import copy
from scipy import stats
from matplotlib.colors import SymLogNorm, PowerNorm, Normalize


#load data, codes

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
n_samples = 20
grid_seeds = np.arange(1,11,1)
tuning = 'full'
all_codes = {}
for grid_seed in grid_seeds:
    path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
    
    # non-shuffled
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    
    print('ns path ok')
    
    (
        grid_counts,
        grid_phases,
        grid_rate_code,
        grid_phase_code,
        grid_polar_code,
    ) = rate_n_phase(grid_spikes, trajectories, n_samples)
    
    (
        granule_counts,
        granule_phases,
        granule_rate_code,
        granule_phase_code,
        granule_polar_code,
    ) = rate_n_phase(granule_spikes, trajectories, n_samples)
    
    # shuffled
    s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
    s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
    s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
    
    print('shuffled path ok')
    
    (
        s_grid_counts,
        s_grid_phases,
        s_grid_rate_code,
        s_grid_phase_code,
        s_grid_polar_code,
    ) = rate_n_phase(s_grid_spikes, trajectories, n_samples)
    
    (
        s_granule_counts,
        s_granule_phases,
        s_granule_rate_code,
        s_granule_phase_code,
        s_granule_polar_code,
    ) = rate_n_phase(s_granule_spikes, trajectories, n_samples)


    all_codes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
    all_codes[grid_seed]["shuffled"] = {"grid": {}, "granule": {}}
    all_codes[grid_seed]["non-shuffled"] = {"grid": {}, "granule": {}}

    all_codes[grid_seed]['non-shuffled']['grid'] = {'rate': grid_rate_code,
                      'phase': grid_phase_code}
    all_codes[grid_seed]['shuffled']['grid'] = {'rate': s_grid_rate_code,
                      'phase': s_grid_phase_code}
    all_codes[grid_seed]['non-shuffled']['granule'] = {'rate': granule_rate_code,
                      'phase': granule_phase_code}
    all_codes[grid_seed]['shuffled']['granule'] = {'rate': s_granule_rate_code,
                      'phase': s_granule_phase_code}





# 75 vs all in one time bin                      
r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for traj_idx in range(len(trajectories)):
                    r_input_code = all_codes[grid_seed][shuffling][cell][code]
                    f = int(r_input_code.shape[0]/8000)
                    baseline_traj = np.concatenate((
                        r_input_code[1000*f:1200*f, 0, 0], 
                        r_input_code[5000*f:5200*f, 0, 0]))
                    for poisson in range(r_input_code.shape[1]):
                        compared_traj = r_input_code[:, poisson, traj_idx]
                        compared_traj = np.concatenate((
                            compared_traj[1000*f:1200*f],
                            compared_traj[5000*f:5200*f]))
                        pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                        spearman_r = spearmanr(baseline_traj, compared_traj)[0]
                        traj = trajectories[traj_idx]
                        idx = 75 - traj
                        comp_trajectories = str(75)+'_'+str(traj)
                        r_data_sing = [idx, pearson_r, spearman_r, poisson,
                                     comp_trajectories, grid_seed, shuffling,
                                     cell, code]
                        r_data.append(copy.deepcopy(r_data_sing))
                        
# all vs all, one time bin
                        
                        
r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for base_traj_idx in range(len(trajectories)):
                        r_input_code = all_codes[grid_seed][shuffling][cell][code]
                        f = int(r_input_code.shape[0]/8000)
                        baseline_traj = np.concatenate((
                            r_input_code[1000*f:1200*f, 0, base_traj_idx], 
                            r_input_code[5000*f:5200*f, 0, base_traj_idx]))
                        for traj_idx in range(len(trajectories)):
                            for poisson in range(r_input_code.shape[1]):
                                compared_traj = r_input_code[:, poisson, traj_idx]
                                compared_traj = np.concatenate((
                                    compared_traj[1000*f:1200*f],
                                    compared_traj[5000*f:5200*f]))
                                pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                                spearman_r = spearmanr(baseline_traj, compared_traj)[0]
                                traj = trajectories[traj_idx]
                                base_traj = trajectories[base_traj_idx]
                                idx = abs(base_traj - traj)
                                comp_trajectories = str(base_traj)+'_'+str(traj)
                                r_data_sing = [idx, pearson_r, spearman_r, poisson,
                                             comp_trajectories, grid_seed, shuffling,
                                             cell, code]
                                r_data.append(copy.deepcopy(r_data_sing))
                        
   
# 75 vs all in all time bins                      
r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for traj_idx in range(len(trajectories)):
                    r_input_code = all_codes[grid_seed][shuffling][cell][code]
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
                        
# all vs all, all time bins
                        
                        
r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for base_traj_idx in range(len(trajectories)):
                        r_input_code = all_codes[grid_seed][shuffling][cell][code]
                        f = int(r_input_code.shape[0]/8000)
                        baseline_traj = r_input_code[:, 0, base_traj_idx]
                        for traj_idx in range(len(trajectories)):
                            for poisson in range(r_input_code.shape[1]):
                                compared_traj = r_input_code[:, poisson, traj_idx]
                                pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                                spearman_r = spearmanr(baseline_traj, compared_traj)[0]
                                traj = trajectories[traj_idx]
                                base_traj = trajectories[base_traj_idx]
                                idx = abs(base_traj - traj)
                                comp_trajectories = str(base_traj)+'_'+str(traj)
                                r_data_sing = [idx, pearson_r, spearman_r, poisson,
                                             comp_trajectories, grid_seed, shuffling,
                                             cell, code]
                                r_data.append(copy.deepcopy(r_data_sing))
                        
                        

   

# all vs all, on time bin

# =============================================================================
# plotting
# =============================================================================

df = pd.DataFrame(r_data,
                  columns=['distance', 'pearson_r','spearman_r', 'poisson_seed',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')

# df_seed = df.loc[(df['grid_seed'] == 1)]

shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'rate') &
                             (df['shuffling'] == 'shuffled')]# &
                             # (df['grid_seed'] == 1)]
                             [['distance', 'grid_seed',
                               'pearson_r']].reset_index(drop=True))
shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'rate') & 
                                (df['shuffling'] == 'shuffled')]
                                # (df['grid_seed'] == 1)]
                                ['pearson_r'].reset_index(drop=True))
shuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                              (df['code_type'] == 'phase') &
                              (df['shuffling'] == 'shuffled')]
                               # (df['grid_seed'] == 1)]
                              ['pearson_r'].reset_index(drop=True))
shuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'shuffled')]
                                 # (df['grid_seed'] == 1)]
                                  ['pearson_r'].reset_index(drop=True))
nonshuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                (df['code_type'] == 'rate') &
                                (df['shuffling'] == 'non-shuffled')]
                                # (df['grid_seed'] == 1)]
                             ['pearson_r'].reset_index(drop=True))
nonshuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                   (df['code_type'] == 'rate') &
                                   (df['shuffling'] == 'non-shuffled')]
                                   # (df['grid_seed'] == 1)]
                                    ['pearson_r'].reset_index(drop=True))
nonshuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'non-shuffled')]
                                 # (df['grid_seed'] == 1)]
                                 ['pearson_r'].reset_index(drop=True))
nonshuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                    (df['code_type'] == 'phase') &
                                    (df['shuffling'] == 'non-shuffled')]
                                    # (df['grid_seed'] == 1)]
                                    ['pearson_r'].reset_index(drop=True))



pearson_r = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    nonshuffled_grid_rate, nonshuffled_granule_rate,
    shuffled_grid_phase, shuffled_granule_phase,
    nonshuffled_grid_phase, nonshuffled_granule_phase], axis=1)
pearson_r.columns = ['distance', 'grid_seed', 
                            's_grid_rate', 's_granule_rate',
                            'ns_grid_rate', 'ns_granule_rate',
                            's_grid_phase', 's_granule_phase',
                            'ns_grid_phase', 'ns_granule_phase'
                            ]


fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "Pearson's R \n all bins \n grid seed = 1")

ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")


hue = list(pearson_r['distance'])

sns.set()
sns.scatterplot(ax=ax1,
              data=pearson_r, x="s_grid_rate", y="s_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)
sns.scatterplot(ax=ax2,
              data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)
sns.scatterplot(ax=ax3,
              data=pearson_r, x="s_grid_phase", y="s_granule_phase",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)
sns.scatterplot(ax=ax4,
              data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)


for ax in fig2.axes:
    ax.get_legend().remove()
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    ax.set_xlim(-0.40,0.8)
    ax.set_ylim(-0.15,0.6)
    # ax.figure.colorbar(sm)

s_rate = stats.binned_statistic(pearson_r['s_grid_rate'], list((pearson_r['s_grid_rate'], pearson_r['s_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
s_phase = stats.binned_statistic(pearson_r['s_grid_phase'], list((pearson_r['s_grid_phase'], pearson_r['s_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4])
ns_rate = stats.binned_statistic(pearson_r['ns_grid_rate'], list((pearson_r['ns_grid_rate'], pearson_r['ns_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ns_phase = stats.binned_statistic(pearson_r['ns_grid_phase'], list((pearson_r['ns_grid_phase'], pearson_r['ns_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])

ax1.plot(s_rate[0][0], s_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax2.plot(ns_rate[0][0], ns_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax3.plot(s_phase[0][0], s_phase[0][1], 'k', linestyle=(0, (6, 1)), linewidth=2)
ax4.plot(ns_phase[0][0], ns_phase[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)


plt.tight_layout()





# =============================================================================
# mean delta R
# =============================================================================

delta_s_rate = []
delta_ns_rate = []
delta_s_phase = []
delta_ns_phase = []
for seed in grid_seeds:
    pearson_r.loc[(pearson_r['grid_seed'] == 1)]
    

    grid_1 = pearson_r.loc[(pearson_r['grid_seed'] == seed)]

    s_rate = stats.binned_statistic(grid_1['s_grid_rate'], list((grid_1['s_grid_rate'], grid_1['s_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9])
    s_phase = stats.binned_statistic(grid_1['s_grid_phase'], list((grid_1['s_grid_phase'], grid_1['s_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])
    ns_rate = stats.binned_statistic(grid_1['ns_grid_rate'], list((grid_1['ns_grid_rate'], grid_1['ns_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    ns_phase = stats.binned_statistic(grid_1['ns_grid_phase'], list((grid_1['ns_grid_phase'], grid_1['ns_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6])

    delta_s_rate.append(np.mean((s_rate[0][0] - s_rate[0][1])[s_rate[0][0]==s_rate[0][0]]))
    delta_ns_rate.append(np.mean((ns_rate[0][0] - ns_rate[0][1])[ns_rate[0][0]==ns_rate[0][0]]))
    delta_s_phase.append(np.mean((s_phase[0][0] - s_phase[0][1])[s_phase[0][0]==s_phase[0][0]]))
    delta_ns_phase.append(np.mean((ns_phase[0][0] - ns_phase[0][1])[ns_phase[0][0]==ns_phase[0][0]]))


deltaR = np.concatenate((delta_ns_rate, delta_s_rate,
                          delta_ns_phase, delta_s_phase))
shuffling = 2*(10*['non-shuffled'] + 10*['shuffled'])
code = 20*['rate']+20*['phase']

deltaR = np.stack((deltaR, shuffling, code), axis=1)

df_deltaR = pd.DataFrame(deltaR, columns=['mean deltaR', 'shuffling', 'code'])
df_deltaR['mean deltaR'] = df_deltaR['mean deltaR'].astype('float')
sns.barplot(data=df_deltaR, x='code', y='mean deltaR', hue='shuffling',
            ci='sd', capsize=0.2, errwidth=2)


ax = sns.catplot(data=df_deltaR, kind='bar', x='code', y='mean deltaR',
                 hue='shuffling', ci='sd', capsize=0.2, errwidth=2)
ax = sns.swarmplot(data=df_deltaR, x='code', y='mean deltaR',
                   hue='shuffling', color='black', dodge=True)
ax.get_legend().set_visible(False)

plt.tight_layout()




















delta_s_rate = np.mean((s_rate[0][0] - s_rate[0][1])[:6])
delta_ns_rate = np.mean((ns_rate[0][0] - ns_rate[0][1])[:6])
delta_s_phase = np.mean((s_phase[0][0] - s_phase[0][1])[:3])
delta_ns_phase = np.mean((ns_phase[0][0] - ns_phase[0][1])[:2])

vals = [delta_ns_rate, delta_s_rate, delta_ns_phase, delta_s_phase]
labels = ['nonshuffled rate', 'shuffled rate', 'nonshuffled phase', 'shuffled phase']

sns.reset_orig()
fig, ax = plt.subplots(figsize = (10,4))
ax.set_title('mean $\u0394R_{out}$')
ind = np.arange(4)
ax.bar(ind, vals)
ax.set_xticks(ind)
ax.set_xticklabels(labels)
plt.tight_layout()

plt.show()




#load data spike times

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

n_samples = 20
grid_seeds = np.arange(1,11,1)
tuning = 'full'
bin_size = 2000
all_codes = {}
gra_spikes = []

for grid_seed in grid_seeds:
    path = "/home/baris/results/"+str(tuning)+"/collective/grid-seed_duration_shuffling_tuning_"
    
    # non-shuffled
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    grid_spikes = load_spikes(ns_path, "grid", trajectories, n_samples)
    granule_spikes = load_spikes(ns_path, "granule", trajectories, n_samples)
    gra_spikes.append(granule_spikes)
    
    print('ns path ok')
    
    (
        grid_counts,
        grid_phases,
        grid_rate_code,
        grid_phase_code,
        grid_polar_code,
    ) = rate_n_phase(grid_spikes, trajectories, n_samples, bin_size_ms=bin_size)
    
    (
        granule_counts,
        granule_phases,
        granule_rate_code,
        granule_phase_code,
        granule_polar_code,
    ) = rate_n_phase(granule_spikes, trajectories, n_samples, bin_size_ms=bin_size)
    
    
    
    
    # shuffled
    s_path = (path + str(grid_seed) + "_2000_shuffled_"+str(tuning))
    s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
    s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)
    
    print('shuffled path ok')
    
    (
        s_grid_counts,
        s_grid_phases,
        s_grid_rate_code,
        s_grid_phase_code,
        s_grid_polar_code,
    ) = rate_n_phase(s_grid_spikes, trajectories, n_samples, bin_size_ms=bin_size)
    
    (
        s_granule_counts,
        s_granule_phases,
        s_granule_rate_code,
        s_granule_phase_code,
        s_granule_polar_code,
    ) = rate_n_phase(s_granule_spikes, trajectories, n_samples, bin_size_ms=bin_size)
    

    all_codes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
    all_codes[grid_seed]["shuffled"] = {"grid": {}, "granule": {}}
    all_codes[grid_seed]["non-shuffled"] = {"grid": {}, "granule": {}}

    all_codes[grid_seed]['non-shuffled']['grid'] = {'rate': grid_counts}
    all_codes[grid_seed]['shuffled']['grid'] = {'rate': s_grid_counts}
    all_codes[grid_seed]['non-shuffled']['granule'] = {'rate': granule_counts}
    all_codes[grid_seed]['shuffled']['granule'] = {'rate': s_granule_counts}







ns_grid_count = np.stack((grid_counts[:,0,0,0], grid_counts[:,0,0,2],grid_counts[:,0,0,16]))
ns_granule_count = np.stack((granule_counts[:,0,0,0], granule_counts[:,0,0,2],granule_counts[:,0,0,16]))
s_grid_count = np.stack((s_grid_counts[:,0,0,0], s_grid_counts[:,0,0,2],s_grid_counts[:,0,0,16]))
s_granule_count = np.stack((s_granule_counts[:,0,0,0], s_granule_counts[:,0,0,2],s_granule_counts[:,0,0,16]))


import pandas as pd

df1 = pd.DataFrame(ns_grid_count.T, columns=[75, 74, 15])
df1.to_excel('ns_grid_count'+'2000ms'+'.xlsx', sheet_name='ns_grid_count')

df2 = pd.DataFrame(s_grid_count.T, columns=[75, 74, 15])
df2.to_excel('s_grid_count'+'2000ms'+'.xlsx', sheet_name='s_grid_count')

df3 = pd.DataFrame(ns_granule_count.T, columns=[75, 74, 15])
df3.to_excel('ns_granule_count'+'2000ms'+'.xlsx', sheet_name='ns_granule_count')

df4 = pd.DataFrame(s_granule_count.T, columns=[75, 74, 15])
df4.to_excel('s_granule_count'+'2000ms'+'.xlsx', sheet_name='s_granule_count')



# =============================================================================
# no codes, rates compared
# =============================================================================

r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for traj_idx in range(len(trajectories)):
                    r_input_code = all_codes[grid_seed][shuffling][cell][code]
                    r_input_code= r_input_code.reshape(r_input_code.shape[0],
                                         r_input_code.shape[2],
                                         r_input_code.shape[3])
                    baseline_traj = r_input_code[:, 0, 0]
                    for poisson in range(r_input_code.shape[2]):
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


                     
# =============================================================================
# plotting
# =============================================================================

df = pd.DataFrame(r_data,
                  columns=['distance', 'pearson_r','spearman_r', 'poisson_seed',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')

df_seed = df.loc[(df['grid_seed'] == 1)]

shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'rate') &
                             (df['shuffling'] == 'shuffled')]# &
                             # (df['grid_seed'] == 1)]
                             [['distance', 'grid_seed',
                               'pearson_r']].reset_index(drop=True))
shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'rate') & 
                                (df['shuffling'] == 'shuffled')]
                                # (df['grid_seed'] == 1)]
                                ['pearson_r'].reset_index(drop=True))

nonshuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                (df['code_type'] == 'rate') &
                                (df['shuffling'] == 'non-shuffled')]
                                # (df['grid_seed'] == 1)]
                                ['pearson_r'].reset_index(drop=True))
nonshuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                   (df['code_type'] == 'rate') &
                                   (df['shuffling'] == 'non-shuffled')]
                                   # (df['grid_seed'] == 1)]
                                    ['pearson_r'].reset_index(drop=True))



pearson_r = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    nonshuffled_grid_rate, nonshuffled_granule_rate], axis=1)
pearson_r.columns = ['distance', 'grid_seed', 
                            's_grid_rate', 's_granule_rate',
                            'ns_grid_rate', 'ns_granule_rate'
                            ]


fig2, ((ax1, ax2)) = plt.subplots(1, 2)
fig2.suptitle(
    "Pearson's R \n all bins \n grid seed = 1")

ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")


hue = list(pearson_r['distance'])

sns.set()
sns.scatterplot(ax=ax1,
              data=pearson_r, x="s_grid_rate", y="s_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)
sns.scatterplot(ax=ax2,
              data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)

# Remove the legend and add a unity line
for ax in fig2.axes:
    ax.get_legend().remove()
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    # ax.set_xlim(-0.25,0.6)
    # ax.set_ylim(-0.15,0.6)
    # ax.figure.colorbar(sm)

s_rate = stats.binned_statistic(pearson_r['s_grid_rate'], list((pearson_r['s_grid_rate'], pearson_r['s_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ns_rate = stats.binned_statistic(pearson_r['ns_grid_rate'], list((pearson_r['ns_grid_rate'], pearson_r['ns_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])

ax1.plot(s_rate[0][0], s_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax2.plot(ns_rate[0][0], ns_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
plt.tight_layout()











# =============================================================================
# all bins, 75 vs all
# =============================================================================


r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for traj_idx in range(len(trajectories)):
                    r_input_code = all_codes[grid_seed][shuffling][cell][code]
                    f = int(r_input_code.shape[0]/8000)
                    baseline_traj = r_input_code[:, 0, 0]
                    for poisson in range(r_input_code.shape[2]):
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
                        
# =============================================================================
# all bins, all vs all                        
# =============================================================================
               

            
r_data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for base_traj_idx in range(len(trajectories)):
                        r_input_code = all_codes[grid_seed][shuffling][cell][code]
                        f = int(r_input_code.shape[0]/8000)
                        baseline_traj = r_input_code[:, 0, base_traj_idx]
                        for traj_idx in range(len(trajectories)):
                            for poisson in range(r_input_code.shape[1]):
                                compared_traj = r_input_code[:, poisson, traj_idx]
                                pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                                spearman_r = spearmanr(baseline_traj, compared_traj)[0]
                                traj = trajectories[traj_idx]
                                base_traj = trajectories[base_traj_idx]
                                idx = abs(base_traj - traj)
                                comp_trajectories = str(base_traj)+'_'+str(traj)
                                r_data_sing = [idx, pearson_r, spearman_r, poisson,
                                             comp_trajectories, grid_seed, shuffling,
                                             cell, code]
                                r_data.append(copy.deepcopy(r_data_sing))

# =============================================================================
# plotting
# =============================================================================

df = pd.DataFrame(r_data,
                  columns=['distance', 'pearson_r','spearman_r', 'poisson_seed',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')

df_seed = df.loc[(df['grid_seed'] == 1)]

shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'rate') &
                             (df['shuffling'] == 'shuffled')]# &
                             # (df['grid_seed'] == 1)]
                             [['distance', 'grid_seed',
                               'pearson_r']].reset_index(drop=True))
shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'rate') & 
                                (df['shuffling'] == 'shuffled')]
                                # (df['grid_seed'] == 1)]
                                ['pearson_r'].reset_index(drop=True))
shuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                              (df['code_type'] == 'phase') &
                              (df['shuffling'] == 'shuffled')]
                               # (df['grid_seed'] == 1)]
                              ['pearson_r'].reset_index(drop=True))
shuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'shuffled')]
                                 # (df['grid_seed'] == 1)]
                                  ['pearson_r'].reset_index(drop=True))
nonshuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                (df['code_type'] == 'rate') &
                                (df['shuffling'] == 'non-shuffled')]
                                # (df['grid_seed'] == 1)]
                             ['pearson_r'].reset_index(drop=True))
nonshuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                   (df['code_type'] == 'rate') &
                                   (df['shuffling'] == 'non-shuffled')]
                                   # (df['grid_seed'] == 1)]
                                    ['pearson_r'].reset_index(drop=True))
nonshuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'non-shuffled')]
                                 # (df['grid_seed'] == 1)]
                                 ['pearson_r'].reset_index(drop=True))
nonshuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                    (df['code_type'] == 'phase') &
                                    (df['shuffling'] == 'non-shuffled')]
                                    # (df['grid_seed'] == 1)]
                                    ['pearson_r'].reset_index(drop=True))



pearson_r = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    nonshuffled_grid_rate, nonshuffled_granule_rate,
    shuffled_grid_phase, shuffled_granule_phase,
    nonshuffled_grid_phase, nonshuffled_granule_phase], axis=1)
pearson_r.columns = ['distance', 'grid_seed', 
                            's_grid_rate', 's_granule_rate',
                            'ns_grid_rate', 'ns_granule_rate',
                            's_grid_phase', 's_granule_phase',
                            'ns_grid_phase', 'ns_granule_phase'
                            ]


fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "Pearson's R \n all bins \n grid seed = 1")

ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")


hue = list(pearson_r['distance'])

sns.set()
sns.scatterplot(ax=ax1,
              data=pearson_r, x="s_grid_rate", y="s_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)
sns.scatterplot(ax=ax2,
              data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)
sns.scatterplot(ax=ax3,
              data=pearson_r, x="s_grid_phase", y="s_granule_phase",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)
sns.scatterplot(ax=ax4,
              data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
              hue=hue, hue_norm=SymLogNorm(10), palette='OrRd', s=25)#



# sns.scatterplot(ax=ax1,
#              data=pearson_r, x="s_grid_rate", y="s_granule_rate",
#              hue=hue, palette='OrRd', s=10)
# sns.scatterplot(ax=ax2,
#              data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
#              hue=hue, palette='OrRd', s=10)
# sns.scatterplot(ax=ax3,
#              data=pearson_r, x="s_grid_phase", y="s_granule_phase",
#              hue=hue, palette='OrRd', s=10)
# sns.scatterplot(ax=ax4,
#              data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
#              hue=hue, palette='OrRd', s=10)


# Remove the legend and add a unity line
for ax in fig2.axes:
    ax.get_legend().remove()
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    ax.set_xlim(-0.25,0.6)
    ax.set_ylim(-0.15,0.6)
    # ax.figure.colorbar(sm)

s_rate = stats.binned_statistic(pearson_r['s_grid_rate'], list((pearson_r['s_grid_rate'], pearson_r['s_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
s_phase = stats.binned_statistic(pearson_r['s_grid_phase'], list((pearson_r['s_grid_phase'], pearson_r['s_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4])
ns_rate = stats.binned_statistic(pearson_r['ns_grid_rate'], list((pearson_r['ns_grid_rate'], pearson_r['ns_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ns_phase = stats.binned_statistic(pearson_r['ns_grid_phase'], list((pearson_r['ns_grid_phase'], pearson_r['ns_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])

ax1.plot(s_rate[0][0], s_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax2.plot(ns_rate[0][0], ns_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax3.plot(s_phase[0][0], s_phase[0][1], 'k', linestyle=(0, (6, 1)), linewidth=2)
ax4.plot(ns_phase[0][0], ns_phase[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)


plt.tight_layout()

delta_s_rate = np.mean((s_rate[0][0] - s_rate[0][1])[:6])
delta_ns_rate = np.mean((ns_rate[0][0] - ns_rate[0][1])[:6])
delta_s_phase = np.mean((s_phase[0][0] - s_phase[0][1])[:3])
delta_ns_phase = np.mean((ns_phase[0][0] - ns_phase[0][1])[:2])

vals = [delta_ns_rate, delta_s_rate, delta_ns_phase, delta_s_phase]
labels = ['nonshuffled rate', 'shuffled rate', 'nonshuffled phase', 'shuffled phase']

sns.reset_orig()
fig, ax = plt.subplots(figsize = (10,4))
ax.set_title('mean $\u0394R_{out}$')
ind = np.arange(4)
ax.bar(ind, vals)
ax.set_xticks(ind)
ax.set_xticklabels(labels)
plt.tight_layout()

plt.show()

# =============================================================================
# spearmans rho
# =============================================================================


shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'rate') &
                             (df['shuffling'] == 'shuffled') &
                             (df['grid_seed'] == 1)]
                             [['distance', 'grid_seed',
                               'spearman_r']].reset_index(drop=True))
shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'rate') & 
                                (df['shuffling'] == 'shuffled') &
                                (df['grid_seed'] == 1)]
                                ['spearman_r'].reset_index(drop=True))
shuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                              (df['code_type'] == 'phase') &
                              (df['shuffling'] == 'shuffled') &
                              (df['grid_seed'] == 1)]
                              ['spearman_r'].reset_index(drop=True))
shuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'shuffled') &
                                 (df['grid_seed'] == 1)]
                                  ['spearman_r'].reset_index(drop=True))
nonshuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                (df['code_type'] == 'rate') &
                                (df['shuffling'] == 'non-shuffled') &
                                (df['grid_seed'] == 1)]
                             ['spearman_r'].reset_index(drop=True))
nonshuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                   (df['code_type'] == 'rate') &
                                   (df['shuffling'] == 'non-shuffled') &
                                    (df['grid_seed'] == 1)]
                                    ['spearman_r'].reset_index(drop=True))
nonshuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'non-shuffled') &
                                  (df['grid_seed'] == 1)]
                                 ['spearman_r'].reset_index(drop=True))
nonshuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                    (df['code_type'] == 'phase') &
                                    (df['shuffling'] == 'non-shuffled') &
                                    (df['grid_seed'] == 1)]
                                    ['spearman_r'].reset_index(drop=True))

spearman_r = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    nonshuffled_grid_rate, nonshuffled_granule_rate,
    shuffled_grid_phase, shuffled_granule_phase,
    nonshuffled_grid_phase, nonshuffled_granule_phase], axis=1)
spearman_r.columns = ['distance', 'grid_seed', 
                            's_grid_rate', 's_granule_rate',
                            'ns_grid_rate', 'ns_granule_rate',
                            's_grid_phase', 's_granule_phase',
                            'ns_grid_phase', 'ns_granule_phase'
                            ]


fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "Spearman's rho \n 1 grid seed")
from matplotlib.colors import SymLogNorm, PowerNorm, Normalize
ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")


hue = list(spearman_r['distance'])

sns.set()
sns.scatterplot(ax=ax1,
             data=spearman_r, x="s_grid_rate", y="s_granule_rate",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=25)
sns.scatterplot(ax=ax2,
             data=spearman_r, x="ns_grid_rate", y="ns_granule_rate",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=25)
sns.scatterplot(ax=ax3,
             data=spearman_r, x="s_grid_phase", y="s_granule_phase",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=25
             )
sns.scatterplot(ax=ax4,
             data=spearman_r, x="ns_grid_phase", y="ns_granule_phase",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=25
             )
# Remove the legend and add a unity line
for ax in fig2.axes:
    ax.get_legend().remove()
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    ax.set_xlim(-0.25,0.6)
    ax.set_ylim(-0.15,0.6)
    # ax.figure.colorbar(sm)

s_rate = stats.binned_statistic(spearman_r['s_grid_rate'], list((spearman_r['s_grid_rate'], spearman_r['s_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
s_phase = stats.binned_statistic(spearman_r['s_grid_phase'], list((spearman_r['s_grid_phase'], spearman_r['s_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4])
ns_rate = stats.binned_statistic(spearman_r['ns_grid_rate'], list((spearman_r['ns_grid_rate'], spearman_r['ns_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ns_phase = stats.binned_statistic(spearman_r['ns_grid_phase'], list((spearman_r['ns_grid_phase'], spearman_r['ns_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])

ax1.plot(s_rate[0][0], s_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax2.plot(ns_rate[0][0], ns_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax3.plot(s_phase[0][0], s_phase[0][1], 'k', linestyle=(0, (6, 1)), linewidth=2)
ax4.plot(ns_phase[0][0], ns_phase[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)


plt.tight_layout()

# =============================================================================
# =============================================================================
# # 'spearman rho'
# =============================================================================
# =============================================================================



