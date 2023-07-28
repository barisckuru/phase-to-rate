"""
Step 2 pattern separation analysis with Pearson's R.

Load the raw data, extract phase and rate code and calculate
Pearson's R between pairs of input and output patterns.
The results are stored in a .csv file. These results are necessary
to reproduce Figures 1 & 2.
"""

from phase_to_rate.neural_coding import load_spikes, rate_n_phase
from phase_to_rate.figure_functions import _make_cmap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import pearsonr,  spearmanr
import copy
from scipy import stats
from matplotlib.colors import SymLogNorm, PowerNorm, Normalize
import copy
import os
import matplotlib as mpl
import pdb

dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')

#load data, codes
trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

n_samples = 20

grid_seeds = np.arange(1,11,1)
tuning = 'disinhibited'
all_codes = {}

for grid_seed in grid_seeds:
    path = os.path.join(results_dir, 'main', str(tuning),  'collective', 'grid-seed_duration_shuffling_tuning_')
    
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


# with open(f'neural_codes_{tuning}.pkl', 'wb') as handle:
#     pickle.dump(all_codes, handle)

   
# 75 vs all in all time bins
# calculate pearson R             
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
                        ndp = np.dot(baseline_traj, compared_traj) / (np.sqrt(np.dot(baseline_traj, baseline_traj)) * np.sqrt(np.dot(compared_traj, compared_traj)))  # TODO
                        pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                        traj = trajectories[traj_idx]
                        idx = 75 - traj
                        comp_trajectories = str(75)+'_'+str(traj)
                        r_data_sing = [idx, ndp, pearson_r, poisson,
                                     comp_trajectories, grid_seed, shuffling,
                                     cell, code]
                        r_data.append(copy.deepcopy(r_data_sing))


data = copy.deepcopy(r_data)
new_data = []
for d in data:
    new_data
                    
                        
# =============================================================================
# plotting
# =============================================================================

sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)

my_pal = {'grid': '#716969', 'granule': '#09316c'}
color_list_1 = ["0a2d27","13594e","1d8676","26b29d","30dfc4","59e5d0","83ecdc"]
color_list_2 = ["572800","ab5100","ff7900","ff9637","ffb570"]
my_cmap = _make_cmap(color_list_1)
my_cmap_2 = _make_cmap(color_list_2)

df = pd.DataFrame(r_data,
                  columns=['distance', 'ndp', 'pearson_r',
                           'poisson_seed', 'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')


shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'rate') &
                             (df['shuffling'] == 'shuffled')]
                             [['distance', 'grid_seed',
                               'ndp']].reset_index(drop=True))
shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'rate') & 
                                (df['shuffling'] == 'shuffled')]
                                ['ndp'].reset_index(drop=True))
shuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                              (df['code_type'] == 'phase') &
                              (df['shuffling'] == 'shuffled')]
                              ['ndp'].reset_index(drop=True))
shuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'shuffled')]
                                  ['ndp'].reset_index(drop=True))
nonshuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                (df['code_type'] == 'rate') &
                                (df['shuffling'] == 'non-shuffled')]
                             ['ndp'].reset_index(drop=True))
nonshuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                   (df['code_type'] == 'rate') &
                                   (df['shuffling'] == 'non-shuffled')]
                                    ['ndp'].reset_index(drop=True))
nonshuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'non-shuffled')]
                                 ['ndp'].reset_index(drop=True))
nonshuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                    (df['code_type'] == 'phase') &
                                    (df['shuffling'] == 'non-shuffled')]
                                    ['ndp'].reset_index(drop=True))

ndp = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    nonshuffled_grid_rate, nonshuffled_granule_rate,
    shuffled_grid_phase, shuffled_granule_phase,
    nonshuffled_grid_phase, nonshuffled_granule_phase], axis=1)
ndp.columns = ['distance', 'grid_seed', 
                            's_grid_rate', 's_granule_rate',
                            'ns_grid_rate', 'ns_granule_rate',
                            's_grid_phase', 's_granule_phase',
                            'ns_grid_phase', 'ns_granule_phase'
                            ]

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "NDP \n all bins \n grid seed = 1")

ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")


hue = list(ndp['distance'])

sns.scatterplot(ax=ax1,
              data=ndp, x="s_grid_rate", y="s_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap, s=1, linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax2,
              data=ndp, x="ns_grid_rate", y="ns_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap, s=1, linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax3,
              data=ndp, x="s_grid_phase", y="s_granule_phase",
              hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap, s=1, linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax4,
              data=ndp, x="ns_grid_phase", y="ns_granule_phase",
              hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap, s=1, linewidth=0.1, alpha=0.8)

for ax in fig2.axes:
    ax.get_legend().remove()
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    ax.set_xlim(-0.10,0.7)
    ax.set_ylim(-0.15,0.7)
    # ax.figure.colorbar(sm)

s_rate = stats.binned_statistic(ndp['s_grid_rate'],
                                list((ndp['s_grid_rate'],
                                      ndp['s_granule_rate'])), 'mean',
                                bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
s_phase = stats.binned_statistic(ndp['s_grid_phase'],
                                 list((ndp['s_grid_phase'],
                                       ndp['s_granule_phase'])), 'mean',
                                 bins=[0,0.1,0.2,0.3,0.4])
ns_rate = stats.binned_statistic(ndp['ns_grid_rate'],
                                 list((ndp['ns_grid_rate'],
                                       ndp['ns_granule_rate'])), 'mean',
                                 bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ns_phase = stats.binned_statistic(ndp['ns_grid_phase'],
                                  list((ndp['ns_grid_phase'],
                                        ndp['ns_granule_phase'])),
                                  'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])

ax1.plot(s_rate[0][0], s_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax2.plot(ns_rate[0][0], ns_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)
ax3.plot(s_phase[0][0], s_phase[0][1], 'k', linestyle=(0, (6, 1)), linewidth=2)
ax4.plot(ns_phase[0][0], ns_phase[0][1], 'k', linestyle=(0, (6, 2)), linewidth=2)


plt.tight_layout()





# =============================================================================
# mean delta R
# =============================================================================

import copy
data = copy.deepcopy(r_data)

df = pd.DataFrame(r_data,
                  columns=['distance', 'ndp', 'pearson_r',
                           'poisson_seed', 'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')


shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'rate') &
                             (df['shuffling'] == 'shuffled')]
                             [['distance', 'grid_seed',
                               'ndp']].reset_index(drop=True))
shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'rate') & 
                                (df['shuffling'] == 'shuffled')]
                                ['ndp'].reset_index(drop=True))
shuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                              (df['code_type'] == 'phase') &
                              (df['shuffling'] == 'shuffled')]
                              ['ndp'].reset_index(drop=True))
shuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'shuffled')]
                                  ['ndp'].reset_index(drop=True))
nonshuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                                (df['code_type'] == 'rate') &
                                (df['shuffling'] == 'non-shuffled')]
                             ['ndp'].reset_index(drop=True))
nonshuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                   (df['code_type'] == 'rate') &
                                   (df['shuffling'] == 'non-shuffled')]
                                    ['ndp'].reset_index(drop=True))
nonshuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'non-shuffled')]
                                 ['ndp'].reset_index(drop=True))
nonshuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                    (df['code_type'] == 'phase') &
                                    (df['shuffling'] == 'non-shuffled')]
                                    ['ndp'].reset_index(drop=True))

ndp = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    nonshuffled_grid_rate, nonshuffled_granule_rate,
    shuffled_grid_phase, shuffled_granule_phase,
    nonshuffled_grid_phase, nonshuffled_granule_phase], axis=1)
ndp.columns = ['distance', 'grid_seed', 
                            's_grid_rate', 's_granule_rate',
                            'ns_grid_rate', 'ns_granule_rate',
                            's_grid_phase', 's_granule_phase',
                            'ns_grid_phase', 'ns_granule_phase'
                            ]

delta_s_rate = []
delta_ns_rate = []
delta_s_phase = []
delta_ns_phase = []
for seed in grid_seeds:
    grid_1 = ndp.loc[(ndp['grid_seed'] == seed)]

    s_rate = stats.binned_statistic(grid_1['s_grid_rate'], list((grid_1['s_grid_rate'], grid_1['s_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8, 0.9])
    s_phase = stats.binned_statistic(grid_1['s_grid_phase'], list((grid_1['s_grid_phase'], grid_1['s_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])
    ns_rate = stats.binned_statistic(grid_1['ns_grid_rate'], list((grid_1['ns_grid_rate'], grid_1['ns_granule_rate'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    ns_phase = stats.binned_statistic(grid_1['ns_grid_phase'], list((grid_1['ns_grid_phase'], grid_1['ns_granule_phase'])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6])

    delta_s_rate.append(np.mean((s_rate[0][0] - s_rate[0][1])[s_rate[0][0]==s_rate[0][0]]))
    delta_ns_rate.append(np.mean((ns_rate[0][0] - ns_rate[0][1])[ns_rate[0][0]==ns_rate[0][0]]))
    delta_s_phase.append(np.mean((s_phase[0][0] - s_phase[0][1])[s_phase[0][0]==s_phase[0][0]]))
    delta_ns_phase.append(np.mean((ns_phase[0][0] - ns_phase[0][1])[ns_phase[0][0]==ns_phase[0][0]]))


plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update({'font.size': 32})


# =============================================================================
# only nonshuffled for figure 1K - mean delta R bargraph
# =============================================================================

ns_deltaR = np.concatenate((delta_ns_rate, delta_ns_phase))
shuffling = 20*['non-shuffled']
code = 10*['rate']+10*['phase']

ns_deltaR = np.stack((ns_deltaR, shuffling, code), axis=1)

ns_df_deltaR = pd.DataFrame(ns_deltaR, columns=['mean deltaR', 'shuffling', 'code'])
ns_df_deltaR['mean deltaR'] = ns_df_deltaR['mean deltaR'].astype('float')
# sns.barplot(data=ns_df_deltaR, x='code', y='mean deltaR',
#             ci='sd', capsize=0.2, errwidth=2)


ax = sns.catplot(data=ns_df_deltaR, kind='bar', x='code', y='mean deltaR',
                 ci='sd', capsize=0.2, errwidth=2)
ax = sns.swarmplot(data=ns_df_deltaR, x='code', y='mean deltaR',
                   color='black', dodge=True)
# ax.get_legend().set_visible(False)

plt.tight_layout()




# =============================================================================
# shuffled and nonshuffled for figure 1K - mean delta R bargraph
# =============================================================================


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



with pd.ExcelWriter('mean_deltaR_2000ms_75vsall_NDP_inactives_deleted.xlsx') as writer:
    deltaR_rate.to_excel(writer, sheet_name='Rate code')
    deltaR_phase.to_excel(writer, sheet_name='Phase code')

"""Pearson R vs NDP"""
(df['shuffling'] == 'non-shuffled') & (df['code_type'] == 'rate')

sns.set(font_scale=1)

sns.scatterplot(data=df[(df['shuffling'] == 'non-shuffled') & (df['code_type'] == 'phase')], x="ndp", y="pearson_r",
              hue='cell_type', s=2, linewidth=0.1, alpha=0.8)
# plt.plot([0,1], [0,1], c='k')
plt.xlim((0,0.7))
plt.ylim((-0.2,0.6))

    

