"""
Step 2 pattern separation analysis with Pearson's R.

Load the raw data, extract phase and rate code and calculate
Pearson's R between pairs of input and output patterns.
The results are stored in a .csv file. These results are necessary
to reproduce Figures 1 & 2.
"""
from phase_to_rate.neural_coding import load_spikes, load_spikes_DMK, load_spikes_DMK_plus_lec, rate_n_phase
from phase_to_rate.figure_functions import (_make_cmap, _precession_spikes,
                              _adjust_box_widths, _adjust_bar_widths)
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
import sys
import matplotlib as mpl

#load data, codes
trajectories = [75, 74.5, 74, 73.5, 73]
n_samples = 10
grid_seeds = np.arange(340,350,1)
tuning = 'full'
pp_strength = 0.0009

all_codes = {}
for grid_seed in grid_seeds:
    path = r'C:\Users\Daniel\repos\phase-to-rate\data\noise_lec_identical\grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_pp-weight_noise-scale_'

    # non-shuffled
    # ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    ns_path = (path + f"{grid_seed}_[75, 74.5, 74, 73.5, 73]_100-109_2000_non-shuffled_"+str(tuning)+f"_{pp_strength}_200")
    
    grid_spikes = load_spikes_DMK_plus_lec(ns_path, "grid", trajectories, n_samples)
    
    granule_spikes = load_spikes_DMK(ns_path, "granule", trajectories, n_samples)
    
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
    s_path = (path + f"{grid_seed}_[75, 74.5, 74, 73.5, 73]_100-109_2000_shuffled_"+str(tuning)+f"_{pp_strength}_200")
    s_grid_spikes = load_spikes_DMK_plus_lec(s_path, "grid", trajectories, n_samples)
    s_granule_spikes = load_spikes_DMK(s_path, "granule", trajectories, n_samples)
    
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
                        pearson_r = pearsonr(baseline_traj, compared_traj)[0]
                        spearman_r = spearmanr(baseline_traj, compared_traj)[0]
                        traj = trajectories[traj_idx]
                        idx = 75 - traj
                        comp_trajectories = str(75)+'_'+str(traj)
                        r_data_sing = [idx, pearson_r, spearman_r, poisson,
                                     comp_trajectories, grid_seed, shuffling,
                                     cell, code, poisson]
                        r_data.append(copy.deepcopy(r_data_sing))

data = copy.deepcopy(r_data)
new_data = []
for d in data:
    new_data

# =============================================================================
# plotting
# =============================================================================
df = pd.DataFrame(r_data,
                  columns=['distance', 'pearson_r', 'spearman_r',
                           'poisson_seed', 'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'poisson'])

df = df.drop(columns='trajectories')

shuffled_grid_rate = (df.loc[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'rate') &
                             (df['shuffling'] == 'shuffled')]
                             [['distance', 'grid_seed', 'poisson', 
                               'pearson_r', ]].reset_index(drop=True))
shuffled_granule_rate = (df.loc[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'rate') & 
                                (df['shuffling'] == 'shuffled')]
                                ['pearson_r'].reset_index(drop=True))
shuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                              (df['code_type'] == 'phase') &
                              (df['shuffling'] == 'shuffled')]
                              ['pearson_r'].reset_index(drop=True))
shuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                 (df['code_type'] == 'phase') &
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
nonshuffled_grid_phase = (df.loc[(df['cell_type'] == 'grid') &
                                 (df['code_type'] == 'phase') &
                                 (df['shuffling'] == 'non-shuffled')]
                                 ['pearson_r'].reset_index(drop=True))
nonshuffled_granule_phase = (df.loc[(df['cell_type'] == 'granule') &
                                    (df['code_type'] == 'phase') &
                                    (df['shuffling'] == 'non-shuffled')]
                                    ['pearson_r'].reset_index(drop=True))

pearson_r = pd.concat([
    shuffled_grid_rate, shuffled_granule_rate,
    nonshuffled_grid_rate, nonshuffled_granule_rate,
    shuffled_grid_phase, shuffled_granule_phase,
    nonshuffled_grid_phase, nonshuffled_granule_phase], axis=1)

pearson_r.columns = ['distance', 'grid_seed', 'poisson',
                            's_grid_rate', 's_granule_rate',
                            'ns_grid_rate', 'ns_granule_rate',
                            's_grid_phase', 's_granule_phase',
                            'ns_grid_phase', 'ns_granule_phase', 
                            ]


sns.set(style='ticks', palette='deep', font='Arial', color_codes=True, font_scale=2)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "Pearson's R \n all bins \n grid seed = 1")

ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")

color_list_1 =["012a4a","013a63","01497c","2a6f97","2c7da0","468faf","61a5c2"]
color_list_2 = ['350000', '530000', 'ad1919', 'cf515f', 'e6889a']
my_cmap = _make_cmap(color_list_1)
my_cmap_2 = _make_cmap(color_list_2)

hue = list(pearson_r['distance'])

sns.scatterplot(ax=ax1,
              data=pearson_r, x="s_grid_rate", y="s_granule_rate",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax2,
              data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax3,
              data=pearson_r, x="s_grid_phase", y="s_granule_phase",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax4,
              data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)

for ax in fig2.axes:
    ax.get_legend().remove()
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    ax.set_xlim(-0.1,0.8)
    ax.set_ylim(-0.15,0.6)
    # ax.figure.colorbar(sm)

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "Pearson's R \n all bins \n grid seed = 1")

ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")

color_list_1 =["012a4a","013a63","01497c","2a6f97","2c7da0","468faf","61a5c2"]
color_list_2 = ['350000', '530000', 'ad1919', 'cf515f', 'e6889a']
my_cmap = _make_cmap(color_list_1)
my_cmap_2 = _make_cmap(color_list_2)

hue = list(pearson_r['distance'])

pearson_r['poisson'] = pearson_r['poisson'] == 0

sns.scatterplot(ax=ax1,
              data=pearson_r, x="s_grid_rate", y="s_granule_rate",
              hue='poisson', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax2,
              data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
              hue='poisson', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax3,
              data=pearson_r, x="s_grid_phase", y="s_granule_phase",
              hue='poisson', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax4,
              data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
              hue='poisson', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)

for ax in fig2.axes:
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    ax.set_xlim(-0.1,0.8)
    ax.set_ylim(-0.15,0.6)

fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "Pearson's R \n all bins \n grid seed = 1")

ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")

color_list_1 =["012a4a","013a63","01497c","2a6f97","2c7da0","468faf","61a5c2"]
color_list_2 = ['350000', '530000', 'ad1919', 'cf515f', 'e6889a']
my_cmap = _make_cmap(color_list_1)
my_cmap_2 = _make_cmap(color_list_2)

hue = list(pearson_r['distance'])

pearson_r['poisson'] = pearson_r['poisson'] == 0

sns.scatterplot(ax=ax1,
              data=pearson_r.query('poisson == False'), x="s_grid_rate", y="s_granule_rate",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax2,
              data=pearson_r.query('poisson == False'), x="ns_grid_rate", y="ns_granule_rate",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax3,
              data=pearson_r.query('poisson == False'), x="s_grid_phase", y="s_granule_phase",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax4,
              data=pearson_r.query('poisson == False'), x="ns_grid_phase", y="ns_granule_phase",
              hue='distance', hue_norm=SymLogNorm(10), palette=my_cmap_2, s=8,
              linewidth=0.1, alpha=0.8)

for ax in fig2.axes:
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),'g--', linewidth=1)
    ax.set_xlim(-0.1,0.8)
    ax.set_ylim(-0.15,0.6)
