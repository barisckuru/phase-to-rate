#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 1 demonstrates the grid cell model with phase precession.
"""
import numpy as np
import seaborn as sns
import pandas as pd
from phase_to_rate import grid_model
from phase_to_rate.neural_coding import load_spikes
from phase_to_rate.figure_functions import (_make_cmap, _precession_spikes,
                              _adjust_box_widths, _adjust_bar_widths)
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import SymLogNorm
import matplotlib.font_manager
from matplotlib.colors import LinearSegmentedColormap as lcmap
from scipy.stats import pearsonr,  spearmanr
from scipy import stats
import copy
import pickle
import os

# Where is the data and where will figure be saved
dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')
save_dir = dirname

# plotting settings
sns.set(style='ticks', palette='deep', font='Arial', color_codes=True)
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 3
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 3
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.size'] = 3
mpl.rcParams['ytick.minor.width'] = 1
plt.rc('font', size=10) #controls default text size
plt.rc('axes', titlesize=8) #fontsize of the title
plt.rc('axes', labelsize=10) #fontsize of the x and y labels
plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10) #fontsize of the legend
cm=1/2.54


# =============================================================================
# Figure 1C
# =============================================================================

spacing = [30, 50, 70, 100]
pos_peak = [[50, 100], [100, 80], [40, 80], [30, 90]]
orientation = [25, 30, 15, 10]
dur_s = 2
dur_ms = dur_s*1000
n_sim = 20
grid_rates = []
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
f1c, axes = plt.subplots(2, 2, sharex=True,
                         sharey=True, figsize=(9, 8))

for i, ax in enumerate(axes.flatten()):
    grid_rate = (grid_model._grid_maker(spacing[i],
                                        orientation[i],
                                        pos_peak[i]).reshape(200, 200, 1))
    im = ax.imshow(20*grid_rate, aspect='equal', cmap=cmap)
    ax.set_title(f' $\lambda$={spacing[i]}, $\Phi$={orientation[i]},'+
                  f' $r_0$={pos_peak[i]}', fontsize=15)
    ax.set_xticks([0, 100, 199])
    ax.set_xticklabels([0, 50, 100])
    ax.set_yticks([0, 100, 199])
    ax.set_yticklabels([0, 50, 100])

cax = f1c.add_axes([0.90, 0.35, 0.025, 0.35])
cbar = f1c.colorbar(im, cax=cax)
cbar.set_label('Hz', labelpad=15, rotation=270)
plt.subplots_adjust(left=0.10, bottom=0.025,
                    right=0.84, top=0.99, wspace=0.15, hspace=0.15)
plt.rcParams['svg.fonttype'] = 'none'
f1c.savefig(os.path.join(save_dir, 'figure01_C.svg'), dpi=200)
f1c.savefig(os.path.join(save_dir, 'figure01_C.png'), dpi=200)

# =============================================================================
# Figure 1D
# =============================================================================

spacing = 20
pos_peak = [100, 100]
orientation = 30
dur_s = 2
dur_ms = dur_s*1000
n_sim = 5
poiss_start = 105

grid_rate = grid_model._grid_maker(spacing,
                                   orientation, pos_peak).reshape(200, 200, 1)
grid_rates = np.append(grid_rate, grid_rate, axis=2)
spacings = [spacing, spacing]
grid_dist = grid_model._rate2dist(grid_rates, spacings)[:, :, 0].reshape(200,
                                                                         200,
                                                                         1)
trajs = np.array([50])
dist_trajs = grid_model._draw_traj(grid_dist, 1, trajs, dur_ms=dur_ms)
rate_trajs = grid_model._draw_traj(grid_rate, 1, trajs, dur_ms=dur_ms)
rate_trajs, rate_t_arr = grid_model._interp(rate_trajs, dur_s, new_dt_s=0.002)

grid_overall = grid_model._overall(dist_trajs,
                                   rate_trajs, 240, 0.1,
                                   1, 1, 5, 20, dur_s)[0, :, 0]

trains, phases, phase_loc = _precession_spikes(grid_overall,
                                               dur_s=2,
                                               shuffle=False,
                                               n_sim=n_sim,
                                               poisson_seed_start=poiss_start)

means = [np.mean(i) for i in phases]
repeated = np.repeat(means, 100)

# plotting
plt.close('all')
mpl.rcParams['svg.fonttype'] = "none"
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
f1d, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1,
                                        gridspec_kw={'height_ratios':
                                                     [1.2, 1, 1, 1]},
                                        figsize=(6*cm, 10*cm))
im1 = ax1.imshow(grid_rate[80:120, :80], cmap=cmap,
           extent=[0,4,2,0], aspect='auto')
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels([0, 10, 20, 30, 40])
ax1.set_yticks([0, 1, 2])
ax1.set_yticklabels([0, 10, 20])
ax1.set_ylabel('Location (cm)')
# ax1.set_xlabel('Location (cm)')
ax1.set_xticklabels([])

ax2.plot(rate_t_arr, grid_overall, color='#6c757d', linewidth=0.7, zorder=10)
ax2.set_ylabel('Rate (Hz)')
ax2.set_xlim([0, 2])
ax2.set_xticklabels([])

ax3.eventplot(np.array(trains[:5]), linewidth=0.7,
              linelengths=0.5, color='#6c757d', zorder=10)
# ax3.set_yticklabels([])
ax3.set_ylabel('Spikes')
ax3.set_xlim([0, 2])
ax3.set_xticklabels([])

ax4.plot(np.arange(0, 2, 2/2000), repeated/180,
         linewidth=0.7, color='#6c757d', zorder=10)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Phase (\u03C0)')
ax4.set_xlim([0, 2])
ax4.set_ylim([0, 2])

xpos = np.arange(0, dur_s+0.1, 0.1)
for xc in xpos:
    ax2.axvline(xc, color='0.2', linestyle='--', linewidth=0.3)
    ax3.axvline(xc, color="0.2", linestyle='--', linewidth=0.3)
    ax4.axvline(xc, color="0.2", linestyle='--', linewidth=0.3)

f1d.subplots_adjust(bottom=0.15, hspace=0.3, top=0.98, left=0.3, right=0.9)
plt.rcParams["svg.fonttype"] = "none"
f1d.savefig(os.path.join(save_dir, 'figure01_D.svg'), dpi=200)
f1d.savefig(os.path.join(save_dir, 'figure01_D.png'), dpi=200)


# =============================================================================
# Figure 1E
# =============================================================================

# color_list = ["103900", "5b7a5b", "eae5d6", "e89005", "853512"]
# my_cmap = _make_cmap(color_list)

spacing = 40
pos_peak = [[100, 100], [100, 100], [80, 100]]
orientation = [30, 25, 25]
dur = 5
shuffle = False

for pos_peak, orientation in zip(pos_peak, orientation):
    print(pos_peak)
    print(orientation)
    grid_rate = grid_model._grid_maker(spacing,
                                       orientation, pos_peak).reshape(200,
                                                                      200, 1)

    grid_rates = np.append(grid_rate, grid_rate, axis=2)
    spacings = [spacing, spacing]
    grid_dist = grid_model._rate2dist(grid_rates,
                                      spacings)[:, :, 0].reshape(200, 200, 1)
    trajs = np.array([50])
    dist_trajs = grid_model._draw_traj(grid_dist, 1, trajs, dur_ms=5000)
    rate_trajs = grid_model._draw_traj(grid_rate, 1, trajs, dur_ms=5000)
    rate_trajs, rate_t_arr = grid_model._interp(rate_trajs, 5, new_dt_s=0.002)

    grid_overall = grid_model._overall(dist_trajs,
                                       rate_trajs, 240, 0.1, 1, 1, 5,
                                       20, 5)[0, :, 0]
    trains, phases, phase_loc = _precession_spikes(grid_overall,
                                                   shuffle=shuffle)

    # plotting

    rate = grid_rate.reshape(200, 200)
    f1e, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                  gridspec_kw={'height_ratios': [1, 2]},
                                  figsize=(7, 9))
    im2 = ax2.imshow(phase_loc, aspect='auto',
                     cmap='RdYlBu_r', extent=[0, 100, 720, 0],
                     vmin=0, vmax=66)
    ax2.set_ylim((0, 720))
    im1 = ax1.imshow(20*rate[60:140, :], aspect='equal',
                     cmap='RdYlBu_r', extent=[0, 100, 40, 0])
    ax1.set_ylabel('Location (cm)')
    ax1.set_yticks(np.arange(0, 60, 20))
    cax = f1e.add_axes([0.85, 0.69, 0.04, 0.15])
    cbar = f1e.colorbar(im1, cax=cax)
    cbar.set_label('Hz', labelpad=15, rotation=270)

    ax2.set_xlabel('Location (cm)')
    ax2.set_ylabel('Theta phase (deg)')
    ax2.set_xticks(np.arange(0, 120, 20))
    ax2.set_yticks(np.arange(0, 1080, 360))
    f1e.subplots_adjust(right=0.8)
    cax2 = f1e.add_axes([0.85, 0.16, 0.04, 0.35])
    cbar = f1e.colorbar(im2, cax=cax2)
    cbar.set_label('Hz', labelpad=15, rotation=270)
    parameters = ('spacing_center_orientation_' +
                  f'{spacing}_{pos_peak[0]}_{orientation}')

    plt.rcParams["svg.fonttype"] = "none"
    f1e.savefig(os.path.join(save_dir, f'figure01_E_{parameters}.svg'), dpi=200)
    f1e.savefig(os.path.join(save_dir, f'figure01_E_{parameters}.png'), dpi=200)


# =============================================================================
# Figure 1G
# =============================================================================
my_pal = {'grid': '#6c757d', 'granule': '#09316c'}
f1g, (ax1, ax2) = plt.subplots(2,1, figsize=[4.4*cm, 5*cm])

trajectories = [75]
n_samples = 20
grid_seeds = np.arange(1, 11, 1)
tuning = 'full'

grid_spikes = []
granule_spikes = []
for grid_seed in grid_seeds:
    path = os.path.join(results_dir,
                        'main',
                        tuning,
                        'collective',
                        f'grid-seed_duration_shuffling_tuning_{grid_seed}_2000_non-shuffled_{tuning}')
    # non-shuffled
    grid_spikes.append(load_spikes(path,
                                   "grid", trajectories, n_samples)[75])
    granule_spikes.append(load_spikes(path, "granule",
                                      trajectories, n_samples)[75])


grid_counts_byseed = []
gra_counts_byseed = []
for grid_seed in range(10):
    grid_counts = []
    granule_counts = []
    for poiss in range(20):
        grids = grid_spikes[grid_seed][poiss]
        granules = granule_spikes[grid_seed][poiss]
        ct_grid = 0
        for grid in grids:
            if len(grid) > 0:
                ct_grid += 1
        grid_counts.append(ct_grid)
        ct_gra = 0
        for gra in granules:
            if len(gra) > 0:
                ct_gra += 1
        granule_counts.append(ct_gra)
    grid_counts_byseed.append(np.mean(grid_counts)/2)
    gra_counts_byseed.append(np.mean(granule_counts)/20)

grid_counts_byseed.extend(gra_counts_byseed)
cell = 10*['grid'] + 10*['granule']
grid_seeds = list(np.arange(1, 11, 1))
grid_seeds.extend(grid_seeds)
act_cell_df = np.stack((grid_counts_byseed, cell, grid_seeds)).T
act_cell_df = pd.DataFrame(act_cell_df, columns=('active cells %',
                                                 'population',
                                                 'grid seed'))
act_cell_df['active cells %'] = act_cell_df['active cells %'].astype('float')
act_cell_df['grid seed'] = act_cell_df['grid seed'].astype('float')

sns.despine(fig=f1g)
sns.barplot(x='population', y='active cells %', ax=ax1, zorder=1, errwidth=0.6,
            data=act_cell_df, palette=my_pal, ci='sd', capsize=.2)
sns.scatterplot(x='population', y='active cells %', ax=ax1, zorder=10,
                data=act_cell_df, color='black', s=3)

all_mean_rates_path = os.path.join(results_dir,
                                   'excel',
                                   'mean_firing_rates.xlsx')
all_mean_rates = pd.read_excel(all_mean_rates_path, index_col=0)
mean_rates = all_mean_rates[(all_mean_rates['shuffling']=='non-shuffled') &
                            (all_mean_rates['tuning']=='full')]

sns.barplot(x='cell', y='mean_rate', ax=ax2, zorder=1, errwidth=0.6,
            data=mean_rates, palette=my_pal, ci='sd', capsize=.2)
sns.scatterplot(x='cell', y='mean_rate', ax=ax2, zorder=10,
                data=mean_rates, ci='sd', color='black', s=3)
ax1.set_xticklabels([])
ax2.set_xticklabels(['EC', 'GC'])
ax1.set(xlabel=None)
ax2.set(xlabel=None)
f1g.subplots_adjust(bottom=0.25, hspace=0.4, top=0.98, left=0.4, right=0.9)
_adjust_bar_widths(ax1, 0.4)
_adjust_bar_widths(ax2, 0.4)
plt.rcParams["svg.fonttype"] = "none"
f1g.savefig(os.path.join(save_dir, 'figure01_G.svg'), dpi=200)
f1g.savefig(os.path.join(save_dir, 'figure01_G.png'), dpi=200)


# =============================================================================
# Figure 1F
# =============================================================================

vline_alpha = 0.3
linewidth=2
xlim = (-10, 2010)

n_grid = 20
n_granule = 20

f1f, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.eventplot(grids[0:n_grid], linelengths=1, linewidth=linewidth,
              lineoffsets=range(1,n_grid+1))
ax1.set_xlim(xlim)
ax1.set_ylabel("Grid Cell #")
ax1.set_ylim((0,n_grid+1))
ax1.set_yticks(range(0,n_grid+1, 5))
ax1.set_xlabel('Time (ms)')
# ax2 = f1f_2.add_subplot(gs[4:8,0:6])
ax2.eventplot(granules[0:n_granule], linelengths=1, linewidth=linewidth,
              lineoffsets=range(1,n_granule+1))
ax2.set_xlim(xlim)
ax2.set_ylabel("Granule Cell #")
ax2.set_xlabel("Time (ms)")
ax2.set_ylim((0,n_granule+1))
ax2.set_yticks(range(0,n_granule+1, 5))

plt.tight_layout()

plt.rcParams['svg.fonttype' ]= 'none'
f1f.savefig(os.path.join(save_dir, 'figure01_F.svg'), dpi=200)
f1f.savefig(os.path.join(save_dir, 'figure01_F.png'), dpi=200)

# =============================================================================
# Figure 1I and 1J
# =============================================================================
# pre - plotting

color_list_1 =["012a4a","013a63","01497c","2a6f97","2c7da0","468faf","61a5c2"]
color_list_2 = ['350000', '530000', 'ad1919', 'cf515f', 'e6889a']
my_cmap = _make_cmap(color_list_1)
my_cmap_2 = _make_cmap(color_list_2)


f1i, ax1 = plt.subplots(figsize=(4*cm, 4*cm))
f1j, ax3 = plt.subplots(figsize=(4*cm, 4*cm))

# load data, codes

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
n_samples = 20
grid_seeds = np.arange(1, 11, 1)
tuning = 'full'

fname = results_dir + 'pickled/neural_codes_full.pkl'
fname = os.path.join(results_dir, 'pickled', 'neural_codes_full.pkl')
with open(fname, 'rb') as f:
    all_codes = pickle.load(f)

# 75 vs all in all time bins
# calculate pearson R
r_data = []
for grid_seed in all_codes:
    shuffling = 'non-shuffled'
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


# preprocessing

df = pd.DataFrame(r_data,
                  columns=['distance', 'pearson_r', 'spearman_r',
                           'poisson_seed', 'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')

nonshuffled_grid_rate = (df[(df['cell_type'] == 'grid') &
                            (df['code_type'] == 'rate') &
                            (df['shuffling'] == 'non-shuffled')]
                         [['distance', 'grid_seed',
                           'pearson_r']].reset_index(drop=True))
nonshuffled_granule_rate = (df[(df['cell_type'] == 'granule') &
                               (df['code_type'] == 'rate') &
                               (df['shuffling'] == 'non-shuffled')]
                            ['pearson_r'].reset_index(drop=True))
nonshuffled_grid_phase = (df[(df['cell_type'] == 'grid') &
                             (df['code_type'] == 'phase') &
                             (df['shuffling'] == 'non-shuffled')]
                          ['pearson_r'].reset_index(drop=True))
nonshuffled_granule_phase = (df[(df['cell_type'] == 'granule') &
                                (df['code_type'] == 'phase') &
                                (df['shuffling'] == 'non-shuffled')]
                             ['pearson_r'].reset_index(drop=True))

pearson_r = pd.concat([
    nonshuffled_grid_rate, nonshuffled_granule_rate,
    nonshuffled_grid_phase, nonshuffled_granule_phase], axis=1)
pearson_r.columns = ['distance', 'grid_seed',
                     'ns_grid_rate', 'ns_granule_rate',
                     'ns_grid_phase', 'ns_granule_phase']



hue = list(pearson_r['distance'])
# rate
sns.scatterplot(ax=ax1,
                data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
                hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap, s=1,
                linewidth=0.1, alpha=0.8)
ax1.get_legend().remove()
ax1.plot(np.arange(-0.2, 1.1, 0.1), np.arange(-0.2, 1.1, 0.1),
         color='#2c423f', alpha=0.4, linewidth=1)
ax1.set_xlim(-0.1, 0.5)
ax1.set_ylim(-0.1, 0.5)

ns_rate = stats.binned_statistic(pearson_r['ns_grid_rate'],
                                 list((pearson_r['ns_grid_rate'],
                                       pearson_r['ns_granule_rate'])),
                                 'mean',
                                 bins=[0, 0.1, 0.2, 0.3, 0.4,
                                       0.5, 0.6, 0.7, 0.8])
ax1.plot(ns_rate[0][0], ns_rate[0][1], 'k',
         linestyle=(0, (6, 2)), linewidth=2)
ax1.set_ylabel('$R_{out}$')
ax1.set_xlabel('$R_{in}$')

norm = mpl.colors.SymLogNorm(vmin=0.5, vmax=65, linthresh=0.1)
# f1i.colorbar(matplotlib.cm.ScalarMappable(cmap=my_cmap, norm=norm), ax=ax1)

# phase
sns.scatterplot(ax=ax3,
                data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
                hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap_2, s=1,
                linewidth=0.1, alpha=0.8)
ax3.get_legend().remove()
ax3.plot(np.arange(-0.2, 1.1, 0.1), np.arange(-0.2, 1.1, 0.1),
         color='#2c423f', alpha=0.4, linewidth=1)
ax3.set_xlim(-0.1, 0.5)
ax3.set_ylim(-0.1, 0.5)

ns_phase = stats.binned_statistic(pearson_r['ns_grid_phase'],
                                  list((pearson_r['ns_grid_phase'],
                                        pearson_r['ns_granule_phase'])),
                                  'mean', bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax3.plot(ns_phase[0][0], ns_phase[0][1], 'k',
         linestyle=(0, (6, 2)), linewidth=2)
ax3.set_ylabel('$R_{out}$')
ax3.set_xlabel('$R_{in}$')
norm = mpl.colors.SymLogNorm(vmin=0.5, vmax=65, linthresh=0.1)
# f1j.colorbar(matplotlib.cm.ScalarMappable(cmap=my_cmap_2, norm=norm), ax=ax3)

# =============================================================================
# mean delta R
# =============================================================================
ax2 = inset_axes(ax1,  "20%", "50%", loc="upper right", borderpad=0)
ax4 = inset_axes(ax3,  "20%", "50%", loc="upper right", borderpad=0)
grid_seeds = np.arange(1, 11, 1)
delta_s_rate = []
delta_ns_rate = []
delta_s_phase = []
delta_ns_phase = []
for seed in grid_seeds:
    grid_1 = pearson_r.loc[(pearson_r['grid_seed'] == seed)]
    ns_rate = stats.binned_statistic(grid_1['ns_grid_rate'],
                                     list((grid_1['ns_grid_rate'],
                                           grid_1['ns_granule_rate'])),
                                     'mean',
                                     bins=np.arange(0, 1, 0.1))
    ns_phase = stats.binned_statistic(grid_1['ns_grid_phase'],
                                      list((grid_1['ns_grid_phase'],
                                            grid_1['ns_granule_phase'])),
                                      'mean',
                                      bins=np.arange(0, 0.6, 0.1))

    delta_ns_rate.append(np.mean((ns_rate[0][0] - ns_rate[0][1])
                                 [ns_rate[0][0] == ns_rate[0][0]]))
    delta_ns_phase.append(np.mean((ns_phase[0][0] - ns_phase[0][1])
                                  [ns_phase[0][0] == ns_phase[0][0]]))


ns_deltaR = np.concatenate((delta_ns_rate, delta_ns_phase))
shuffling = 20*['non-shuffled']
code = 10*['rate']+10*['phase']
seeds = 2*list(range(1, 11))
ns_delta = np.stack((ns_deltaR, shuffling, code, seeds), axis=1)
ns_deltaR = pd.DataFrame(ns_delta, columns=['mean deltaR',
                                            'shuffling',
                                            'code',
                                            'grid_seed'])
ns_deltaR['mean deltaR'] = ns_deltaR['mean deltaR'].astype('float')
ns_deltaR['grid_seed'] = ns_deltaR['grid_seed'].astype('float')

# plotting mean deltaR
my_pal1 = {'#01497c'}
my_pal2 = {'#ad1919'}

sns.boxplot(x='code', y='mean deltaR', ax=ax2, linewidth=0.5, fliersize=0.5,
            data=ns_deltaR[ns_deltaR['code'] == 'rate'], palette=my_pal1,
            width=0.5)
sns.scatterplot(x='code', y='mean deltaR', ax=ax2, s=3,
            data=ns_deltaR[ns_deltaR['code'] == 'rate'], color='black')

sns.boxplot(x='code', y='mean deltaR', ax=ax4, linewidth=0.5, fliersize=0.5,
            data=ns_deltaR[ns_deltaR['code'] == 'phase'], palette=my_pal2,
            width=0.5)
sns.scatterplot(x='code', y='mean deltaR', ax=ax4, s=3,
            data=ns_deltaR[ns_deltaR['code'] == 'phase'], color='black')


ax2.set_ylabel('mean $\u0394R$')
ax2.set(xticklabels=[])  # remove the tick labels
ax4.set_ylabel('mean $\u0394R$')
ax4.set_xlabel('')
ax2.yaxis.set_label_position("right")
ax4.yaxis.set_label_position("right")
sns.despine(ax=ax1)
sns.despine(ax=ax2, right=False, left=True)
sns.despine(ax=ax3)
sns.despine(ax=ax4, right=False, left=True)
f1i.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.9, wspace=1)
f1j.subplots_adjust(left=0.2, bottom=0.3, right=0.9, top=0.9, wspace=1)
_adjust_box_widths(f1i, 0.7)
_adjust_box_widths(f1j, 0.7)
plt.rcParams["svg.fonttype"] = "none"
f1i.savefig(os.path.join(save_dir, 'figure01_I_cbar.svg'), dpi=200)
f1i.savefig(os.path.join(save_dir, 'figure01_I_cbar.png'), dpi=200)
f1j.savefig(os.path.join(save_dir, 'figure01_J_cbar.svg'), dpi=200)
f1j.savefig(os.path.join(save_dir, 'figure01_J_cbar.png'), dpi=200)

