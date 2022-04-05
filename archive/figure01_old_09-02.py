# -*- coding: utf-8 -*-
"""
Figure 1 demonstrates the grid cell model with phase precession and the
shuffling feature.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import grid_model
from neural_coding import load_spikes, rate_n_phase
import matplotlib.gridspec as gridspec
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation as stg
from scipy import ndimage
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap as lcmap
from scipy.stats import pearsonr,  spearmanr
from scipy import stats
import copy
from matplotlib.colors import SymLogNorm, PowerNorm, Normalize


# =============================================================================
# =============================================================================
# # Figure 01 C, D and E
# =============================================================================
# =============================================================================

# make colormaps with hex codes
def make_cmap (color_list, N=100):
    colors = []
    for color in color_list:
        colors.append('#'+color)
        my_cmap = lcmap.from_list('my_cmap', colors, N=100)
    return my_cmap


# function to produce samples with phase precession from overall firing
def precession_spikes(overall, dur_s=5, n_sim=1000, T=0.1,
                      dt_s=0.002, bins_size_deg=7.2, shuffle=False,
                      poisson_seed_start = 100):
    dur_ms = dur_s*1000
    asig = AnalogSignal(overall,
                        units=1*pq.Hz,
                        t_start=0*pq.s,
                        t_stop=dur_s*pq.s,
                        sampling_period=dt_s*pq.s,
                        sampling_interval=dt_s*pq.s)
    
    times = np.arange(0, dur_s+T, T)
    n_time_bins = int(dur_s/T)
    phase_norm_fact = 360/bins_size_deg
    n_phase_bins = int(720/bins_size_deg)
    phases = [[] for _ in range(n_time_bins)]
    phases_doubled = [[] for _ in range(n_time_bins)]
    trains = []
    np.random.seed(poisson_seed_start)
    for i in range(n_sim):
        train = stg.inhomogeneous_poisson_process(asig,
                                                  refractory_period=(0.001 *
                                                                     pq.s),
                                                  as_array=True)*1000
        if shuffle is True:
            train = grid_model._randomize_grid_spikes(train, 100,
                                                      time_ms=dur_ms)/1000
        else:
            train = train/1000
        trains.append(train)
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = train[np.logical_and(train > time,
                                              train < times[j+1])]
            if curr_train.size > 0:
                phases[j] += list(curr_train % (T)/(T)*360)
                phases_doubled[j] += list(curr_train % (T)/(T)*360)
                phases_doubled[j] += list(curr_train % (T)/(T)*360+360)
    counts = np.empty((n_phase_bins, n_time_bins))
    for i in range(n_phase_bins):
        for j, phases_in_time in enumerate(phases_doubled):
            phases_in_time = np.array(phases_in_time)
            counts[i][j] = ((bins_size_deg*(i) < phases_in_time) &
                            (phases_in_time < bins_size_deg*(i+1))).sum()
    f = int(1/T)
    phase_loc = counts*phase_norm_fact*f/n_sim
    phase_loc = ndimage.gaussian_filter(phase_loc, sigma=[1, 1])
    return trains, phases, phase_loc




# =============================================================================
# 1C
# =============================================================================

spacing = [30, 50, 70 ,100]
pos_peak = [[50, 100], [100,80], [40, 80], [30, 100]]
orientation = [25, 30, 15, 10]
dur_s = 2
dur_ms = dur_s*1000
n_sim = 20
grid_rates = []
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
f1c, axes = plt.subplots(2, 2, sharex=True,
                        sharey=True, figsize=(8, 6))

for i, ax in enumerate(axes.flatten()):
    grid_rate  = (grid_model._grid_maker(spacing[i],
                                         orientation[i],
                                         pos_peak[i]).reshape(200, 200, 1))
    im = ax.imshow(20*grid_rate, aspect='equal', cmap=cmap)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
cax = f1c.add_axes([0.85, 0.08, 0.025, 0.35])
cbar = f1c.colorbar(im, cax=cax)
cbar.set_label('Hz', labelpad=15, rotation=270)
plt.subplots_adjust(left=0.10, bottom=0.025,
                    right=0.84, top=0.99, wspace=0, hspace=0.040)

plt.rcParams['svg.fonttype'] = 'none'
save_dir = '/home/baris/paper/figures/figure01/'
f1c.savefig(f'{save_dir}figure01_C.svg', dpi=200)
f1c.savefig(f'{save_dir}figure01_C.png', dpi=200)

# =============================================================================
# 1D
# =============================================================================

spacing = 20
pos_peak = [100,100]
orientation = 30
dur_s = 2
dur_ms = dur_s*1000
n_sim = 20

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

trains, phases, phase_loc = precession_spikes(grid_overall,
                                               dur_s=2,
                                               shuffle=False,
                                               n_sim=20,
                                               poisson_seed_start = 101)

means = [np.mean(i) for i in phases]
repeated = np.repeat(means, 100)

# plotting

plt.close('all')
sns.reset_orig()
sns.set(style='dark', palette='deep', font='Arial',
        font_scale=1, color_codes=True)
mpl.rcParams['svg.fonttype'] = "none"
cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
f1, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True,
                                        gridspec_kw={'height_ratios':
                                                     [1.8,1,1,1]},
                                        figsize=(6, 8))

ax1.imshow(grid_rate[80:120,:80],cmap=cmap, extent=[0,2,1,0], aspect='auto')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_ylabel('Grid field')

ax2.plot(rate_t_arr, grid_overall)
ax2.set_ylabel('Frequency (Hz)')
    
ax3.eventplot(np.array(trains[:5]), linewidth=0.7, linelengths=0.5)
ax3.set_yticklabels([])
ax3.set_ylabel('Spike trains')

ax4.plot(np.arange(0, 2, 2/2000), repeated/180)
ax4.set_ylim([0, 2])
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Phases (\u03C0)')

xpos = np.arange(0, dur_s+0.1, 0.1)
for xc in xpos:
    ax2.axvline(xc, color='0.5', linestyle='-.', linewidth=0.5)
    ax3.axvline(xc, color="0.5", linestyle='-.', linewidth=0.5)

f1.tight_layout()
plt.rcParams["svg.fonttype"] = "none"
save_dir = '/home/baris/paper/figures/figure01/'
f1.savefig(save_dir+'figure01_D.svg', dpi=200)
f1.savefig(save_dir+'figure01_D.png', dpi=200)



# =============================================================================
# 1E
# =============================================================================

color_list = ["103900", "5b7a5b","eae5d6","e89005","853512"]
my_cmap = make_cmap(color_list)

spacing = 40
pos_peak = [[100, 100], [100,100], [80, 100]]
orientation = [30, 25, 25]
dur = 5
shuffle = False

for pos_peak, orientation in zip(pos_peak, orientation):
    print(pos_peak)
    print(orientation)
    grid_rate = grid_model._grid_maker(spacing,
                                 orientation, pos_peak).reshape(200, 200, 1)
    
    grid_rates = np.append(grid_rate, grid_rate, axis=2)
    spacings = [spacing, spacing]
    grid_dist = grid_model._rate2dist(grid_rates,
                                      spacings)[:, :, 0].reshape(200, 200, 1)
    trajs = np.array([50])
    dist_trajs = grid_model._draw_traj(grid_dist, 1, trajs, dur_ms=5000)
    rate_trajs = grid_model._draw_traj(grid_rate, 1, trajs, dur_ms=5000)
    rate_trajs, rate_t_arr = grid_model._interp(rate_trajs, 5, new_dt_s=0.002)
    
    grid_overall = grid_model._overall(dist_trajs,
                                 rate_trajs, 240, 0.1, 1, 1, 5, 20, 5)[0, :, 0]
    trains, phases, phase_loc = precession_spikes(grid_overall,
                                                  shuffle=shuffle)

    # plotting
    
    cmap=my_cmap
    cmap2 = sns.color_palette('RdYlBu_r', as_cmap=True)

    rate = grid_rate.reshape(200, 200)
    f2, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                  gridspec_kw={'height_ratios': [1, 2]},
                                  figsize=(7, 9))
    # f2.tight_layout(pad=0.1)
    im2 = ax2.imshow(phase_loc, aspect='auto',
                     cmap=cmap, extent=[0, 100, 720, 0],
                     vmin=0, vmax=66)
    ax2.set_ylim((0, 720))
    im1 = ax1.imshow(rate[60:140, :], aspect='equal',
                     cmap=cmap2, extent=[0, 100, 40, 0])
    # ax1.set_xticklabels([])
    ax1.set_ylabel('Location (cm)')
    ax1.set_yticks(np.arange(0,60,20))
    # ax1.set_yticklabels([])
    # ax1.set_title(f'\u0394= {spacing} cm    ' +
    #               f'[$x_{0}$ $y_{0}$]={(100-np.array(pos_peak))}     ' +
    #               f'\u03A8={orientation} degree', loc='center')
    ax2.set_xlabel('Location (cm)')
    ax2.set_ylabel('Theta phase (deg)')
    ax2.set_xticks(np.arange(0,120,20))
    ax2.set_yticks(np.arange(0,1080,360))
    f2.subplots_adjust(right=0.8)
    cax = f2.add_axes([0.85, 0.16, 0.04, 0.35])
    cbar = f2.colorbar(im2, cax=cax)
    cbar.set_label('Hz', labelpad=15, rotation=270)
    # plt.tight_layout()
    parameters = (f'spacing_center_orientation_' +
                  f'{spacing}_{pos_peak[0]}_{orientation}')
    
    plt.rcParams["svg.fonttype"] = "none"
    save_dir = '/home/baris/paper/figures/figure01/'
    f2.savefig(f'{save_dir}figure01_E_{parameters}.svg', dpi=200)
    f2.savefig(f'{save_dir}figure01_E_{parameters}.png', dpi=200)



# =============================================================================
# 1F
# =============================================================================

trajectories = [75]
n_samples = 20
grid_seeds = np.arange(1, 11, 1)
tuning = 'full'

grid_spikes = []
granule_spikes = []
for grid_seed in grid_seeds:
    path = ("/home/baris/results/" + str(tuning) +
            "/collective/grid-seed_duration_shuffling_tuning_")
    # non-shuffled
    ns_path = (path + str(grid_seed) + "_2000_non-shuffled_"+str(tuning))
    grid_spikes.append(load_spikes(ns_path,
                                   "grid", trajectories, n_samples)[75])
    granule_spikes.append(load_spikes(ns_path, "granule",
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
my_pal = {'#09316c', '#066d49'}
f1f, ax = plt.subplots()
sns.barplot(x='population', y='active cells %', ax=ax,
            data=act_cell_df, palette=my_pal, ci='sd', capsize=.2)

plt.rcParams["svg.fonttype"] = "none"
save_dir = '/home/baris/paper/figures/figure01/'
f1f.savefig(f'{save_dir}figure01_F.svg', dpi=200)
f1f.savefig(f'{save_dir}figure01_F.png', dpi=200) 





# =============================================================================
# Figure 01 I, J and K
# =============================================================================


# =============================================================================
# I, J
# =============================================================================


# to do
# Rin Rout from thesis

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
    print('path ok')
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
    

    all_codes[grid_seed] = {"shuffled": {}, "non-shuffled": {}}
    all_codes[grid_seed]["shuffled"] = {"grid": {}, "granule": {}}
    all_codes[grid_seed]["non-shuffled"] = {"grid": {}, "granule": {}}

    all_codes[grid_seed]['non-shuffled']['grid'] = {'rate': grid_rate_code,
                      'phase': grid_phase_code}
    all_codes[grid_seed]['non-shuffled']['granule'] = {'rate': granule_rate_code,
                      'phase': granule_phase_code}


# 75 vs all in all time bins
# calculate pearson R             
r_data = []
for grid_seed in all_codes:
    shuffling = 'non-shuffled'
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
                    



 
# dataframe

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
pearson_r.columns = ['distance', 'grid_seed' ,
                            'ns_grid_rate', 'ns_granule_rate',
                            'ns_grid_phase', 'ns_granule_phase']

# plotting

# "0f0f0f",
color_list_1 = ["2d2e2e", "716969","9a8f97", "a7a1af"]
# color_list = ["ced4da","adb5bd","6c757d","495057","343a40"]
# color_list = ["03045e","023e8a","0096c7","00b4d8","48cae4"]

color_list_2 = ['350000', '420000', '530000', '670000', '762431']



my_cmap = make_cmap(color_list_1)
my_cmap_2 = make_cmap(color_list_2)

sns.set(style='darkgrid', font='Arial',
        font_scale=1)

f1ij, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))

ax1.set_title("Rate Code")
ax2.set_title("Phase Code")
hue = list(pearson_r['distance'])
sns.scatterplot(ax=ax1,
              data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
              hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap, s=15,
              linewidth=0.1, alpha=0.8)
sns.scatterplot(ax=ax2,
              data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
              hue=hue, hue_norm=SymLogNorm(10), palette=my_cmap_2, s=15,
              linewidth=0.1, alpha=0.8)


for ax in f1ij.axes:
    ax.get_legend().remove()
    ax.plot(np.arange(-0.2,1.1,0.1),np.arange(-0.2,1.1,0.1),
            color = '#2c423f', alpha=0.4, linewidth=4)
    ax.set_xlim(-0.1,0.5)
    ax.set_ylim(-0.1,0.5)
    # ax.figure.colorbar(sm)

ns_rate = stats.binned_statistic(pearson_r['ns_grid_rate'],
                                 list((pearson_r['ns_grid_rate'],
                                       pearson_r['ns_granule_rate'])),
                                 'mean',
                                 bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
ns_phase = stats.binned_statistic(pearson_r['ns_grid_phase'],
                                  list((pearson_r['ns_grid_phase'],
                                        pearson_r['ns_granule_phase'])),
                                  'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])

ax1.plot(ns_rate[0][0], ns_rate[0][1], 'k',
         linestyle=(0, (6, 2)), linewidth=2)
ax2.plot(ns_phase[0][0], ns_phase[0][1], 'k',
         linestyle=(0, (6, 2)), linewidth=2)

ax1.set_ylabel('$R_{out}$')
ax1.set_xlabel('$R_{in}$')
ax2.set_xlabel('$R_{in}$')

plt.tight_layout()

plt.rcParams["svg.fonttype"] = "none"
save_dir = '/home/baris/paper/figures/figure01/'
f1ij.savefig(f'{save_dir}figure01_I-J.svg', dpi=200)
f1ij.savefig(f'{save_dir}figure01_I-J.png', dpi=200)


# =============================================================================
# K
# =============================================================================


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

# plotting

my_pal = {'#762431', '#2d2e2e'}
f1k, ax = plt.subplots()
sns.boxplot(x='code', y='mean deltaR', ax=ax,
            data=ns_deltaR, palette=my_pal)

for grid in grid_seeds:
    rate_data = ns_deltaR[(ns_deltaR['code'] == 'rate') &
                          (ns_deltaR['grid_seed'] == grid)]
    rate_data = rate_data['mean deltaR'].iloc[0]
    phase_data = ns_deltaR[(ns_deltaR['code'] == 'phase') &
                           (ns_deltaR['grid_seed'] == grid)]
    phase_data = phase_data['mean deltaR'].iloc[0]
    sns.lineplot(x=[0, 1], y=[rate_data, phase_data],
                 color='k', linewidth=0.5)
    sns.scatterplot(x=[0, 1], y=[rate_data, phase_data],
                    color='k', s=20)
ax.set_ylabel('mean $\u0394R$')    
plt.rcParams["svg.fonttype"] = "none"
save_dir = '/home/baris/paper/figures/figure01/'
f1k.savefig(f'{save_dir}figure01_K.svg', dpi=200)
f1k.savefig(f'{save_dir}figure01_K.png', dpi=200)    





# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# spacing = 50
# orientation = 30
# pos_peak = [20, 20]
# arr_size = 200
# dur_ms = 2000
# dt_s = 0.002

# trajectories = [75.0]
# simulation_result = grid_model.grid_simulate(
#     trajectories,
#     dur_ms=dur_ms,
#     grid_seed=1,
#     poiss_seeds=[1],
#     shuffle='non-shuffled',
#     n_grid=200,
#     speed_cm=20,
#     rate_scale=5,
#     arr_size=arr_size,
#     f=10,
#     shift_deg=180,
# <<<<<<< HEAD
#     dt_s=0.001,
# =======
#     dt_s=dt_s,
# >>>>>>> 5887f8a0c0c1aebf6c85bad742693598fed6cba3
#     large_output=True
# )

# grid_length_cm = 100
# grid_axes_ticks = np.arange(0,grid_length_cm,grid_length_cm/arr_size)

# trajectories_y = 100 - np.array(trajectories)
# trajectories_xstart = 0
# trajectories_xend= 40

# fig = plt.figure(constrained_layout=True)
# gs = fig.add_gridspec(6,2)
# ax1 = fig.add_subplot(gs[:3,0])
# ax1.imshow(simulation_result[3][:,:,1], extent=[0,100,0,100]) # TODO UNITS
# for trajectory in trajectories_y:
#     ax1.plot([trajectories_xstart, trajectories_xend], [trajectory]*2,
#              linewidth=3)
# ax2 = fig.add_subplot(gs[0:2,1])
# x = np.arange(0,dur_ms,dt_s*1000)
# ax2.plot(x,simulation_result[4][1,:,0])
# ax2.plot(x,simulation_result[4][1,:,1])
# ax2.plot(x,simulation_result[4][1,:,2])
# ax2.set_xlabel("Time (ms)")
# ax2.set_ylabel("Frequency (Hz)")
# ax2.legend(("Traj1", "Traj2", "Traj3"))
# ax3 = fig.add_subplot(gs[2:3,1])
# ax3.eventplot(simulation_result[0][trajectories[0]][1][1], lineoffsets=3, color=colors[0])
# ax3.eventplot(simulation_result[0][trajectories[1]][1][1], lineoffsets=2, color=colors[1])
# ax3.eventplot(simulation_result[0][trajectories[2]][1][1], lineoffsets=1, color=colors[2])
# ax3.set_ylabel("Trajectory")
# ax3.set_xlabel("Time (ms)")
# ax3.get_yaxis().set_visible(False)
# xlim=[-10, 2010]
# ax2.set_xlim((xlim))
# ax3.set_xlim((xlim))



