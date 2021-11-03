#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:19:20 2021

@author: baris


"""

from neural_coding import load_spikes, rate_n_phase
from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
from scipy.spatial import distance



# =============================================================================
# =============================================================================
# # Load the data
# =============================================================================
# =============================================================================


trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

# trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
#                 71, 70, 69, 68, 67, 66, 65, 60]

n_samples = 20
# grid_seeds = np.arange(1,11,1)

grid_seeds = np.array([1])

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





# =============================================================================
# =============================================================================
# # Load the data
# =============================================================================
# =============================================================================







# =============================================================================
# =============================================================================
# # ''' PCA'''
# =============================================================================
# =============================================================================

# all principal components vs explained ratio in all sampels pooled
seed = 1
cells = ['grid','granule']
codes = ['rate','phase']
shufflings = ['non-shuffled','shuffled']
compared_traj = 16 # baseline traj is at 75 cm --> 0
n_comp = 160
select_bin = 5
# comp_ratio = []
ct = 0

plt.figure()
for cell in cells:
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for code in codes:
        for shuffling in shufflings:
            x = all_codes[seed][shuffling][cell][code]
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]).T
            cut = select_bin*n_cell
            mid = int(x.shape[1]/2)
            x = np.concatenate((x[:, cut:(cut+n_cell)], 
                                x[:, (mid+cut):(mid+cut+n_cell)]), axis=1)
            
            pca = decomposition.PCA(n_components=n_comp)
            components = pca.fit(x)
            comp_ratio = pca.explained_variance_ratio_
            plt.plot(comp_ratio, label = f'{cell} {code} {shuffling}')
            ct +=1
plt.legend()        
plt.ylabel('Ratio (0-1)')
plt.xlabel('Principal Component')
plt.title("All trajectories (75-15cm) pooled explained varience \n 1 time bin, 1 grid seed")   



# sum of first 3 components

seed = 1
cells = ['grid','granule']
codes = ['rate','phase']
shufflings = ['non-shuffled','shuffled']
compared_traj = 16 # baseline traj is at 75 cm --> 0
n_comp = 40
select_bin = 5
# comp_ratio = []
ct = 0

plt.figure()
for cell in cells:
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for code in codes:
        for shuffling in shufflings:
            ratios_explained_3 = []
            for compared_traj in range(len(trajectories)):
                grid_rate_shuffled_1 = all_codes[seed][shuffling][cell][code][:,:,0]
                grid_rate_shuffled_2 = all_codes[seed][shuffling][cell][code][:,:,compared_traj]
                x = np.hstack((grid_rate_shuffled_1, grid_rate_shuffled_2)).T
                cut = select_bin*n_cell
                mid = int(x.shape[1]/2)
                x = np.concatenate((x[:, cut:(cut+n_cell)], 
                                    x[:, (mid+cut):(mid+cut+n_cell)]), axis=1)
                
                pca = decomposition.PCA(n_components=n_comp)
                pca.fit_transform(x)
                comp_ratio = pca.explained_variance_ratio_
                ct +=1
                ratios_explained_3.append(sum(comp_ratio[:3]))
            plt.plot(ratios_explained_3, label = f'{cell} {code} {shuffling}')
plt.xticks(np.arange(0,16), trajectories[1:])
plt.legend()        
plt.ylabel('Summed ratio')
plt.xlabel('75 vs Trajectories')
plt.title("Sum of explained varience of first 3 PCA components \n 1 time bin, 1 grid seed")   





# sparse PCA


plt.figure()
for cell in cells:
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for code in codes:
        for shuffling in shufflings:
            ratios_explained_3 = []
            for compared_traj in range(len(trajectories)):
                grid_rate_shuffled_1 = all_codes[seed][shuffling][cell][code][:,:,0]
                grid_rate_shuffled_2 = all_codes[seed][shuffling][cell][code][:,:,compared_traj]
                x = np.hstack((grid_rate_shuffled_1, grid_rate_shuffled_2)).T
                cut = select_bin*n_cell
                mid = int(x.shape[1]/2)
                x = np.concatenate((x[:, cut:(cut+n_cell)], 
                                    x[:, (mid+cut):(mid+cut+n_cell)]), axis=1)
                
                pca = decomposition.sparsePCA(n_components=n_comp)
                pca.fit_transform(x)


# dictionary learning and other dim reduction techniques to be tried









# measuring the distance after compression via PCA
# find the PCs for each trajectory and compress with PCA
# check for the distance between componenets
# you can directly automatize to see the correlation between trajectory pairs
seed = 1
cells = ['grid', 'granule']
codes = ['rate']
shufflings = ['non-shuffled','shuffled']

trajectories = [75, 60, 15]
n_comp = 20
select_bin = 5
# comp_ratio = []
ct = 0
components = []
distances = []

fig2, axes = plt.subplots(3, 4)
fig2.suptitle(
    "PCA Correlation Matrix")

# plt.figure()
ct = 0
for cell in cells:
    if cell == 'grid':
        n_cell = 200
    elif cell == 'granule':
        n_cell = 2000
    for code in codes:
        for shuffling in shufflings:
            ratios_explained_3 = []
            for compared_traj in range(len(trajectories)):
                base_x = all_codes[seed][shuffling][cell][code][:,:,0].T
                x = all_codes[seed][shuffling][cell][code][:,:,compared_traj].T
                cut = select_bin*n_cell
                mid = int(x.shape[1]/2)
                x = np.concatenate((x[:, cut:(cut+n_cell)], 
                                    x[:, (mid+cut):(mid+cut+n_cell)]), axis=1)
                
                base_x = np.concatenate((base_x[:, cut:(cut+n_cell)], 
                                         base_x[:, (mid+cut):(mid+cut+n_cell)]), axis=1)
                
                pca = decomposition.PCA(n_components=n_comp)
                
                base_pcs = pca.fit_transform(base_x)
                
                pca = decomposition.PCA(n_components=n_comp)
                pcs = pca.fit_transform(x)
                
                all_pcs = np.vstack((base_pcs, pcs))
                
                dist = distance.cdist(all_pcs, all_pcs, 'euclidean')
                ax = fig2.axes[ct]
                ax.set_title(f'75-{trajectories[compared_traj]} {cell} {code} {shuffling}')
                ax.imshow(dist, cmap='gray')
                ct +=1
                # distances.append(dist[np.triu_indices(20, k=1)])
                distances.append(dist)
                
                components.append(pcs)
plt.tight_layout()               
                
                
                
                

# arr = np.array(distances).T.flatten()

# trajectory = 760*['75 vs 75', '75 vs 60', '75 vs 15']
# shuffling =  380*(3*['non-shuffled']+3*['shuffled'])
# cell =  190*(6*['grid']+6*['granule'])


# distances_df = np.stack((arr, shuffling, cell, trajectory), axis=1)
# distances_df = pd.DataFrame(distances_df, columns=['distances', 'shuffling', 'cell', 'trajectory'])
# distances_df['distances'] = distances_df['distances'].astype('float')

# # sns.catplot(col='trajectory', data=distances_df, x= 'cell', y='distances', hue='shuffling',
# #             ci='sd')

# sns.catplot(col='trajectory', data=distances_df, x= 'cell', y='distances', hue='shuffling',
#             ci='sd', kind='bar')

# plt.suptitle(f'Correlation Distance between PCs \n {n_comp} PCs, 20 samples per trajectory \n 1 time bin, 1 grid seed')
# plt.tight_layout()

# dista = distance.cdist(base_pcs, base_pcs, 'correlation')







# sns.reset_orig()
# fig = plt.figure()
# ax = fig.add_subplot()
# t = np.arange(1,18)
# for comp in components:
#     ax.scatter(comp[:3, :], comp[1, :], c=t, cmap='Red')

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(components[0][:,0], components[0][:,1],components[0][:,2], marker='o', color='blue', label='75cm')
# ax.scatter(components[14][:,0], components[14][:,1],components[14][:,2], marker='*', color='green', label='60cm')
# ax.scatter(components[16][:,0], components[16][:,1],components[16][:,2], marker='v', color='red', label='15cm')

# plt.title('First 3 PCs \n 20 samples per trajectory, 1 time bin, 1 grid seed')
# plt.legend()




# distances_75 = distance.cdist(components[0], components[0], 'euclidean')
# distances_75 = distances[np.triu_indices(20, k=1)]


# distances_60 = distance.cdist(components[0], components[0], 'euclidean')
# distances_60 = distances[np.triu_indices(20, k=1)]

# distances_15 = distance.cdist(components[0], components[0], 'euclidean')
# distances_15 = distances[np.triu_indices(20, k=1)]

# dist_data = np.concatenate((distances_75, distances_60, distances_15))



# =============================================================================
# =============================================================================
# # PCA
# =============================================================================
# =============================================================================


# LDA

# labels = np.array(20*[1]+20*[0])
# lda = LDA()
# lda.fit(x, labels)
# lda.explained_variance_ratio_






# =============================================================================
# =============================================================================
# # # Mean Rates
# =============================================================================
# =============================================================================

sns.set()
ns_grid_code = np.array(((all_codes[grid_seed]['non-shuffled']['grid']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))
s_grid_code = np.array(((all_codes[grid_seed]['shuffled']['grid']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))
ns_granule_code = np.array(((all_codes[grid_seed]['non-shuffled']['granule']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))
s_granule_code = np.array(((all_codes[grid_seed]['shuffled']['granule']['rate'])
                /(np.sin(np.pi/4))*10).mean(axis=0).reshape(340))


means = np.concatenate((ns_grid_code, s_grid_code,
                       ns_granule_code, s_granule_code))
shuffling =  2*(340*['non-shuffled']+340*['shuffled'])
cell =  680*['grid']+680*['granule']

avarage_rate = np.stack((means, shuffling, cell), axis=1)

means_df = pd.DataFrame(avarage_rate,
                        columns=['mean_rate', 'shuffling', 'cell'])

means_df['mean_rate'] = means_df['mean_rate'].astype('float')

sns.barplot(data=means_df, x= 'cell', y='mean_rate', hue='shuffling',
            ci='sd', capsize=0.2, errwidth=(2))


 # np.mean(all_codes[grid_seed]['non-shuffled']['grid']['rate'])/(np.sin(np.pi/4))*10


# =============================================================================
# =============================================================================
# # Mean Rates
# =============================================================================
# =============================================================================



# =============================================================================
# =============================================================================
# # Perceptron
# =============================================================================
# =============================================================================
rate_learning_rate = 1e-4
phase_learning_rate = 1e-3
n_iter = 2000
optimizer='SparseAdam'

import copy
data = []
for grid_seed in all_codes:
    for shuffling in all_codes[grid_seed]:
        for cell in all_codes[grid_seed][shuffling]:
            for code in all_codes[grid_seed][shuffling][cell]:
                for traj_idx in range(len(trajectories)-1):
                    if code == 'phase':
                        learning_rate = phase_learning_rate
                    elif code == 'rate':
                        learning_rate = rate_learning_rate
                        if cell == 'granule':
                            n_iter = 10000
                    input_code = all_codes[grid_seed][shuffling][cell][code]
                    perceptron_input = np.hstack((input_code[:, :, 0], input_code[:, :, traj_idx + 1]))
                    th_cross, train_loss = run_perceptron(
                        perceptron_input,
                        grid_seed,
                        optimizer=optimizer,
                        learning_rate=learning_rate,
                        n_iter=n_iter)
                    traj = trajectories[traj_idx+1]
                    idx = 75 - traj
                    comp_trajectories = str(75)+'_'+str(traj)
                    data_sing = [idx, 1/th_cross, th_cross, comp_trajectories, grid_seed, shuffling, cell, code, learning_rate]
                    data.append(copy.deepcopy(data_sing))
                    print(traj)
                    print(grid_seed)
                    print(shuffling)
                    print(cell)
                    print(code)
    


add_data = []

for traj_idx in range(len(trajectories)-1):
    n_iter = 10000
    input_code = all_codes[1]['shuffled']['grid']['phase']
    perceptron_input = np.hstack((input_code[:, :, 0], input_code[:, :, traj_idx + 1]))
    th_cross, train_loss = run_perceptron(
        perceptron_input,
        grid_seed,
        learning_rate=learning_rate,
        n_iter=n_iter)
    traj = trajectories[traj_idx+1]
    idx = 75 - traj
    comp_trajectories = str(75)+'_'+str(traj)
    data_sing = [idx, 1/th_cross, th_cross, comp_trajectories, grid_seed, shuffling, cell, code, learning_rate]
    add_data.append(copy.deepcopy(data_sing))
    print(traj)
    print(grid_seed)
    print(shuffling)
    print(cell)
    print(code)



    


# load pickled perceptron results

# data = pickle.load( open( "75-15_disinhibited_perceptron_speed.pkl", "rb" ) )

# drop rows for 45 cm and 30 cm for perceptron figures
# i = df[(df.distance==45) | (df.distance==60)].index
# df = df.drop(i)


df = pd.DataFrame(data,
                  columns=['distance', 'speed', 'threshold_crossing',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type', 'learning_rate'])


grid_rate = df.groupby(['cell_type', 'code_type']).get_group(('grid', 'rate'))
grid_phase = df.groupby(['cell_type', 'code_type']).get_group(('grid', 'phase'))
granule_rate = df.groupby(['cell_type', 'code_type']).get_group(('granule', 'rate'))
granule_phase = df.groupby(['cell_type', 'code_type']).get_group(('granule', 'phase'))




'pickle perceptron results'
file_name = "75-60_"+str(tuning)+"_perceptron_speed.pkl"
open_file = open(file_name, "wb")
pickle.dump(data, open_file)
open_file.close()

import copy
data1 = copy.deepcopy(data)


fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col")

fig1.suptitle(
    "Perceptron learning speed \n  "
    + str(n_samples)
    + " samples per trajectory, "
    + str(len(grid_seeds))
    + " grid seeds \n\n rate code learning rate = "
    + format(rate_learning_rate, ".1E")
    + "\n phase code learning rate = "
    + format(phase_learning_rate, ".1E")
)

# for ax in fig1.axes:
#     ax.set_ylabel("Speed (1/N)")

ax1.set_title("grid rate code")
ax2.set_title("grid phase code")
ax3.set_title("granule rate code")
ax4.set_title("granule phase code")


sns.lineplot(ax=ax1,
             data=grid_rate, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd'
             )
sns.lineplot(ax=ax2,
             data=grid_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )
sns.lineplot(ax=ax3,
             data=granule_rate, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd'
             )
sns.lineplot(ax=ax4,
             data=granule_phase, x="distance", y="speed",
             hue="shuffling", err_style="bars", ci='sd',
             )


plt.tight_layout()



# =============================================================================
# =============================================================================
# # Perceptron
# =============================================================================
# =============================================================================


# =============================================================================
# =============================================================================
# # 'pearson r'
# =============================================================================
# =============================================================================

from scipy import stats
from scipy.stats import pearsonr,  spearmanr

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
                    for poisson in range(r_input_code.shape[2]):
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

# mean_r = pearson_r.groupby(['grid_seed']).mean()

# mean_r['s_delta_rate'] = mean_r['s_grid_rate']-mean_r['s_granule_rate']
# mean_r['s_delta_phase'] = mean_r['s_grid_phase']-mean_r['s_granule_phase']
# mean_r['ns_delta_rate'] = mean_r['ns_grid_rate']-mean_r['ns_granule_rate']
# mean_r['ns_delta_phase'] = mean_r['ns_grid_phase']-mean_r['ns_granule_phase']

ax1 = sns.barplot(x='code', y='mean $\u0394R_{out}$', hue='tuning', data='mean_r')#, ax=axis)
ax1.set(ylim=(-0.05,0.28))





fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig2.suptitle(
    "Pearson's R \n 10 grid seeds")
from matplotlib.colors import SymLogNorm, PowerNorm, Normalize
ax1.set_title("rate shuffled")
ax2.set_title("rate nonshuffled")
ax3.set_title("phase shuffled")
ax4.set_title("phase nonshuffled")


hue = list(pearson_r['distance'])

sns.set()
sns.scatterplot(ax=ax1,
             data=pearson_r, x="s_grid_rate", y="s_granule_rate",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5)
sns.scatterplot(ax=ax2,
             data=pearson_r, x="ns_grid_rate", y="ns_granule_rate",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5)
sns.scatterplot(ax=ax3,
             data=pearson_r, x="s_grid_phase", y="s_granule_phase",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5
             )
sns.scatterplot(ax=ax4,
             data=pearson_r, x="ns_grid_phase", y="ns_granule_phase",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5
             )
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

ax1.plot(s_rate[0][0], s_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=3)
ax2.plot(ns_rate[0][0], ns_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=3)
ax3.plot(s_phase[0][0], s_phase[0][1], 'k', linestyle=(0, (6, 1)), linewidth=3)
ax4.plot(ns_phase[0][0], ns_phase[0][1], 'k', linestyle=(0, (6, 2)), linewidth=3)


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
# =============================================================================
# # 'pearson r'
# =============================================================================
# =============================================================================


# =============================================================================
# =============================================================================
# # 'spearman rho'
# =============================================================================
# =============================================================================

from scipy import stats
from scipy.stats import pearsonr,  spearmanr

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
                    for poisson in range(r_input_code.shape[2]):
                        compared_traj = r_input_code[:, poisson, traj_idx]
                        compared_traj = np.concatenate((
                            compared_traj[1000*f:1200*f],
                            compared_traj[5000*f:5200*f]))
                        spearman_r = spearmanr(baseline_traj, compared_traj)[0]
                        traj = trajectories[traj_idx]
                        idx = 75 - traj
                        comp_trajectories = str(75)+'_'+str(traj)
                        r_data_sing = [idx, spearman_r, poisson,
                                     comp_trajectories, grid_seed, shuffling,
                                     cell, code]
                        r_data.append(copy.deepcopy(r_data_sing))

df = pd.DataFrame(r_data,
                  columns=['distance', 'spearman_r', 'poisson_seed',
                           'trajectories', 'grid_seed',
                           'shuffling', 'cell_type', 'code_type'])

df = df.drop(columns='trajectories')



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
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5)
sns.scatterplot(ax=ax2,
             data=spearman_r, x="ns_grid_rate", y="ns_granule_rate",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5)
sns.scatterplot(ax=ax3,
             data=spearman_r, x="s_grid_phase", y="s_granule_phase",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5
             )
sns.scatterplot(ax=ax4,
             data=spearman_r, x="ns_grid_phase", y="ns_granule_phase",
             hue=hue, hue_norm=SymLogNorm(1), palette='OrRd', s=5
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

ax1.plot(s_rate[0][0], s_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=3)
ax2.plot(ns_rate[0][0], ns_rate[0][1], 'k', linestyle=(0, (6, 2)), linewidth=3)
ax3.plot(s_phase[0][0], s_phase[0][1], 'k', linestyle=(0, (6, 1)), linewidth=3)
ax4.plot(ns_phase[0][0], ns_phase[0][1], 'k', linestyle=(0, (6, 2)), linewidth=3)


plt.tight_layout()

# =============================================================================
# =============================================================================
# # 'spearman rho'
# =============================================================================
# =============================================================================