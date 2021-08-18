
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:53:37 2021

@author: baris
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.stats import pearsonr,  spearmanr


import glob
import numpy as np


'''PUT RESULTS FROM DIFF SEEDS TOGETHER''' 

import numpy as np
import os, glob

path = '/home/baris/results/perceptron_th_n_codes/71-70-65-60'
npzfiles = []
grid_rate_code =[]
grid_phase_code =[]
grid_complex_code =[]
grid_th_cross = []
gra_rate_code = []
gra_phase_code= []
gra_complex_code= []
gra_th_cross= []
for file in sorted(glob.glob(os.path.join(path,'*.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    grid_rate_code.append(load['grid_rate_code'])
    grid_phase_code.append(load['grid_phase_code'])
    grid_complex_code.append(load['grid_complex_code'])
    # grid_th_cross.append(load['grid_th_cross'])
    gra_rate_code.append(load['gra_rate_code'])
    gra_phase_code.append(load['gra_phase_code'])
    gra_complex_code.append(load['gra_complex_code'])
    # gra_th_cross.append(load['gra_th_cross'])
    
sim_traj = load['sim_traj']
diff_traj = load['diff_traj']
np.savez('rate_n_phase_codes_traj_'+str(sim_traj[0])+'-'+str(sim_traj[1])+'-'+str(diff_traj[0])+'-'+str(diff_traj[1])+'_net-seeds_410-419_'+str(dur_ms)+'ms', 
         grid_rate_code = grid_rate_code,
         grid_phase_code = grid_phase_code,
         grid_complex_code = grid_complex_code,
         
         gra_rate_code = gra_rate_code,
         gra_phase_code = gra_phase_code,
         gra_complex_code = gra_complex_code)






'''INPUT EXPLORER'''

data = np.load("rate_n_phase_codes_perceptron_traj_70-65_2000ms_net-seeds_410-419.npz", allow_pickle=True)


rate_code_sim = data['grid_rate_code'][0,1,:,0]
phase_code_sim = data['grid_phase_code'][0,1,:,0]
complex_code_sim = data['grid_complex_code'][0,1,:,0]
mean_rate = np.mean(rate_code_sim)
mean_phase = np.mean(np.abs(phase_code_sim))
mean_complex = np.mean(np.abs(complex_code_sim))


fig, ax = plt.subplots()
ax.plot(rate_code_sim, alpha=0.3)
ax.plot(phase_code_sim, alpha=0.3)
ax.plot(complex_code_sim, alpha=0.3)
ax.legend(("Rate", "Phase", "Complex"))


rate_code_sim = data['grid_rate_code'][:,:,:,0]
rate_code_diff = data['grid_rate_code'][:,:,:,1]
phase_code_sim = data['grid_phase_code'][:,:,:,0]
phase_code_diff = data['grid_phase_code'][:,:,:,1]
complex_code_sim = data['grid_complex_code'][:,:,:,0]
complex_code_diff = data['grid_complex_code'][:,:,:,1]


rate_code_sim = data['gra_rate_code'][:,:,:,0]
rate_code_diff = data['gra_rate_code'][:,:,:,1]
phase_code_sim = data['gra_phase_code'][:,:,:,0]
phase_code_diff = data['gra_phase_code'][:,:,:,1]
complex_code_sim = data['gra_complex_code'][:,:,:,0]
complex_code_diff = data['gra_complex_code'][:,:,:,1]

codes = [rate_code_sim, rate_code_diff, phase_code_sim, phase_code_diff, complex_code_sim, complex_code_diff]


# Calculate mean pearson R
codes_name = ['rate_code_sim', 'rate_code_diff', 
              'phase_code_sim','phase_code_diff', 
              'complex_code_sim', 'complex_code_diff']
corr_dict = {}
for idx, code in enumerate(codes):
    corr_matrices = []
    for grid in range(code.shape[0]):
        corr_matrix = []
        for traj_l in range(code.shape[1]//2):
            for traj_r in range(code.shape[1]//2,
                                code.shape[1]):

                pr = pearsonr(code[grid,traj_l,:],
                              code[grid,traj_r,:]
                               )[0]
                corr_matrix.append(pr)
        corr_matrices.append(corr_matrix)
    corr_dict[codes_name[idx]] = [np.array(x).mean() for x in corr_matrices]

df = pd.DataFrame.from_dict(corr_dict)
df['grid_seed'] = df.index
df = df.melt(id_vars=["grid_seed"], var_name = "code_sim", value_name = "pearsonr")

codings = df['code_sim'].str.split(pat='_', expand=True)
codings.columns = ["code", "whatever", "similarity"]
#codings['grid_seed'] = df['grid_seed']
df = df.join(codings, how='inner')
plt.plot()
plt.close('all')
sns.violinplot(x='code', y='pearsonr', hue="similarity", data =df)





'''BARPLOTS'''


data = np.load("rate_n_phase_codes_perceptron_traj_70-65_2000ms_net-seeds_410-419.npz", allow_pickle=True)

n_seeds = 10
th_grid = data['grid_th_cross'].T.reshape(6*n_seeds)
th_gra = data['gra_th_cross'].T.reshape(6*n_seeds)
speed_grid = 1/th_grid
speed_gra = 1/th_gra

hue = (['similar']*n_seeds+ ['distinct']*n_seeds)*3
code = ['rate']*n_seeds*2 +['phase']*n_seeds*2+['complex']*n_seeds*2


grid_df = pd.DataFrame({'threshold cross': th_grid,
                   'encoding speed': speed_grid,
                   'trajectories': pd.Categorical(hue), 
                   'code': pd.Categorical(code)})

gra_df = pd.DataFrame({'threshold cross': th_gra,
                   'encoding speed': speed_gra,
                   'trajectories': pd.Categorical(hue), 
                   'code': pd.Categorical(code)})


"BARPLOT"

n_seeds = 10
th_grid = data['grid_th_cross']
th_gra = data['gra_th_cross']
speed_grid = 1/th_grid
speed_gra = 1/th_gra

# lr_grid = data['lr_grid']
# lr_gra = data['lr_gra']

#grid th cross
plt.close('all')
ax1 = sns.barplot(x='code', y='threshold cross', hue='trajectories', data=grid_df, order=['rate', 'phase', 'complex'])
plt.title('Threshold Crossing for Grid Codes in 2000ms')
parameters = ('lr = 5*$10^{-4}$,  10 Grid seeds, 5 Poisson seeds,'+ 
              '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

width = ax1.patches[0].get_width()

for idx in range(n_seeds):
    loc = width/2
    rate_sim = th_grid[idx,0]
    rate_diff = th_grid[idx,1]
    phase_sim = th_grid[idx,2]
    phase_diff = th_grid[idx,3]
    complex_sim = th_grid[idx,4]
    complex_diff = th_grid[idx,5]
    sns.lineplot(x=[-loc,loc], y=[rate_diff, rate_sim], color='k')
    sns.lineplot(x=[1-loc,1+loc], y=[phase_diff, phase_sim], color='k')
    sns.lineplot(x=[2-loc,2+loc], y=[complex_diff, complex_sim], color='k')

#gra threshold cross
plt.figure()
ax1 = sns.barplot(x='code', y='threshold cross', hue='trajectories', data=gra_df, order=['rate', 'phase', 'complex'])
plt.title('Threshold Crossing for Granule Codes in 2000ms')
parameters = ('lr = 5*$10^{-3}$,  10 Grid seeds, 5 Poisson seeds,'+ 
              '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

width = ax1.patches[0].get_width()
for idx in range(n_seeds):
    loc = width/2
    rate_sim = th_gra[idx,0]
    rate_diff = th_gra[idx,1]
    phase_sim = th_gra[idx,2]
    phase_diff = th_gra[idx,3]
    complex_sim = th_gra[idx,4]
    complex_diff = th_gra[idx,5]
    sns.lineplot(x=[-loc,loc], y=[rate_diff, rate_sim], color='k')
    sns.lineplot(x=[1-loc,1+loc], y=[phase_diff, phase_sim], color='k')
    sns.lineplot(x=[2-loc,2+loc], y=[complex_diff, complex_sim], color='k')
    
    
    
    
    
'''PEARSON'S R'''
import os, glob
import matplotlib.pyplot as plt

def pearson_r(x,y):
    #corr mat is doubled in each axis since it is 2d*2d
    corr_mat = np.corrcoef(x, y) 
    #slice out the 1 of 4 identical mat
    corr_mat = corr_mat[int(corr_mat.shape[0]/2):, :int(corr_mat.shape[0]/2)] 
    # indices in upper triangle
    iu =np.triu_indices(int(corr_mat.shape[0]), k=1)
    # corr arr is the values vectorized 
    diag_low = corr_mat[iu]
    diag = corr_mat.diagonal()
    return diag_low

path = '/home/baris/results/perceptron_th_n_codes/'
data = np.load(os.path.join(path,"rate_n_phase_codes_traj_75.0-74.5-74.0-73.5_net-seeds_410-419_2000ms.npz"), allow_pickle=True)
data1 = np.load(os.path.join(path,"rate_n_phase_codes_traj_73.0-72.5-72.0-71.5_net-seeds_410-419_2000ms.npz"), allow_pickle=True)
data2 = np.load(os.path.join(path,"rate_n_phase_codes_traj_71-70-65-60_net-seeds_410-419_2000ms.npz"), allow_pickle=True)



'GRID'
#rate
grid_rate_75 = data['grid_rate_code'][:,:5,:,0]
grid_rate_745 = data1['grid_rate_code'][:,5:10,:,0]
grid_rate_74 = data2['grid_rate_code'][:,5:10,:,0]
grid_rate_74 = data2['grid_rate_code'][:,5:10,:,0]
grid_rate_73 = data2['grid_rate_code'][:,5:10,:,1]
grid_rate_72 = data3['grid_rate_code'][:,5:10,:,0]
grid_rate_71 = data3['grid_rate_code'][:,5:10,:,1]
grid_rate_70 = data['grid_rate_code'][:,5:10,:,0]
grid_rate_65 = data['grid_rate_code'][:,5:10,:,1]
grid_rate_60 = data1['grid_rate_code'][:,5:10,:,1]

grid_rate = np.stack((grid_rate_75,grid_rate_745,grid_rate_74,grid_rate_73,grid_rate_72,grid_rate_71,grid_rate_70,grid_rate_65,grid_rate_60))

#phase
grid_phase_75 = data['grid_phase_code'][:,:5,:,0]
grid_phase_745 = data1['grid_phase_code'][:,5:10,:,0]
grid_phase_74 = data2['grid_phase_code'][:,5:10,:,0]
grid_phase_73 = data2['grid_phase_code'][:,5:10,:,1]
grid_phase_72 = data3['grid_phase_code'][:,5:10,:,0]
grid_phase_71 = data3['grid_phase_code'][:,5:10,:,1]
grid_phase_70 = data['grid_phase_code'][:,5:10,:,0]
grid_phase_65 = data['grid_phase_code'][:,5:10,:,1]
grid_phase_60 = data1['grid_phase_code'][:,5:10,:,1]
grid_phase = np.stack((grid_phase_75,grid_phase_745,grid_phase_74,grid_phase_73,grid_phase_72,grid_phase_71,grid_phase_70,grid_phase_65,grid_phase_60))

#complex
grid_complex_75 = data['grid_complex_code'][:,:5,:,0]
grid_complex_745 = data1['grid_complex_code'][:,5:10,:,0]
grid_complex_74 = data2['grid_complex_code'][:,5:10,:,0]
grid_complex_73 = data2['grid_complex_code'][:,5:10,:,1]
grid_complex_72 = data3['grid_complex_code'][:,5:10,:,0]
grid_complex_71 = data3['grid_complex_code'][:,5:10,:,1]
grid_complex_70 = data['grid_complex_code'][:,5:10,:,0]
grid_complex_65 = data['grid_complex_code'][:,5:10,:,1]
grid_complex_60 = data1['grid_complex_code'][:,5:10,:,1]
grid_complex = np.stack((grid_complex_75,grid_complex_745,grid_complex_74,grid_complex_73,grid_complex_72,grid_complex_71,grid_complex_70,grid_complex_65,grid_complex_60))

'GRANULE'

gra_rate_75 = data['gra_rate_code'][:,:5,:,0]
gra_rate_745 = data1['gra_rate_code'][:,5:10,:,0]
gra_rate_74 = data2['gra_rate_code'][:,5:10,:,0]
gra_rate_73 = data2['gra_rate_code'][:,5:10,:,1]
gra_rate_72 = data3['gra_rate_code'][:,5:10,:,0]
gra_rate_71 = data3['gra_rate_code'][:,5:10,:,1]
gra_rate_70 = data['gra_rate_code'][:,5:10,:,0]
gra_rate_65 = data['gra_rate_code'][:,5:10,:,1]
gra_rate_60 = data1['gra_rate_code'][:,5:10,:,1]
gra_rate = np.stack((gra_rate_75,gra_rate_745,gra_rate_74,gra_rate_73, gra_rate_72,gra_rate_71,gra_rate_70,gra_rate_65,gra_rate_60))

#phase
gra_phase_75 = data['gra_phase_code'][:,:5,:,0]
gra_phase_745 = data1['gra_phase_code'][:,5:10,:,0]
gra_phase_74 = data2['gra_phase_code'][:,5:10,:,0]
gra_phase_73 = data2['gra_phase_code'][:,5:10,:,1]
gra_phase_72 = data3['gra_phase_code'][:,5:10,:,0]
gra_phase_71 = data3['gra_phase_code'][:,5:10,:,1]
gra_phase_70 = data['gra_phase_code'][:,5:10,:,0]
gra_phase_65 = data['gra_phase_code'][:,5:10,:,1]
gra_phase_60 = data1['gra_phase_code'][:,5:10,:,1]
gra_phase = np.stack((gra_phase_75,gra_phase_745,gra_phase_74,gra_phase_73,gra_phase_72,gra_phase_71,gra_phase_70,gra_phase_65,gra_phase_60))

#complex
gra_complex_75 = data['gra_complex_code'][:,:5,:,0]
gra_complex_745 = data1['gra_complex_code'][:,5:10,:,0]
gra_complex_74 = data2['gra_complex_code'][:,5:10,:,0]
gra_complex_73 = data2['gra_complex_code'][:,5:10,:,1]
gra_complex_72 = data3['gra_complex_code'][:,5:10,:,0]
gra_complex_71 = data3['gra_complex_code'][:,5:10,:,1]
gra_complex_70 = data['gra_complex_code'][:,5:10,:,0]
gra_complex_65 = data['gra_complex_code'][:,5:10,:,1]
gra_complex_60 = data1['gra_complex_code'][:,5:10,:,1]
gra_complex = np.stack((gra_complex_75,gra_complex_745,gra_complex_74,gra_complex_73,gra_complex_72,gra_complex_71,gra_complex_70,gra_complex_65,gra_complex_60))


n_gra = 2000
n_grid = 200
n_bin = 20 
# one bin

grid_rate = np.concatenate((grid_rate[:,:,:,3800:4000], grid_rate[:,:,:,7800:8000]), axis=3)
grid_phase = np.concatenate((grid_phase[:,:,:,3800:4000], grid_phase[:,:,:,7800:8000]), axis=3)
grid_complex = np.concatenate((grid_complex[:,:,:,3800:4000], grid_complex[:,:,:,7800:8000]), axis=3)
gra_rate = np.concatenate((gra_rate[:,:,:,38000:40000], gra_rate[:,:,:,78000:80000]), axis=3)
gra_phase = np.concatenate((gra_phase[:,:,:,38000:40000], gra_phase[:,:,:,78000:80000]), axis=3)
gra_complex = np.concatenate((gra_complex[:,:,:,38000:40000], gra_complex[:,:,:,78000:80000]), axis=3)



rate_grid_corr = np.zeros((10,5,36))
rate_gra_corr = np.zeros((10,5,36))
phase_grid_corr = np.zeros((10,5,36))
phase_gra_corr = np.zeros((10,5,36))
complex_grid_corr = np.zeros((10,5,36))
complex_gra_corr = np.zeros((10,5,36))

# fig, (ax_rate, ax_phase, ax_complex) = plt.subplots(1,3, sharey=True)

plt.close('all')

fig1, ax_rate = plt.subplots(1,1)
fig2, ax_phase = plt.subplots(1,1)
fig3, ax_complex = plt.subplots(1,1)

for grid in range(10):
    for poiss in range(5):
        rate_grid_corr[grid, poiss, :] = pearson_r(grid_rate[:,grid,poiss,:], grid_rate[:,grid,poiss,:])
        rate_gra_corr[grid, poiss, :] = pearson_r(gra_rate[:,grid,poiss,:], gra_rate[:,grid,poiss,:])
        phase_grid_corr[grid, poiss, :] = pearson_r(grid_phase[:,grid,poiss,:], grid_phase[:,grid,poiss,:])
        phase_gra_corr[grid, poiss, :] = pearson_r(gra_phase[:,grid,poiss,:], gra_phase[:,grid,poiss,:])
        complex_grid_corr[grid, poiss, :] = pearson_r(grid_complex[:,grid,poiss,:], grid_complex[:,grid,poiss,:])
        complex_gra_corr[grid, poiss, :] = pearson_r(gra_complex[:,grid,poiss,:], gra_complex[:,grid,poiss,:])
        ax_rate.scatter(rate_grid_corr[grid, poiss, :], rate_gra_corr[grid, poiss, :], c='b')
        ax_phase.scatter(phase_grid_corr[grid, poiss, :], phase_gra_corr[grid, poiss, :], c='b')
        ax_complex.scatter(complex_grid_corr[grid, poiss, :], complex_gra_corr[grid, poiss, :], c='b')

  
# mean_rate_grid = np.mean(rate_grid_corr, axis=tuple([0,1]))     
# mean_rate_gra = np.mean(rate_gra_corr, axis=tuple([0,1]))   
# mean_phase_grid = np.mean(phase_grid_corr, axis=tuple([0,1]))        
# mean_phase_gra = np.mean(phase_gra_corr, axis=tuple([0,1])) 
# mean_complex_grid = np.mean(complex_grid_corr, axis=tuple([0,1]))        
# mean_complex_gra = np.mean(complex_gra_corr, axis=tuple([0,1]))       



mean_rate_grid = np.mean(rate_grid_corr, axis=1).flatten()           
mean_rate_gra = np.mean(rate_gra_corr, axis=1).flatten()   
mean_phase_grid = np.mean(phase_grid_corr, axis=1).flatten()        
mean_phase_gra = np.mean(phase_gra_corr, axis=1).flatten()    
mean_complex_grid = np.mean(complex_grid_corr, axis=1).flatten()        
mean_complex_gra = np.mean(complex_gra_corr, axis=1).flatten()       

# mean_rate_grid = np.mean(rate_grid_corr, axis=0).flatten()           
# mean_rate_gra = np.mean(rate_gra_corr, axis=0).flatten()   
# mean_phase_grid = np.mean(phase_grid_corr, axis=0).flatten()        
# mean_phase_gra = np.mean(phase_gra_corr, axis=0).flatten()    
# mean_complex_grid = np.mean(complex_grid_corr, axis=0).flatten()        
# mean_complex_gra = np.mean(complex_gra_corr, axis=0).flatten()       


# for one bin
ax_rate.set_title('Rate Code Pearson R for 100ms time bin')
ax_rate.set_xlabel('Rin')
ax_rate.set_ylabel('Rout')
ax_rate.set_aspect('equal')
ax_rate.set_xlim(-0.15,0.7)
ax_rate.set_ylim(-0.15,0.7)
ax_rate.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'g--')
ax_phase.set_title('Phase Code Pearson R for 100ms time bin')
ax_phase.set_xlabel('Rin')
ax_phase.set_ylabel('Rout')
ax_phase.set_aspect('equal')
ax_phase.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'g--')
ax_complex.set_title('Complex Code Pearson R for 100ms time bin')
ax_complex.set_xlabel('Rin')
ax_complex.set_ylabel('Rout')
ax_complex.set_aspect('equal')
ax_complex.plot(np.arange(0,0.7,0.1),np.arange(0,0.7,0.1),'g--')

ax_rate.plot(mean_rate_grid, mean_rate_gra, 'kX', label='mean (Poisson)')
ax_phase.plot(mean_phase_grid, mean_phase_gra, 'kX', label='mean (Poisson)')
ax_complex.plot(mean_complex_grid, mean_complex_gra, 'kX', label='mean (Poisson)')

ax_rate.legend()
ax_phase.legend()
ax_complex.legend()


# in case you wanna use diff perc seeds in one console, put this part into a loop

#Input generation
#Rate trajs with phase info; oscillations implemented in the rate profile
grid_phases_sim,  gra_phases_sim, grid_spikes_sim, gra_spikes_sim, rate_trajs_sim, dt_s, theta_phase_sim, phase_code_dir_sim, overall_dir_sim, traj_dist_dir_sim, dist_trajs_sim, direction_sim, spacing_sim, grids_sim = phase_code(sim_traj, dur_ms, grid_seeds, poiss_seeds, pp_weight)
grid_phases_diff, gra_phases_diff, grid_spikes_diff, gra_spikes_diff, rate_trajs_diff, dt_s, theta_phase_diff, phase_code_dir_diff, overall_dir_diff, traj_dist_dir_diff, dist_trajs_diff, direction_diff, spacing_diff, grids_diff = phase_code(diff_traj, dur_ms, grid_seeds, poiss_seeds, pp_weight)
#grid and granule spike counts \ rate codes
grid_sim_traj_cts, gra_sim_traj_cts = overall_spike_ct(grid_spikes_sim, gra_spikes_sim, dur_ms, poiss_seeds, n_traj=n_traj)
grid_diff_traj_cts, gra_diff_traj_cts = overall_spike_ct(grid_spikes_diff, gra_spikes_diff, dur_ms, poiss_seeds, n_traj=n_traj)

lr_grid = 5e-4
lr_gra = 5e-4 #was 5e-3 and good for 500ms, and for 2000ms 5e-4 was set
grid_rate_code, grid_phase_code, grid_complex_code = code_maker(grid_sim_traj_cts, grid_diff_traj_cts, grid_phases_sim, grid_phases_diff)
gra_rate_code, gra_phase_code, gra_complex_code = code_maker(gra_sim_traj_cts, gra_diff_traj_cts, gra_phases_sim, gra_phases_diff)

rate_1 = np.concatenate((grid_rate_code[0,:200,0], grid_rate_code[0,400:600,0]))
rate_2 = np.concatenate((grid_rate_code[2,:200,0], grid_rate_code[2,400:600,0]))
pearsonr(rate_1, rate_2)

phase_1 = np.concatenate((grid_phase_code[0,:200,0], grid_phase_code[0,400:600,0]))
phase_2 = np.concatenate((grid_phase_code[2,:200,0], grid_phase_code[2,400:600,0]))
pearsonr(phase_1, phase_2)

plt.figure()
plt.plot(grid_rate[0,0,0,:])
plt.plot(grid_rate[1,0,0,:])

grid_r = pearson_r(grid_rate[:,2,0,:],grid_rate[:,2,0,:])
gra_r = 
pearson_r(gra_rate[:,7,4,:],gra_rate[:,7,4,:])


plt.figure()
plt.plot(gra_rate[0,8,0,:])
plt.plot(gra_rate[1,8,0,:])

pearsonr(grid_rate[0,3,0,:],grid_rate[1,3,0,:])

spearmanr(grid_rate[0,3,0,:],grid_rate[1,3,0,:])



plt.figure()
plt.eventplot(grid_spikes_sim[0][1,0])

plt.figure()
plt.plot(grid_sim_traj_cts[2,:])

plt.figure()
plt.plot(gra_sim_traj_cts[0,:])



load20 = np.load('/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/rate_n_phase_traj_diff_poiss_75.0-74.5-74.0-73.5_net-seeds_410-429_2000ms.npz')

grid_rate = load20['grid_rate_code']

