#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:31:58 2021

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

dur_ms = 2000
path = '/home/baris/results/perceptron_th_n_codes/'
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
np.savez(os.path.join(path,'rate_n_phase_traj_'+str(sim_traj[0])+'-'+str(sim_traj[1])+'-'+str(diff_traj[0])+'-'+str(diff_traj[1])+'_net-seeds_410-419_'+str(dur_ms)+'ms'), 
         grid_rate_code = grid_rate_code,
         grid_phase_code = grid_phase_code,
         grid_complex_code = grid_complex_code,
         grid_th_cross = grid_th_cross,
         
         gra_rate_code = gra_rate_code,
         gra_phase_code = gra_phase_code,
         gra_complex_code = gra_complex_code,
         gra_th_cross = gra_th_cross)






'''INPUT EXPLORER'''


single_data = np.load(npzfiles[0], allow_pickle=True)




gra_spikes = single_data['gra_spikes_sim'][0][:,0]

gra_cts = single_data['gra_sim_traj_cts'][0,34000:36000]


gra_cts = single_data['gra_diff_traj_cts'][1,:]

np.sum(gra_cts!=0)/2000


data = np.load(os.path.join(path,"rate_n_phase_traj_75.0-74.5-74.0-73.5_net-seeds_410-419_2000ms.npz"), allow_pickle=True)


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



fig, ax = plt.subplots()
ax.plot(grid_rate[1,1,1,:], alpha=0.3)
ax.plot(grid_phase[1,1,1,:], alpha=0.3)
ax.plot(grid_complex[1,1,1,:], alpha=0.3)
ax.legend(("Rate", "Phase", "Complex"))

grid_rate.shape
grid_phase
grid_phase


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


data = np.load("rate_n_phase_perceptron_traj_72-71_net-seeds_410-419_2000ms.npz", allow_pickle=True)

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

# path1 = '/home/baris/results/perceptron_th_n_codes/same_poiss/74-73.5-73-72.5'
# path2 = '/home/baris/results/perceptron_th_n_codes/same_poiss/72-71-65-60'
# data1 = np.load(os.path.join(path1,"rate_n_phase_traj_74.0-73.5-73.0-72.5_net-seeds_410-419_2000ms.npz"), allow_pickle=True)
# data2 = np.load(os.path.join(path2,"rate_n_phase_traj_72-71-65-60_net-seeds_410-419_2000ms.npz"), allow_pickle=True)

path = '/home/baris/results/perceptron_th_n_codes/'
data1 = np.load(os.path.join(path,"rate_n_phase_traj_75.0-74.5-74.0-73.5_net-seeds_410-419_2000ms.npz"), allow_pickle=True)
data2 = np.load(os.path.join(path,"rate_n_phase_traj_73.0-72.5-72.0-71.5_net-seeds_410-419_2000ms.npz"), allow_pickle=True)
data3 = np.load(os.path.join(path,"rate_n_phase_traj_71-70-65-60_net-seeds_410-419_2000ms.npz"), allow_pickle=True)



data=[data1, data2, data3]
grid_rate = []
grid_phase = []
grid_complex = []
gra_rate = []
gra_phase=[]
gra_complex=[]
for datum in data:
    for i in range(2):
        grid_rate.append(datum['grid_rate_code'][:,:5,:,i])
        grid_rate.append(datum['grid_rate_code'][:,5:10,:,i])
        grid_phase.append(datum['grid_phase_code'][:,:5,:,i])
        grid_phase.append(datum['grid_phase_code'][:,5:10,:,i])
        grid_complex.append(datum['grid_complex_code'][:,:5,:,i])
        grid_complex.append(datum['grid_complex_code'][:,5:10,:,i])
        
        gra_rate.append(datum['gra_rate_code'][:,:5,:,i])
        gra_rate.append(datum['gra_rate_code'][:,5:10,:,i])
        gra_phase.append(datum['gra_phase_code'][:,:5,:,i])
        gra_phase.append(datum['gra_phase_code'][:,5:10,:,i])
        gra_complex.append(datum['gra_complex_code'][:,:5,:,i])
        gra_complex.append(datum['gra_complex_code'][:,5:10,:,i])
        
grid_rate = np.array(grid_rate)
grid_phase = np.array(grid_phase)
grid_complex = np.array(grid_complex)
gra_rate = np.array(gra_rate)
gra_phase = np.array(gra_phase)
gra_complex = np.array(gra_complex)

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

grid_rate = np.concatenate((grid_rate[:,:,:,1000:1200], grid_rate[:,:,:,5000:5200]), axis=3)
grid_phase = np.concatenate((grid_phase[:,:,:,1000:1200], grid_phase[:,:,:,5000:5200]), axis=3)
grid_complex = np.concatenate((grid_complex[:,:,:,1000:1200], grid_complex[:,:,:,5000:5200]), axis=3)
gra_rate = np.concatenate((gra_rate[:,:,:,10000:12000], gra_rate[:,:,:,50000:52000]), axis=3)
gra_phase = np.concatenate((gra_phase[:,:,:,10000:12000], gra_phase[:,:,:,50000:52000]), axis=3)
gra_complex = np.concatenate((gra_complex[:,:,:,10000:12000], gra_complex[:,:,:,50000:52000]), axis=3)


n_comp = int(grid_rate.shape[0]*(grid_rate.shape[0]-1)/2)
rate_grid_corr = np.zeros((10,5,n_comp))
rate_gra_corr = np.zeros((10,5,n_comp))
phase_grid_corr = np.zeros((10,5,n_comp))
phase_gra_corr = np.zeros((10,5,n_comp))
complex_grid_corr = np.zeros((10,5,n_comp))
complex_gra_corr = np.zeros((10,5,n_comp))

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



for grid in range(10):
    for poiss in range(5):
        grid_rate_corr = pearson_r(grid_rate[:,grid,poiss,:], grid_rate[:,grid,poiss,:])[sort]
        grid_phase_corr = pearson_r(grid_phase[:,grid,poiss,:], grid_phase[:,grid,poiss,:])[sort]
        grid_complex_corr = pearson_r(grid_complex[:,grid,poiss,:], grid_complex[:,grid,poiss,:])[sort]
        gra_rate_corr = pearson_r(gra_rate[:,grid,poiss,:], gra_rate[:,grid,poiss,:])[sort]
        gra_phase_corr = pearson_r(gra_phase[:,grid,poiss,:], gra_phase[:,grid,poiss,:])[sort]
        gra_complex_corr = pearson_r(gra_complex[:,grid,poiss,:], gra_complex[:,grid,poiss,:])[sort]
        rate_grid_corr[grid, poiss, :] = grid_rate_corr
        rate_gra_corr[grid, poiss, :] = gra_rate_corr
        phase_grid_corr[grid, poiss, :] = grid_phase_corr
        phase_gra_corr[grid, poiss, :] = gra_phase_corr
        complex_grid_corr[grid, poiss, :] = grid_complex_corr
        complex_gra_corr[grid, poiss, :] = gra_complex_corr
        t = np.arange(66)
        ax_rate.scatter(rate_grid_corr[grid, poiss, :], rate_gra_corr[grid, poiss, :], c=t)
        ax_phase.scatter(phase_grid_corr[grid, poiss, :], phase_gra_corr[grid, poiss, :], c=t)
        ax_complex.scatter(complex_grid_corr[grid, poiss, :], complex_gra_corr[grid, poiss, :], c=t)



deltaR_rate = np.mean(rate_grid_corr-rate_gra_corr, axis=2)

deltaR_phase = np.mean(phase_grid_corr-phase_gra_corr, axis=2)

deltaR_complex = np.mean(complex_grid_corr-complex_gra_corr, axis=2)





# fig, (ax_rate, ax_phase, ax_complex) = plt.subplots(1,3, sharey=True)


#for all bins

plt.close('all')

fig1, ax_rate = plt.subplots(1,1)
fig2, ax_phase = plt.subplots(1,1)
fig3, ax_complex = plt.subplots(1,1)

n_bin = 20
n_grid = 200
n_gra = 2000


n_comp = int(grid_rate.shape[0]*(grid_rate.shape[0]-1)/2)
rate_grid_corr = np.zeros((10,5,n_comp,n_bin ))
rate_gra_corr = np.zeros((10,5,n_comp,n_bin))
phase_grid_corr = np.zeros((10,5,n_comp,n_bin))
phase_gra_corr = np.zeros((10,5,n_comp,n_bin))
complex_grid_corr = np.zeros((10,5,n_comp,n_bin))
complex_gra_corr = np.zeros((10,5,n_comp,n_bin))

for grid in range(10):
    for poiss in range(5):
        for i in range(n_bin):
            idx1 = i*n_grid
            idx2 = (i+1)*n_grid
            offset1 = 4000
            idx3 = i*n_gra
            idx4 = (i+1)*n_gra
            offset2 = 40000             
            grid_rate_bin = np.concatenate((grid_rate[:,:,:,idx1:idx2], grid_rate[:,:,:,offset1+idx1:offset1+idx2]), axis=3)
            grid_phase_bin = np.concatenate((grid_phase[:,:,:,idx1:idx2], grid_phase[:,:,:,offset1+idx1:offset1+idx2]), axis=3)
            grid_complex_bin = np.concatenate((grid_complex[:,:,:,idx1:idx2], grid_complex[:,:,:,offset1+idx1:offset1+idx2]), axis=3)
            gra_rate_bin = np.concatenate((gra_rate[:,:,:,idx3:idx4], gra_rate[:,:,:,offset2+idx3:offset2+idx4]), axis=3)
            gra_phase_bin = np.concatenate((gra_phase[:,:,:,idx3:idx4], gra_phase[:,:,:,offset2+idx3:offset2+idx4]), axis=3)
            gra_complex_bin = np.concatenate((gra_complex[:,:,:,idx3:idx4], gra_complex[:,:,:,offset2+idx3:offset2+idx4]), axis=3)

            rate_grid_corr[grid, poiss,:, i] = pearson_r(grid_rate_bin[:,grid,poiss,:], grid_rate_bin[:,grid,poiss,:])
            rate_gra_corr[grid, poiss, :,i] = pearson_r(gra_rate_bin[:,grid,poiss,:], gra_rate_bin[:,grid,poiss,:])
            phase_grid_corr[grid, poiss, :,i] = pearson_r(grid_phase_bin[:,grid,poiss,:], grid_phase_bin[:,grid,poiss,:])
            phase_gra_corr[grid, poiss, :,i] = pearson_r(gra_phase_bin[:,grid,poiss,:], gra_phase_bin[:,grid,poiss,:])
            complex_grid_corr[grid, poiss, :,i] = pearson_r(grid_complex_bin[:,grid,poiss,:], grid_complex_bin[:,grid,poiss,:])
            complex_gra_corr[grid, poiss, :,i] = pearson_r(gra_complex_bin[:,grid,poiss,:], gra_complex_bin[:,grid,poiss,:])
            ax_rate.scatter(rate_grid_corr[grid, poiss, :,i], rate_gra_corr[grid, poiss, :,i], s=0.1, c='b')
            ax_phase.scatter(phase_grid_corr[grid, poiss, :,i], phase_gra_corr[grid, poiss, :,i], s=0.1, c='b')
            ax_complex.scatter(complex_grid_corr[grid, poiss, :,i], complex_gra_corr[grid, poiss, :,i], s=0.1, c='b')




  
mean_rate_grid = np.mean(rate_grid_corr, axis=tuple([0,1])).flatten()     
mean_rate_gra = np.mean(rate_gra_corr, axis=tuple([0,1])).flatten()   
mean_phase_grid = np.mean(phase_grid_corr, axis=tuple([0,1])).flatten()        
mean_phase_gra = np.mean(phase_gra_corr, axis=tuple([0,1])).flatten() 
mean_complex_grid = np.mean(complex_grid_corr, axis=tuple([0,1])).flatten()        
mean_complex_gra = np.mean(complex_gra_corr, axis=tuple([0,1])).flatten()    


mean_phase_gra[mean_phase_gra>mean_phase_grid]
   
(mean_phase_gra>mean_phase_grid) = True

plt.figure()
plt.plot(mean_phase_gra>mean_phase_grid)

mean_rate_grid = np.mean(rate_grid_corr, axis=tuple([0,1]))[0,:]  
mean_rate_gra = np.mean(rate_gra_corr, axis=tuple([0,1]))[0,:]  
mean_phase_grid = np.mean(phase_grid_corr, axis=tuple([0,1]))[0,:]         
mean_phase_gra = np.mean(phase_gra_corr, axis=tuple([0,1]))[0,:] 
mean_complex_grid = np.mean(complex_grid_corr, axis=tuple([0,1]))[0,:]        
mean_complex_gra = np.mean(complex_gra_corr, axis=tuple([0,1]))[0,:]        

mean_rate_grid = np.mean(rate_grid_corr, axis=tuple([1])).flatten()     
mean_rate_gra = np.mean(rate_gra_corr, axis=tuple([1])).flatten()   
mean_phase_grid = np.mean(phase_grid_corr, axis=tuple([1])).flatten()        
mean_phase_gra = np.mean(phase_gra_corr, axis=tuple([1])).flatten() 
mean_complex_grid = np.mean(complex_grid_corr, axis=tuple([1])).flatten()        
mean_complex_gra = np.mean(complex_gra_corr, axis=tuple([1])).flatten()  

  
mean_rate_grid = np.mean(rate_grid_corr, axis=tuple([0,1,3]))     
mean_rate_gra = np.mean(rate_gra_corr, axis=tuple([0,1,3]))   
mean_phase_grid = np.mean(phase_grid_corr, axis=tuple([0,1,3]))        
mean_phase_gra = np.mean(phase_gra_corr, axis=tuple([0,1,3])) 
mean_complex_grid = np.mean(complex_grid_corr, axis=tuple([0,1,3]))        
mean_complex_gra = np.mean(complex_gra_corr, axis=tuple([0,1,3]))  
  
mean_rate_grid = np.mean(rate_grid_corr, axis=tuple([0,1]))     
mean_rate_gra = np.mean(rate_gra_corr, axis=tuple([0,1]))   
mean_phase_grid = np.mean(phase_grid_corr, axis=tuple([0,1]))        
mean_phase_gra = np.mean(phase_gra_corr, axis=tuple([0,1])) 
mean_complex_grid = np.mean(complex_grid_corr, axis=tuple([0,1]))        
mean_complex_gra = np.mean(complex_gra_corr, axis=tuple([0,1]))       



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
# ax_rate.set_xlim(-0.15,1)
# ax_rate.set_ylim(-0.15,1)
ax_rate.plot(np.arange(0,1,0.1),np.arange(0,1,0.1),'g--')
ax_phase.set_title('Phase Code Pearson R for 100ms time bin')
ax_phase.set_xlabel('Rin')
ax_phase.set_ylabel('Rout')
ax_phase.set_aspect('equal')
ax_phase.plot(np.arange(0,1,0.1),np.arange(0,1,0.1),'g--')
ax_complex.set_title('Complex Code Pearson R for 100ms time bin')
ax_complex.set_xlabel('Rin')
ax_complex.set_ylabel('Rout')
ax_complex.set_aspect('equal')
ax_complex.plot(np.arange(0,1,0.1),np.arange(0,1,0.1),'g--')

ax_rate.plot(mean_rate_grid, mean_rate_gra, 'kX', markersize=2, label='mean')
ax_phase.plot(mean_phase_grid, mean_phase_gra, 'kX', markersize=2,label='mean')
ax_complex.plot(mean_complex_grid, mean_complex_gra, 'kX', markersize=2,label='mean')

ax_rate.legend()
ax_phase.legend()
ax_complex.legend()




plt.figure()
plt.plot(grid_rate[0,0,0,:])
plt.plot(grid_rate[1,0,0,:])

plt.figure()
plt.plot(gra_rate[0,0,0,:])
plt.plot(gra_rate[1,0,0,:])


plt.figure()
plt.plot(grid_complex[0,0,0,:])
plt.plot(grid_complex[1,0,0,:])

plt.figure()
plt.plot(gra_complex[0,0,0,:])
plt.plot(gra_complex[1,0,0,:])

grid_r = pearson_r(grid_rate[:,2,0,:],grid_rate[:,2,0,:])
gra_r = 
pearson_r(gra_rate[:,7,4,:],gra_rate[:,7,4,:])


plt.figure()
plt.plot(gra_rate[0,8,0,:])
plt.plot(gra_rate[1,8,0,:])

pearsonr(grid_rate[0,2,0,:],grid_rate[3,2,0,:])
np.mean(gra_rate)
np.sum(gra_rate)
spearmanr(gra_rate[0,3,0,:],gra_rate[1,3,0,:])





 grid_phases_sim = grid_phases_sim,
         grid_phases_diff = grid_phases_diff,
         grid_spikes_sim = grid_spikes_sim,
         grid_spikes_diff = grid_spikes_diff,
         grid_sim_traj_cts = grid_sim_traj_cts,
         grid_diff_traj_cts = grid_diff_traj_cts,
         
         grid_rate_code = grid_rate_code,
         grid_phase_code = grid_phase_code,
         grid_complex_code = grid_complex_code,
         # grid_th_cross = grid_th_cross,
         
         gra_phases_sim = gra_phases_sim,
         gra_phases_diff = gra_phases_diff,
         gra_spikes_sim = gra_spikes_sim,
         gra_spikes_diff = gra_spikes_diff,
         gra_sim_traj_cts = gra_sim_traj_cts,
         gra_diff_traj_cts = gra_diff_traj_cts,
         
         gra_rate_code = gra_rate_code,
         gra_phase_code = gra_phase_code,
         gra_complex_code = gra_complex_code,
         # gra_th_cross = gra_th_cross,


plt.figure()
plt.plot(grid_sim_traj_cts[0,:])



plt.figure()
plt.plot(gra_sim_traj_cts[0,:])


fig, ax = plt.subplots()
ax.plot(grid_rate_code[0,:,0], alpha=0.3)
ax.plot(grid_phase_code[0,:,0], alpha=0.3)
ax.plot(grid_complex_code[0,:,0], alpha=0.3)
ax.legend(("Rate", "Phase", "Complex"))



pearsonr(grid_rate_code[0,:,0][1100:1200],grid_rate_code[2,:,0][1100:1200])

pearsonr(grid_phase_code[0,:,0],grid_phase_code[1,:,0])


pearsonr(grid_complex_code[0,:,0],grid_complex_code[1,:,0])



trajectories = np.array([75, 74.5, 74, 73.5, 73, 72.5, 72, 71.5, 71, 70, 65, 60])
diff = np.subtract.outer(trajectories, trajectories)
plt.figure()
plt.imshow(np.subtract.outer(trajectories, trajectories))
plt.colorbar()

np.triu_indices(diff,1)

diff = diff[np.triu_indices(12,1)]

sort = np.argsort(diff, kind='stable')

ind1 = np.logical_and(diff>0, diff<1.1)
ind2 = np.logical_and(diff>1.4, diff<2.6)
ind3 = np.logical_and(diff>2.9, diff<7.1)
ind4 = np.logical_and(diff>7.4, diff<15.1)
