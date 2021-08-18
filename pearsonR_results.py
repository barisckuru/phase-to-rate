#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:36:54 2021

@author: baris
"""


import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.stats import pearsonr,  spearmanr
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from result_processor import all_codes


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


def binned_mean (path, n_grid_seed):
    grid_rate, grid_phase, grid_complex, gra_rate, gra_phase, gra_complex = all_codes(path)
    #one time bin
    grid_rate = np.concatenate((grid_rate[:,:,:,1000:1200], grid_rate[:,:,:,5000:5200]), axis=3)
    grid_phase = np.concatenate((grid_phase[:,:,:,1000:1200], grid_phase[:,:,:,5000:5200]), axis=3)
    grid_complex = np.concatenate((grid_complex[:,:,:,1000:1200], grid_complex[:,:,:,5000:5200]), axis=3)
    gra_rate = np.concatenate((gra_rate[:,:,:,10000:12000], gra_rate[:,:,:,50000:52000]), axis=3)
    gra_phase = np.concatenate((gra_phase[:,:,:,10000:12000], gra_phase[:,:,:,50000:52000]), axis=3)
    gra_complex = np.concatenate((gra_complex[:,:,:,10000:12000], gra_complex[:,:,:,50000:52000]), axis=3)

    n_comp = int(grid_rate.shape[0]*(grid_rate.shape[0]-1)/2)
    rate_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    rate_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    phase_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    phase_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    complex_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    complex_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    
    ###sorting - color code
    trajectories = np.array([75, 74.5, 74, 73.5, 73, 72.5, 72, 71.5, 71, 70, 65, 60])
    diff = np.subtract.outer(trajectories, trajectories)
    diff = diff[np.triu_indices(12,1)]
    sort = np.argsort(diff, kind='stable')

    for grid in range(n_grid_seed):
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
            t = diff[sort]
    rate_grid_corr = rate_grid_corr.reshape(n_grid_seed,330)
    phase_grid_corr = phase_grid_corr.reshape(n_grid_seed,330)
    complex_grid_corr = complex_grid_corr.reshape(n_grid_seed,330)
    rate_gra_corr = rate_gra_corr.reshape(n_grid_seed,330)
    phase_gra_corr = phase_gra_corr.reshape(n_grid_seed,330)
    complex_gra_corr = complex_gra_corr.reshape(n_grid_seed,330)
    bin_mean_rate = np.empty(n_grid_seed)
    bin_mean_phase = np.empty(n_grid_seed)
    bin_mean_complex = np.empty(n_grid_seed)
    
    #if diff poiss
    if path == '/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/410-419/':
        for i in range(n_grid_seed):
            rate = stats.binned_statistic(rate_grid_corr[i], list((rate_grid_corr[i], rate_gra_corr[i])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6]) #cut for diff poiss,0.7,0.8])
            phase = stats.binned_statistic(phase_grid_corr[i], list((phase_grid_corr[i], phase_gra_corr[i])), 'mean', bins=[0,0.1,0.2,0.3,0.4])#,0.5])
            compl = stats.binned_statistic(complex_grid_corr[i], list((complex_grid_corr[i], complex_gra_corr[i])), 'mean', bins=[0,0.1,0.2,0.3,0.4])#,0.5,0.6])
            bin_mean_rate[i]=np.mean(rate[0][0]-rate[0][1])
            bin_mean_phase[i]=np.mean(phase[0][0]-phase[0][1])
            bin_mean_complex[i]=np.mean(compl[0][0]-compl[0][1])
    else: 
        for i in range(n_grid_seed):
            rate = stats.binned_statistic(rate_grid_corr[i], list((rate_grid_corr[i], rate_gra_corr[i])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
            phase = stats.binned_statistic(phase_grid_corr[i], list((phase_grid_corr[i], phase_gra_corr[i])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5])
            compl = stats.binned_statistic(complex_grid_corr[i], list((complex_grid_corr[i], complex_gra_corr[i])), 'mean', bins=[0,0.1,0.2,0.3,0.4,0.5,0.6])
            bin_mean_rate[i]=np.mean(rate[0][0]-rate[0][1])
            bin_mean_phase[i]=np.mean(phase[0][0]-phase[0][1])
            bin_mean_complex[i]=np.mean(compl[0][0]-compl[0][1])
    return bin_mean_rate, bin_mean_phase, bin_mean_complex
    # return rate[0], phase[0], compl[0]

def correlation (path, n_grid_seed, fname, plot='no'):
    grid_rate, grid_phase, grid_complex, gra_rate, gra_phase, gra_complex = all_codes(path)
    
    #one time bin
    grid_rate = np.concatenate((grid_rate[:,:,:,1000:1200], grid_rate[:,:,:,5000:5200]), axis=3)
    grid_phase = np.concatenate((grid_phase[:,:,:,1000:1200], grid_phase[:,:,:,5000:5200]), axis=3)
    grid_complex = np.concatenate((grid_complex[:,:,:,1000:1200], grid_complex[:,:,:,5000:5200]), axis=3)
    gra_rate = np.concatenate((gra_rate[:,:,:,10000:12000], gra_rate[:,:,:,50000:52000]), axis=3)
    gra_phase = np.concatenate((gra_phase[:,:,:,10000:12000], gra_phase[:,:,:,50000:52000]), axis=3)
    gra_complex = np.concatenate((gra_complex[:,:,:,10000:12000], gra_complex[:,:,:,50000:52000]), axis=3)
    n_gra = 2000
    n_grid = 200
    n_bin = 20 

    n_comp = int(grid_rate.shape[0]*(grid_rate.shape[0]-1)/2)
    rate_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    rate_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    phase_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    phase_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    complex_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    complex_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    
    ###sorting - color code
    trajectories = np.array([75, 74.5, 74, 73.5, 73, 72.5, 72, 71.5, 71, 70, 65, 60])
    diff = np.subtract.outer(trajectories, trajectories)
    diff = diff[np.triu_indices(12,1)]
    sort = np.argsort(diff, kind='stable')

    
    mean_rate, mean_phase, mean_complex = binned_mean(path,1)
    m_rate_grid = mean_rate[0]
    m_rate_gra = mean_rate[1]
    m_phase_grid = mean_phase[0]
    m_phase_gra = mean_phase[1]
    m_complex_grid = mean_complex[0]
    m_complex_gra = mean_complex[1]
    
    sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=2.5,color_codes=True)
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    fig, (ax_rate, ax_phase, ax_complex) = plt.subplots(1,3, sharey=True, gridspec_kw={'width_ratios':[1,1,1]}, figsize=(15,5))
    ax_rate.plot(m_rate_grid,m_rate_gra, 'k--', linewidth=3)
    ax_phase.plot(m_phase_grid,m_phase_gra, 'k--', linewidth=3)
    ax_complex.plot(m_complex_grid,m_complex_gra, 'k--', linewidth=3)
    for grid in range(n_grid_seed):
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
            t = diff[sort]
            
            if plot=='yes':
                im2 = ax_phase.scatter(phase_grid_corr[grid, poiss, :], phase_gra_corr[grid, poiss, :], c=t, s=30, cmap=cmap)
                im3 = ax_complex.scatter(complex_grid_corr[grid, poiss, :], complex_gra_corr[grid, poiss, :], c=t, s=30, cmap=cmap)
                im1 = ax_rate.scatter(rate_grid_corr[grid, poiss, :], rate_gra_corr[grid, poiss, :], c=t, s=30, cmap=cmap)
                # for one bin
                # ax_rate.set_title('Rate')
                ax_rate.set_xlabel('$R_{in}$')
                ax_rate.set_ylabel('$R_{out}$')
                ax_rate.set_aspect('equal')
                ax_rate.set_xlim(-0.15,1)
                ax_rate.set_ylim(-0.10,1)
                ax_rate.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'g-', linewidth=1)
                # ax_rate.tick_params(labelbottom= False)
                # ax_phase.set_title('Phase')
                ax_phase.set_xlabel('$R_{in}$')
                ax_phase.set_aspect('equal')
                ax_phase.set_xlim(-0.15,1)
                ax_phase.set_ylim(-0.10,1)
                ax_phase.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'g-', linewidth=1)
                # ax_phase.tick_params(labelbottom= False)
                # ax_complex.set_title('Polar')
                ax_complex.set_xlabel('$R_{in}$')
                ax_complex.set_aspect('equal')
                ax_complex.set_xlim(-0.15,1)
                ax_complex.set_ylim(-0.10,1)
                ax_complex.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'g-', linewidth=1)
                # ax_complex.tick_params(labelbottom= False)

                
                fig.subplots_adjust(right=0.8)
                cax = fig.add_axes([0.83,0.30,0.01,0.40])
                cbar = fig.colorbar(im3, cax=cax)
                cbar.set_label('distance (cm)', labelpad=20, rotation=270)
                save_dir = '/home/baris/figures/'
                fig.savefig(save_dir+'Rin_Rout_codes_'+fname+'.eps', dpi=200)
                fig.savefig(save_dir+'Rin_Rout_codes_'+fname+'.png', dpi=200)
            else:
                pass
            
    return rate_grid_corr, phase_grid_corr, complex_grid_corr, rate_gra_corr, phase_gra_corr, complex_gra_corr



path = paths[3]

# path = '/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/410-419/'

fname= 'noinh'

rate_grid_corr, phase_grid_corr, complex_grid_corr, rate_gra_corr, phase_gra_corr, complex_gra_corr = correlation(path, 1,fname, plot='yes')


'''BARPLOT delta R - tuned_2'''
paths = ['/home/baris/results/perceptron_th_n_codes/results_factor_5/same_poiss/',
'/home/baris/results/perceptron_th_n_codes/results_factor_5/noff/', 
'/home/baris/results/perceptron_th_n_codes/results_factor_5/nofb/',
'/home/baris/results/perceptron_th_n_codes/results_factor_5/noinh/']

all_deltaR = np.empty(0)
# for path in paths:
#     rate_grid_corr, phase_grid_corr, complex_grid_corr, rate_gra_corr, phase_gra_corr, complex_gra_corr = correlation(path, 10, 'asd', plot='no')
#     #change n_grid_seds to 10 upstream and regenerate the data
#     deltaR_rate = np.mean(rate_grid_corr-rate_gra_corr, axis=2)[:,0]
#     deltaR_phase = np.mean(phase_grid_corr-phase_gra_corr, axis=2)[:,0]
#     deltaR_complex = np.mean(complex_grid_corr-complex_gra_corr, axis=2)[:,0]
#     deltaR = np.concatenate((deltaR_rate, deltaR_phase, deltaR_complex))
#     all_deltaR = np.append(all_deltaR, deltaR)


for path in paths:
    rate, phase, compl = binned_mean(path, 10)
    deltaR = np.concatenate((rate, phase, compl))
    all_deltaR = np.append(all_deltaR, deltaR)

n_seeds = 10
n_tune = 4
n_codes = 3
code_order = ['rate', 'phase', 'polar']
tuned_order = ['full model', 'no FF', 'no FB', 'no inhibition']
code = n_tune*(['rate']*n_seeds+ ['phase']*n_seeds + ['polar']*n_seeds)
tune = ['full model']*n_seeds*n_codes + ['no FF']*n_seeds*n_codes + ['no FB']*n_seeds*n_codes + ['no inhibition']*n_seeds*n_codes
df = pd.DataFrame({'mean $\u0394R_{out}$': all_deltaR,
                   'code': pd.Categorical(code),
                  'tuning':pd.Categorical(tune)})

plt.close('all')
sns.set(context='paper',style='whitegrid',
        font='Arial',font_scale=2.5,color_codes=True,rc={'figure.figsize':(12,6)})
sns.set_palette(sns.color_palette(['#02818a', '#41ae76', '#67a9cf', '#99d8c9']))
# sns.color_palette('Pastel1', 4), #66c2a4, #016450
import pylab as plott
params = {'legend.fontsize':17, 'legend.handlelength':1.5}
plott.rcParams.update(params)
#palette='Spectral',
# fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,15))

# axis = ax2

ax1 = sns.barplot(x='code', y='mean $\u0394R_{out}$', hue='tuning', data=df, order=code_order, hue_order=tuned_order)#, ax=axis)
ax1.set(ylim=(-0.05,0.28))

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles[0:], labels=labels[0:])

# ax1 = sns.barplot(x='tuning', y='mean $\u0394R_{out}$', hue='code', data=df, order=['full model', 'no FF', 'no FB', 'no INH'], hue_order=hue_order)#, ax=axis)


# plt.title('Different Poisson Seeds')
# parameters = ('lr = 5*$10^{-4}$,  10 Grid seeds, 5 Poisson seeds,'+ 
#               '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
# plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

loc = ax1.patches[0].get_width()
for idx in range(n_seeds):
    rate_R = all_deltaR[idx]
    rate_R_noff = all_deltaR[30+idx]
    rate_R_nofb = all_deltaR[60+idx]
    rate_R_noinh = all_deltaR[90+idx]
    sns.lineplot(x=[-1.5*loc,-0.5*loc, 0.5*loc, 1.5*loc], y=[rate_R, rate_R_noff, rate_R_nofb, rate_R_noinh], 
                 color='k',style=True, dashes=[(2,2)], linewidth = 0.5, legend=False)
    
    phase_R = all_deltaR[10+idx]
    phase_R_noff = all_deltaR[40+idx]
    phase_R_nofb = all_deltaR[70+idx]
    phase_R_noinh = all_deltaR[100+idx]
    sns.lineplot(x=[1-1.5*loc,1-0.5*loc, 1+0.5*loc, 1+1.5*loc], y=[phase_R, phase_R_noff, phase_R_nofb, phase_R_noinh], 
                 color='k',style=True, dashes=[(2,2)], linewidth = 0.5, legend=False)#, ax=axis)
    
    
    
    complex_R = all_deltaR[20+idx]
    complex_R_noff = all_deltaR[50+idx]
    complex_R_nofb = all_deltaR[80+idx]
    complex_R_noinh = all_deltaR[110+idx]
    sns.lineplot(x=[2-1.5*loc,2-0.5*loc, 2+0.5*loc, 2+1.5*loc], y=[complex_R, complex_R_noff, complex_R_nofb, complex_R_noinh], 
                  color='k',style=True, dashes=[(2,2)], linewidth = 0.5, legend=False)#, ax=axis)
    #color='k', markersize= 3, marker='o',
    

save_dir = '/home/baris/figures/'
plt.savefig(save_dir+'binned_mean_deltaR_all_same_poiss.eps', dpi=200)
plt.savefig(save_dir+'binned_mean_deltaR_all_same_poiss.png', dpi=200)


writer = pd.ExcelWriter('deltaR_tuned.xlsx', engine='xlsxwriter')
df.to_excel(writer)
writer.save()

binned = binned_mean('/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/410-419/', 10)


'''BARPLOT delta R - diff poiss 2'''

paths = ['/home/baris/results/perceptron_th_n_codes/results_factor_5/same_poiss/',
'/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/410-419/']
all_deltaR = np.empty(0)
# for path in paths:
#     rate_grid_corr, phase_grid_corr, complex_grid_corr, rate_gra_corr, phase_gra_corr, complex_gra_corr = correlation(path, 10, 'asd')
#     #change n_grid_seds to 10 upstream and regenerate the data
#     deltaR_rate = np.mean(rate_grid_corr-rate_gra_corr, axis=2)[:,0]
#     deltaR_phase = np.mean(phase_grid_corr-phase_gra_corr, axis=2)[:,0]
#     deltaR_complex = np.mean(complex_grid_corr-complex_gra_corr, axis=2)[:,0]
#     deltaR = np.concatenate((deltaR_rate, deltaR_phase, deltaR_complex))
#     all_deltaR = np.append(all_deltaR, deltaR)



for path in paths:
    rate, phase, compl = binned_mean(path, 10)
    deltaR = np.concatenate((rate, phase, compl))
    all_deltaR = np.append(all_deltaR, deltaR)

n_seeds = 10
n_tune = 2
n_codes = 3
hue_order = ['rate', 'phase', 'polar']
code = n_tune*(['rate']*n_seeds+ ['phase']*n_seeds + ['polar']*n_seeds)
tune = ['same Poisson']*n_seeds*n_codes + ['different Poisson']*n_seeds*n_codes
df = pd.DataFrame({'mean $\u0394R_{out}$': all_deltaR,
                   'code': pd.Categorical(code),
                  'seeding':pd.Categorical(tune)})

plt.close('all')
sns.set(context='paper',style='whitegrid',
        font='Arial',font_scale=2.5,color_codes=True,rc={'figure.figsize':(12,6)})
sns.set_palette(sns.color_palette(['#02818a', '#fd8d3c']))
import pylab as plott
params = {'legend.fontsize':20, 'legend.handlelength':1.5}
plott.rcParams.update(params)

# fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,15))

# axis = ax2

ax2 = sns.barplot(x='code', y='mean $\u0394R_{out}$', hue='seeding', data=df, order=hue_order, hue_order=['same Poisson', 'different Poisson'])#, ax=axis)
ax2.set(ylim=(-0.04,0.28))
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles[0:], labels=labels[0:])
# plt.title('Different Poisson Seeds')
# parameters = ('lr = 5*$10^{-4}$,  10 Grid seeds, 5 Poisson seeds,'+ 
#               '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
# plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

loc = ax2.patches[0].get_width()/2
for idx in range(n_seeds):
    rate_R = all_deltaR[idx]
    rate_R_diff = all_deltaR[30+idx]
    sns.lineplot(x=[-loc,loc], y=[rate_R, rate_R_diff], color='k', style=True, dashes=[(2,2)], linewidth = 0.7, legend=False)#, ax=axis)
    
    phase_R = all_deltaR[10+idx]
    phase_R_diff = all_deltaR[40+idx]
    sns.lineplot(x=[1-loc,1+loc], y=[phase_R, phase_R_diff], color='k', style=True, dashes=[(2,2)], linewidth = 0.7, legend=False)#, ax=axis)
    
    complex_R = all_deltaR[20+idx]
    complex_R_diff = all_deltaR[50+idx]
    sns.lineplot(x=[2-loc,2+loc], y=[complex_R, complex_R_diff], color='k', style=True, dashes=[(2,2)], linewidth = 0.7, legend=False)#, ax=axis)
     
save_dir = '/home/baris/figures/'
plt.savefig(save_dir+'binned_mean_deltaR2_diff_same_poiss.eps', dpi=200)
plt.savefig(save_dir+'binned_mean_deltaR2_diff_same_poiss.png', dpi=200)

writer = pd.ExcelWriter('deltaR_diff.xlsx', engine='xlsxwriter')
df.to_excel(writer)
writer.save()


'''BARPLOT delta R - tuned'''
all_deltaR = np.empty(0)
# for path in paths:
#     rate_grid_corr, phase_grid_corr, complex_grid_corr, rate_gra_corr, phase_gra_corr, complex_gra_corr = correlation(path, 10, 'asd')
#     #change n_grid_seds to 10 upstream and regenerate the data
#     deltaR_rate = np.mean(rate_grid_corr-rate_gra_corr, axis=2)[:,0]
#     deltaR_phase = np.mean(phase_grid_corr-phase_gra_corr, axis=2)[:,0]
#     deltaR_complex = np.mean(complex_grid_corr-complex_gra_corr, axis=2)[:,0]
#     rate, phase, compl = binned_mean(path)
#     deltaR = np.concatenate((rate, phase, compl))
#     all_deltaR = np.append(all_deltaR, deltaR)

for path in paths:
    rate, phase, compl = binned_mean(path, 10)
    deltaR = np.concatenate((rate, phase, compl))
    all_deltaR = np.append(all_deltaR, deltaR)

n_seeds = 10
n_tune = 4
n_codes = 3
hue_order = ['rate', 'phase', 'polar']
code = n_tune*(['rate']*n_seeds+ ['phase']*n_seeds + ['polar']*n_seeds)
tune = ['full model']*n_seeds*n_codes + ['no FF']*n_seeds*n_codes + ['no FB']*n_seeds*n_codes + ['no INH']*n_seeds*n_codes
df = pd.DataFrame({'mean $\u0394R_{out}$': all_deltaR,
                   'code': pd.Categorical(code),
                  'tuning':pd.Categorical(tune)})

plt.close('all')
sns.set(context='paper',style='whitegrid',palette='deep',
        font='Arial',font_scale=3,color_codes=True,rc={'figure.figsize':(12,6)})
import pylab as plott
params = {'legend.fontsize':20, 'legend.handlelength':1.5}
plott.rcParams.update(params)

# fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,15))

# axis = ax2

ax1 = sns.barplot(x='tuning', y='mean $\u0394R_{out}$', hue='code', data=df, order=['full model', 'no FF', 'no FB', 'no INH'], hue_order=hue_order)#, ax=axis)
ax1.set(ylim=(-0.04,0.28))
# plt.title('Different Poisson Seeds')
# parameters = ('lr = 5*$10^{-4}$,  10 Grid seeds, 5 Poisson seeds,'+ 
#               '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
# plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

loc = ax1.patches[0].get_width()
for idx in range(n_seeds):
    rate_R = all_deltaR[idx]
    phase_R = all_deltaR[10+idx]
    complex_R = all_deltaR[20+idx]
    sns.lineplot(x=[-loc,0,loc], y=[rate_R, phase_R, complex_R], color='k', linewidth = 0.5)#, ax=axis)
    
    rate_R_noff = all_deltaR[30+idx]
    phase_R_noff = all_deltaR[40+idx]
    complex_R_noff = all_deltaR[50+idx]
    sns.lineplot(x=[1-loc,1,1+loc], y=[rate_R_noff, phase_R_noff, complex_R_noff], color='k', linewidth = 0.5)#, ax=axis)
    
    rate_R_nofb = all_deltaR[60+idx]
    phase_R_nofb = all_deltaR[70+idx]
    complex_R_nofb = all_deltaR[80+idx]
    sns.lineplot(x=[2-loc,2,2+loc], y=[rate_R_nofb, phase_R_nofb, complex_R_nofb], color='k', linewidth = 0.5)#, ax=axis)
    
    rate_R_noinh = all_deltaR[90+idx]
    phase_R_noinh = all_deltaR[100+idx]
    complex_R_noinh = all_deltaR[110+idx]
    sns.lineplot(x=[3-loc,3,3+loc], y=[rate_R_noinh, phase_R_noinh, complex_R_noinh], color='k', linewidth = 0.5)#, ax=axis)


save_dir = '/home/baris/figures/'
plt.savefig(save_dir+'mean_deltaR_all2_same_poiss.eps', dpi=200)
plt.savefig(save_dir+'mean_deltaR_all2_same_poiss.png', dpi=200)



'''BARPLOT delta R - diff poiss'''

paths = ['/home/baris/results/perceptron_th_n_codes/results_factor_5/same_poiss/',
'/home/baris/results/perceptron_th_n_codes/results_factor_5/diff_poiss/410-419/']
all_deltaR = np.empty(0)
for path in paths:
    rate_grid_corr, phase_grid_corr, complex_grid_corr, rate_gra_corr, phase_gra_corr, complex_gra_corr = correlation(path, 10, 'asd')
    #change n_grid_seds to 10 upstream and regenerate the data
    deltaR_rate = np.mean(rate_grid_corr-rate_gra_corr, axis=2)[:,0]
    deltaR_phase = np.mean(phase_grid_corr-phase_gra_corr, axis=2)[:,0]
    deltaR_complex = np.mean(complex_grid_corr-complex_gra_corr, axis=2)[:,0]
    deltaR = np.concatenate((deltaR_rate, deltaR_phase, deltaR_complex))
    all_deltaR = np.append(all_deltaR, deltaR)

n_seeds = 10
n_tune = 2
n_codes = 3
hue_order = ['rate', 'phase', 'polar']
code = n_tune*(['rate']*n_seeds+ ['phase']*n_seeds + ['polar']*n_seeds)
tune = ['same Poisson']*n_seeds*n_codes + ['different Poisson']*n_seeds*n_codes
df = pd.DataFrame({'mean $\u0394R_{out}$': all_deltaR,
                   'code': pd.Categorical(code),
                  'seeding':pd.Categorical(tune)})

plt.close('all')
sns.set(context='paper',style='whitegrid',palette='deep',
        font='Arial',font_scale=3,color_codes=True,rc={'figure.figsize':(12,6)})
import pylab as plott
params = {'legend.fontsize':20, 'legend.handlelength':1.5}
plott.rcParams.update(params)

# fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(10,15))

# axis = ax2

ax1 = sns.barplot(x='seeding', y='mean $\u0394R_{out}$', hue='code', data=df, order=['same Poisson', 'different Poisson'], hue_order=hue_order)#, ax=axis)
ax1.set(ylim=(-0.04,0.28))
# plt.title('Different Poisson Seeds')
# parameters = ('lr = 5*$10^{-4}$,  10 Grid seeds, 5 Poisson seeds,'+ 
#               '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
# plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

loc = ax1.patches[0].get_width()
for idx in range(n_seeds):
    rate_R = all_deltaR[idx]
    phase_R = all_deltaR[10+idx]
    complex_R = all_deltaR[20+idx]
    sns.lineplot(x=[-loc,0,loc], y=[rate_R, phase_R, complex_R], color='k', linewidth = 0.5)#, ax=axis)
    
    rate_R_noff = all_deltaR[30+idx]
    phase_R_noff = all_deltaR[40+idx]
    complex_R_noff = all_deltaR[50+idx]
    sns.lineplot(x=[1-loc,1,1+loc], y=[rate_R_noff, phase_R_noff, complex_R_noff], color='k', linewidth = 0.5)#, ax=axis)
    
save_dir = '/home/baris/figures/'
plt.savefig(save_dir+'mean_deltaR_diff_same_poiss.eps', dpi=200)
plt.savefig(save_dir+'mean_deltaR_diff_same_poiss.png', dpi=200)




                #MEAN#
                # mean_rate_grid = np.mean(rate_grid_corr, axis=1).flatten()           
                # mean_rate_gra = np.mean(rate_gra_corr, axis=1).flatten()   
                # mean_phase_grid = np.mean(phase_grid_corr, axis=1).flatten()        
                # mean_phase_gra = np.mean(phase_gra_corr, axis=1).flatten()    
                # mean_complex_grid = np.mean(complex_grid_corr, axis=1).flatten()        
                # mean_complex_gra = np.mean(complex_gra_corr, axis=1).flatten()    
                # ax_rate.scatter(mean_rate_grid, mean_rate_gra, marker='x', color='pink', s=10, label='mean')
                # ax_phase.scatter(mean_phase_grid, mean_phase_gra, marker='x', color='pink', s=10,label='mean')
                # ax_complex.scatter(mean_complex_grid, mean_complex_gra, marker='x', color='pink', s=10,label='mean')
                # ax_rate.legend()
                # ax_phase.legend()
                # ax_complex.legend()

    
    # grid_rate = np.concatenate((grid_rate[:,:,:,3800:4000], grid_rate[:,:,:,7800:8000]), axis=3)
    # grid_phase = np.concatenate((grid_phase[:,:,:,3800:4000], grid_phase[:,:,:,7800:8000]), axis=3)
    # grid_complex = np.concatenate((grid_complex[:,:,:,3800:4000], grid_complex[:,:,:,7800:8000]), axis=3)
    # gra_rate = np.concatenate((gra_rate[:,:,:,38000:40000], gra_rate[:,:,:,78000:80000]), axis=3)
    # gra_phase = np.concatenate((gra_phase[:,:,:,38000:40000], gra_phase[:,:,:,78000:80000]), axis=3)
    # gra_complex = np.concatenate((gra_complex[:,:,:,38000:40000], gra_complex[:,:,:,78000:80000]), axis=3)
    
    