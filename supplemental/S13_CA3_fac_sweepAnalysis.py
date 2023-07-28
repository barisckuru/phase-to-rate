# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:50:46 2022

@author: olive
"""
import pandas as pd
import os 
import shelve
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from matplotlib import colors

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


with open('sweep_condition_dict1.pkl', 'rb') as f:
    condition_dict = pickle.load(f)

tauSTDP_list = list(condition_dict.keys())
CA3_inh_list = list(condition_dict[tauSTDP_list[0]].keys())
gseeds = len(condition_dict[tauSTDP_list[0]][CA3_inh_list[0]]['full'])
first_gseed =list(condition_dict[tauSTDP_list[0]][CA3_inh_list[0]]['full'].keys())[0]
simulation_time = 2


''' Analysis across tauSTDP '''
'compute ratios of noFB to full across parameters'
weight_ratio = np.zeros([len(CA3_inh_list),len(tauSTDP_list)])
rates2weights_ratio = np.zeros([len(CA3_inh_list),len(tauSTDP_list)])
weights_full = np.zeros([len(CA3_inh_list),len(tauSTDP_list)])
weights_noFB = np.zeros([len(CA3_inh_list),len(tauSTDP_list)])

CA3_spikes_noFB = np.zeros([len(CA3_inh_list),len(tauSTDP_list)])
CA3_spikes_full = np.zeros([len(CA3_inh_list),len(tauSTDP_list)])

for tau,tauSTDP in enumerate(tauSTDP_list):
    for inh,inhibition in enumerate(CA3_inh_list):

        full_w_mean = np.zeros(gseeds)
        noFB_w_mean = np.zeros(gseeds)
        full_s_mean = np.zeros(gseeds)
        noFB_s_mean = np.zeros(gseeds)
        full_gc_s_mean = np.zeros(gseeds)
        noFB_gc_s_mean = np.zeros(gseeds)
        full_i_s_mean = np.zeros(gseeds)
        noFB_i_s_mean = np.zeros(gseeds)
        
        full_w_corr = np.zeros(gseeds)
        noFB_w_corr = np.zeros(gseeds)
        full_s_corr = np.zeros(gseeds)
        noFB_s_corr = np.zeros(gseeds)
        
        for gseed in range(first_gseed,first_gseed+gseeds):
            #print(gseed)
            i = gseed-first_gseed 
            noFB_w_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['noFB'][gseed]['weights'])
            noFB_s_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['noFB'][gseed]['CA3_spikes'])/simulation_time
            noFB_gc_s_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['noFB'][gseed]['GC_spikes'])/simulation_time
            noFB_i_s_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['noFB'][gseed]['I_spikes'])/simulation_time
            
            weights = pd.DataFrame(condition_dict[tauSTDP][inhibition]['noFB'][gseed]['weights'])
            weights = weights.transpose()
            wncorr_matrix = weights.corr()
            wncorr_matrix[wncorr_matrix==1]=np.NaN
            noFB_w_corr[i] = np.nanmean(wncorr_matrix)
            
            meanGCrate = np.mean(condition_dict[tauSTDP][inhibition]['noFB'][gseed]['GC_spikes'])/simulation_time
            if meanGCrate < 0.2 or meanGCrate > 0.3:
                noFB_w_mean[i] = np.nan
                #print('nan Ahoi')
                noFB_s_mean[i] = np.nan
                noFB_gc_s_mean[i] = np.nan
                noFB_w_corr[i] = np.nan
            
            #print(gseed)
            full_w_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['full'][gseed]['weights'])    
            full_s_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['full'][gseed]['CA3_spikes'])/simulation_time    
            full_gc_s_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['full'][gseed]['GC_spikes'])/simulation_time
            full_i_s_mean[i] = np.mean(condition_dict[tauSTDP][inhibition]['full'][gseed]['I_spikes'])/simulation_time
            
            weights = pd.DataFrame(condition_dict[tauSTDP][inhibition]['full'][gseed]['weights'])
            weights = weights.transpose()
            wfcorr_matrix = weights.corr()
            wfcorr_matrix[wfcorr_matrix==1]=np.NaN
            full_w_corr[i] = np.nanmean(wfcorr_matrix)
            
            meanGCrate = np.mean(condition_dict[tauSTDP][inhibition]['full'][gseed]['GC_spikes'])/simulation_time
            if meanGCrate < 0.2 or meanGCrate > 0.3:
                full_w_mean[i] = np.nan
                full_s_mean[i] = np.nan
                full_gc_s_mean[i] = np.nan
                full_w_corr[i] = np.nan
                full_s_corr[i] = np.nan
        
        weight_ratio[inh,tau] = np.nanmean(full_w_mean)/np.nanmean(noFB_w_mean)
        rates2weights_ratio[inh,tau] = np.nanmean(full_w_mean/full_s_mean)/np.nanmean(noFB_w_mean/noFB_s_mean)
        weights_full[inh,tau] = np.nanmean(full_w_mean)
        weights_noFB[inh,tau] = np.nanmean(noFB_w_mean)

        CA3_spikes_noFB[inh,tau] = np.nanmean(noFB_s_mean)
        CA3_spikes_full[inh,tau] = np.nanmean(full_s_mean)
'''
# generate meshgrid for plot
yax = tauSTDP_list
xax = CA3_inh_list
yax, xax = np.meshgrid(yax, xax)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(yax, xax, weight_ratio, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
ax.plot_wireframe(yax, xax, weight_ratio, color='k', lw=0.05, alpha=0.3)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('inh. (mV)')
ax.set_xlabel('tau STDP (ms)')
ax.set_zlabel('weight ratio')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(yax, xax, rates2weights_ratio, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
ax.plot_wireframe(yax, xax, rates2weights_ratio, color='k', lw=0.05, alpha=0.3)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('inh. (mV)')
ax.set_xlabel('tau STDP (ms)')
ax.set_zlabel('spikes to weights ratio')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(yax, xax, weights_full, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
ax.plot_wireframe(yax, xax, weights_full, color='k', lw=0.05, alpha=0.3)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('inh. (mV)')
ax.set_xlabel('tau STDP (ms)')
ax.set_zlabel('weights full')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(yax, xax, CA3_spikes_full, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
ax.plot_wireframe(yax, xax, CA3_spikes_full, color='k', lw=0.05, alpha=0.3)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('inh. (mV)')
ax.set_xlabel('tau STDP (ms)')
ax.set_zlabel('CA3 spikes full')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(yax, xax, CA3_spikes_noFB, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
ax.plot_wireframe(yax, xax, CA3_spikes_noFB, color='k', lw=0.05, alpha=0.3)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_ylabel('inh. (mV)')
ax.set_xlabel('tau STDP (ms)')
ax.set_zlabel('CA3 spikes noFB')
'''

f1, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize=(9, 12 ))
f1.subplots_adjust(hspace=.6, wspace=.6, left=0.1, right=0.9)

im1 = ax1.imshow(weights_full, norm=colors.LogNorm())
cbar1 = plt.colorbar(im1,ax=ax1)
cbar1.set_label('weights full')
ax1.set_xticklabels(tauSTDP_list)
ax1.set_xlabel('STDP tau')
ax1.set_yticklabels(CA3_inh_list)
ax1.set_ylabel('inhibition')

im2 = ax2.imshow(weights_noFB, norm=colors.LogNorm(vmin=np.min(weights_full),vmax=np.max(weights_full)))
cbar2 = plt.colorbar(im2,ax=ax2)
cbar2.set_label('weights noFB')
ax2.set_xticklabels(tauSTDP_list)
ax2.set_xlabel('STDP tau')
ax2.set_yticklabels(CA3_inh_list)
ax2.set_ylabel('inhibition')

im3 = ax3.imshow(rates2weights_ratio, norm=colors.LogNorm(vmin=.9, vmax=2.5))
cbar3 = plt.colorbar(im3,ax=ax3)
cbar3.set_label('rates2weights ratio')
ax3.set_xticklabels(tauSTDP_list)
ax3.set_xlabel('STDP tau')
ax3.set_yticklabels(CA3_inh_list)
ax3.set_ylabel('inhibition')

im4 = ax4.imshow(weight_ratio, norm=colors.LogNorm(vmin=.9))
cbar4 = plt.colorbar(im4,ax=ax4)
cbar4.set_label('weight ratio')
ax4.set_xticklabels(tauSTDP_list)
ax4.set_xlabel('STDP tau')
ax4.set_yticklabels(CA3_inh_list)
ax4.set_ylabel('inhibition')

im5 = ax5.imshow(CA3_spikes_full, norm=colors.LogNorm())
cbar5 = plt.colorbar(im5,ax=ax5)
cbar5.set_label('CA3 spikes full')
ax5.set_xticklabels(tauSTDP_list)
ax5.set_xlabel('STDP tau')
ax5.set_yticklabels(CA3_inh_list)
ax5.set_ylabel('inhibition')

im6 = ax6.imshow(CA3_spikes_noFB, norm=colors.LogNorm(vmin=np.min(CA3_spikes_full),vmax=np.max(CA3_spikes_full)))
cbar6 = plt.colorbar(im6,ax=ax6)
cbar6.set_label('CA3 spikes noFB')
ax6.set_xticklabels(tauSTDP_list)
ax6.set_xlabel('STDP tau')
ax6.set_yticklabels(CA3_inh_list)
ax6.set_ylabel('inhibition')

f1.savefig("sweeps_extended1.pdf")