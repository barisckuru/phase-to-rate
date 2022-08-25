#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:27:14 2022

@author: baris
"""

from phase_to_rate import grid_model
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation as stg
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap as lcmap
import numpy as np
from matplotlib.patches import PathPatch
import os
import pickle
import pandas as pd

# make colormaps with hex codes
def _make_cmap(color_list, N=100):
    colors = []
    for color in color_list:
        colors.append('#'+color)
        my_cmap = lcmap.from_list('my_cmap', colors, N=100)
    return my_cmap


# function to produce samples with phase precession from overall firing
def _precession_spikes(overall, dur_s=5, n_sim=1000, T=0.1,
                       dt_s=0.002, bins_size_deg=7.2, shuffle=False,
                       poisson_seed_start=100):
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


def _adjust_box_widths(g, fac):
    """ Adjust the widths of a seaborn-generated boxplot. """
    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def _adjust_bar_widths(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value
        # change the bar width
        patch.set_width(new_value)
        # recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
                
        
# figure 5 data
def f5_load_data (fname, path):
    with open(os.path.join(path, fname), 'rb') as f:
        condition_dict = pickle.load(f)
    
    gseeds = 30
    simulation_time = 2    # simulation time in s
    nan = np.nan
    
    ''' Analysis across grid seeds '''
           
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
    
    full_inter_corr = np.zeros(gseeds)
    noFB_inter_corr = np.zeros(gseeds)
    
    for gseed in range(1,gseeds+1):
        #print(gseed)
        i = gseed-1  
        noFB_w_mean[i] = np.mean(condition_dict['noFB'][gseed]['weights'])
        noFB_s_mean[i] = np.mean(condition_dict['noFB'][gseed]['CA3_spikes'])/simulation_time
        noFB_gc_s_mean[i] = np.mean(condition_dict['noFB'][gseed]['GC_spikes'])/simulation_time
        noFB_i_s_mean[i] = np.mean(condition_dict['noFB'][gseed]['I_spikes'])/simulation_time
        
        weights = pd.DataFrame(condition_dict['noFB'][gseed]['weights'])
        weights = weights.transpose()
        wncorr_matrix = weights.corr()
        wncorr_matrix[wncorr_matrix==1]=np.NaN
        noFB_w_corr[i] = np.nanmean(wncorr_matrix)
        
        spikes = pd.DataFrame(condition_dict['noFB'][gseed]['CA3_spikes'])
        spikes = spikes.transpose()
        sncorr_matrix = spikes.corr()
        sncorr_matrix[sncorr_matrix==1]=np.NaN
        noFB_s_corr[i] = np.nanmean(sncorr_matrix)
        
        wncorr_matrix.fillna(np.nanmean(wncorr_matrix), inplace=True)
        sncorr_matrix.fillna(np.nanmean(sncorr_matrix), inplace=True) 
        wncorr_matrix = wncorr_matrix.stack()
        sncorr_matrix = sncorr_matrix.stack()
        noFB_inter_corr[i] = wncorr_matrix.corr(sncorr_matrix)
        
        meanGCrate = np.mean(condition_dict['noFB'][gseed]['GC_spikes'])/simulation_time
        #print(gseed)
        full_w_mean[i] = np.mean(condition_dict['full'][gseed]['weights'])    
        full_s_mean[i] = np.mean(condition_dict['full'][gseed]['CA3_spikes'])/simulation_time  
        full_i_s_mean[i] = np.mean(condition_dict['full'][gseed]['I_spikes'])/simulation_time
        full_gc_s_mean[i] = np.mean(condition_dict['full'][gseed]['GC_spikes'])/simulation_time
        
        weights = pd.DataFrame(condition_dict['full'][gseed]['weights'])
        weights = weights.transpose()
        wfcorr_matrix = weights.corr()
        wfcorr_matrix[wfcorr_matrix==1]=np.NaN
        full_w_corr[i] = np.nanmean(wfcorr_matrix)
        
        spikes = pd.DataFrame(condition_dict['full'][gseed]['CA3_spikes'])
        spikes = spikes.transpose()
        sfcorr_matrix = spikes.corr()
        sfcorr_matrix[sfcorr_matrix==1]=np.NaN
        full_s_corr[i] = np.nanmean(sfcorr_matrix)
        
        wfcorr_matrix.fillna(np.nanmean(wfcorr_matrix), inplace=True)    
        sfcorr_matrix.fillna(np.nanmean(sfcorr_matrix), inplace=True)
        wfcorr_matrix = wfcorr_matrix.stack()
        sfcorr_matrix = sfcorr_matrix.stack()
        full_inter_corr[i] = wfcorr_matrix.corr(sfcorr_matrix)
        
    'filtering the mean rates between 0.2 - 0.3'
    for gseed in range(1,gseeds+1):
        i = gseed-1
        meanGCrate = np.mean(condition_dict['full'][gseed]['GC_spikes'])/simulation_time
        if meanGCrate < 0.2 or meanGCrate > 0.3:
            full_w_mean[i] = nan
            full_s_mean[i] = nan
            full_i_s_mean[i] = nan
            full_gc_s_mean[i] = nan
            full_w_corr[i] = nan
            full_s_corr[i] = nan
            full_inter_corr[i] = nan
        meanGCrate = np.mean(condition_dict['noFB'][gseed]['GC_spikes'])/simulation_time
        if meanGCrate < 0.2 or meanGCrate > 0.3:
            noFB_w_mean[i] = nan
            print('nan Ahoi')
            noFB_s_mean[i] = nan
            noFB_gc_s_mean[i] = nan
            noFB_w_corr[i] = nan
            noFB_s_corr[i] = nan
            noFB_inter_corr[i] = nan  
        
    # mean weights dataframe
    weight_means = np.concatenate((full_w_mean, noFB_w_mean))
    grid_seeds = 2*list(range(1,31,1))
    tuning = 30*['full'] + 30*['noFB']
    weights_df = pd.DataFrame([grid_seeds, weight_means, tuning]).transpose()
    weights_df.columns=['grid seed', 'mean weight', 'tuning']
    
    # mean rates dataframe
    grid_seeds = np.array(6*list(range(1,31,1)))
    cell = 2*(30*['granule'] + 30*['ca3'] + 30*['interneuron'])
    tuning = 90*['full'] + 90*['noFB']
    mean_rates = np.concatenate((full_gc_s_mean, full_s_mean, full_i_s_mean,
                                 noFB_gc_s_mean, noFB_s_mean, noFB_i_s_mean))
    rates_df = pd.DataFrame([grid_seeds, mean_rates, cell, tuning]).transpose()
    rates_df.columns=['grid seed', 'mean rate', 'cell', 'tuning']

    return rates_df, weights_df