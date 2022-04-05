#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:27:14 2022

@author: baris
"""

import grid_model
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation as stg
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap as lcmap
import numpy as np
from matplotlib.patches import PathPatch

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