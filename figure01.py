# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:47:34 2021

@author: Daniel
"""

import matplotlib.pyplot as plt
import numpy as np
import grid_model
import matplotlib.gridspec as gridspec

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

spacing = 50
orientation = 30
pos_peak = [20, 20]

trajectories = [75.0, 70.0, 50.0]
simulation_result = grid_model.grid_simulate(
    trajectories,
    dur_ms=2000,
    grid_seed=1,
    poiss_seeds=[1],
    shuffle='non-shuffled',
    n_grid=200,
    speed_cm=20,
    rate_scale=5,
    arr_size=200,
    f=10,
    shift_deg=180,
    dt_s=0.002,
    large_output=True
)
"""
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6,2)
ax1 = fig.add_subplot(gs[:3,0])
ax1.imshow(simulation_result[3][:,:,1])
ax2 = fig.add_subplot(gs[0,1])
ax2.plot(simulation_result[4][1,:,0])
ax3 = fig.add_subplot(gs[1,1])
ax3.plot(simulation_result[4][1,:,1])
ax4 = fig.add_subplot(gs[2,1])
ax4.plot(simulation_result[4][1,:,2])
"""

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6,2)
ax1 = fig.add_subplot(gs[:3,0])
ax1.imshow(simulation_result[3][:,:,1]) # TODO UNITS
ax2 = fig.add_subplot(gs[0:2,1])
ax2.plot(simulation_result[4][1,:,0])
ax2.plot(simulation_result[4][1,:,1])
ax2.plot(simulation_result[4][1,:,2])
ax2.set_xlabel("???")
ax2.set_ylabel("Frequency (Hz)")
ax2.legend(("Traj1", "Traj2", "Traj3"))
ax3 = fig.add_subplot(gs[2:3,1])
ax3.eventplot(simulation_result[0][trajectories[0]][1][1], lineoffsets=3, color=colors[0])
ax3.eventplot(simulation_result[0][trajectories[1]][1][1], lineoffsets=2, color=colors[1])
ax3.eventplot(simulation_result[0][trajectories[2]][1][1], lineoffsets=1, color=colors[2])
