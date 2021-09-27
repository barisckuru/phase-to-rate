# -*- coding: utf-8 -*-
"""
Figure 1 demonstrates the grid cell model with phase precession and the
shuffling feature.
"""

import matplotlib.pyplot as plt
import numpy as np
import grid_model
import matplotlib.gridspec as gridspec

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

spacing = 50
orientation = 30
pos_peak = [20, 20]
arr_size = 200
dur_ms = 2000
dt_s = 0.002

trajectories = [75.0]
simulation_result = grid_model.grid_simulate(
    trajectories,
    dur_ms=dur_ms,
    grid_seed=1,
    poiss_seeds=[1],
    shuffle='non-shuffled',
    n_grid=200,
    speed_cm=20,
    rate_scale=5,
    arr_size=arr_size,
    f=10,
    shift_deg=180,
<<<<<<< HEAD
    dt_s=0.001,
=======
    dt_s=dt_s,
>>>>>>> 5887f8a0c0c1aebf6c85bad742693598fed6cba3
    large_output=True
)

grid_length_cm = 100
grid_axes_ticks = np.arange(0,grid_length_cm,grid_length_cm/arr_size)

trajectories_y = 100 - np.array(trajectories)
trajectories_xstart = 0
trajectories_xend= 40

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6,2)
ax1 = fig.add_subplot(gs[:3,0])
ax1.imshow(simulation_result[3][:,:,1], extent=[0,100,0,100]) # TODO UNITS
for trajectory in trajectories_y:
    ax1.plot([trajectories_xstart, trajectories_xend], [trajectory]*2,
             linewidth=3)
ax2 = fig.add_subplot(gs[0:2,1])
x = np.arange(0,dur_ms,dt_s*1000)
ax2.plot(x,simulation_result[4][1,:,0])
ax2.plot(x,simulation_result[4][1,:,1])
ax2.plot(x,simulation_result[4][1,:,2])
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Frequency (Hz)")
ax2.legend(("Traj1", "Traj2", "Traj3"))
ax3 = fig.add_subplot(gs[2:3,1])
ax3.eventplot(simulation_result[0][trajectories[0]][1][1], lineoffsets=3, color=colors[0])
ax3.eventplot(simulation_result[0][trajectories[1]][1][1], lineoffsets=2, color=colors[1])
ax3.eventplot(simulation_result[0][trajectories[2]][1][1], lineoffsets=1, color=colors[2])
ax3.set_ylabel("Trajectory")
ax3.set_xlabel("Time (ms)")
ax3.get_yaxis().set_visible(False)
xlim=[-10, 2010]
ax2.set_xlim((xlim))
ax3.set_xlim((xlim))
