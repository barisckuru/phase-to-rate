import matplotlib.pyplot as plt
import numpy as np
import grid_model
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import shelve
import os
from neural_coding import _code_maker, load_spikes, rate_n_phase

dir_name = os.path.dirname(__file__)  # Name of script dir
results_dir = os.path.join(dir_name, 'results', 'example_simulations')
fname_non_shuffled = 'grid-seed_duration_shuffling_tuning_10_2000_non-shuffled_tuned'
fname_shuffled = 'grid-seed_duration_shuffling_tuning_10_2000_shuffled_tuned'

path_non_shuffled = os.path.join(results_dir, fname_non_shuffled)
path_shuffled = os.path.join(results_dir, fname_shuffled)

non_shuffled = shelve.open(path_non_shuffled)
shuffled = shelve.open(path_shuffled)

theta_lines = np.arange(0,2001,100)
vline_alpha = 0.3
linewidth=3
xlim = (-10, 2010)

n_grid = 20
n_granule = 20

fig = plt.figure(constrained_layout=True, figsize=(8.25,11+17/24))
gs = fig.add_gridspec(12,12)
ax1 = fig.add_subplot(gs[0:4,0:6])
ax1.eventplot(non_shuffled['75']['grid_spikes'][100][0:n_grid], linelengths=1, linewidth=linewidth, lineoffsets=range(1,n_grid+1))
ax1.set_xlim(xlim)
ax1.set_ylabel("Grid Cell #")
ax1.vlines(x=theta_lines, ymin=-5, ymax=205, alpha=vline_alpha)
ax1.set_ylim((0,n_grid+1))
ax1.set_yticks(range(0,n_grid+1, 5))
ax1.set_title("Spike times")
ax2 = fig.add_subplot(gs[4:8,0:6])
ax2.eventplot(non_shuffled['75']['granule_spikes'][100][0:n_granule], linelengths=1, linewidth=linewidth, lineoffsets=range(1,n_granule+1))
ax2.set_xlim(xlim)
ax2.set_ylabel("Granule Cell #")
ax2.set_xlabel("Time (ms)")
ax2.vlines(x=theta_lines, ymin=-50, ymax=2050, alpha=vline_alpha)
ax2.set_ylim((0,n_granule+1))
ax1.set_yticks(range(0,n_granule+1, 5))

trajectories = [x for x in list(non_shuffled.keys())]
n_samples = 20

grid_spikes_non_shuffled = load_spikes(path_non_shuffled, "grid", trajectories, n_samples)
granule_spikes_non_shuffled = load_spikes(path_non_shuffled, "granule", trajectories, n_samples)

_, _, grid_rate_code, grid_phase_code, grid_polar_code = rate_n_phase(grid_spikes_non_shuffled, trajectories, n_samples)

_, _, granule_rate_code, granule_phase_code, granule_polar_code = rate_n_phase(granule_spikes_non_shuffled, trajectories, n_samples)

grid_rate_code_x = grid_rate_code[:4000,0,0].reshape((200,20))
grid_rate_code_y = grid_rate_code[4000:,0,0].reshape((200,20))

grid_phase_code_x = grid_phase_code[:4000,0,0].reshape((200,20))
grid_phase_code_y = grid_phase_code[4000:,0,0].reshape((200,20))

grid_polar_code_x = grid_polar_code[:4000,0,0].reshape((200,20))
grid_polar_code_y = grid_polar_code[4000:,0,0].reshape((200,20))

colors = "Greys"
ax3 = fig.add_subplot(gs[0:4,6])
im3 = ax3.imshow(grid_rate_code_x[:n_grid,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_grid+0.5))
ax3.set_title("Rate Codex")
plt.colorbar(im3, ax=ax3, orientation="horizontal")
ax4 = fig.add_subplot(gs[0:4,7])
im4 = ax4.imshow(grid_rate_code_y[:n_grid,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_grid+0.5))
ax4.set_title("Rate Code y")
plt.colorbar(im4, ax=ax4, orientation="horizontal")
ax5 = fig.add_subplot(gs[0:4,8])
im5 = ax5.imshow(grid_phase_code_x[:n_grid,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_grid+0.5))
ax5.set_title("Phase Code x")
plt.colorbar(im5, ax=ax5, orientation="horizontal")
ax6 = fig.add_subplot(gs[0:4,9])
im6 = ax6.imshow(grid_phase_code_y[:n_grid,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_grid+0.5))
ax6.set_title("Phase Code y")
plt.colorbar(im6, ax=ax6, orientation="horizontal")
ax7 = fig.add_subplot(gs[0:4,10])
im7 = ax7.imshow(grid_polar_code_x[:n_grid,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_grid+0.5))
ax7.set_title("Polar Code x")
plt.colorbar(im7, ax=ax7, orientation="horizontal")
ax8 = fig.add_subplot(gs[0:4,11])
im8 = ax8.imshow(grid_polar_code_y[:n_grid,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_grid+0.5))
ax8.set_title("Polar Code y")
plt.colorbar(im8, ax=ax8, orientation="horizontal")

for ax in [ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.set_yticks(range(0,n_grid+1, 5))
    ax.set_ylim(ax1.get_ylim())
    
for ax in [ax1, ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.axes.get_xaxis().set_visible(False)



granule_rate_code_x = granule_rate_code[:40000,0,0].reshape((2000,20))
granule_rate_code_y = granule_rate_code[40000:,0,0].reshape((2000,20))

granule_phase_code_x = granule_phase_code[:40000,0,0].reshape((2000,20))
granule_phase_code_y = granule_phase_code[40000:,0,0].reshape((2000,20))

granule_polar_code_x = granule_polar_code[:40000,0,0].reshape((2000,20))
granule_polar_code_y = granule_polar_code[40000:,0,0].reshape((2000,20))

ax9 = fig.add_subplot(gs[4:8,6])
im9 = ax9.imshow(granule_rate_code_x[:n_granule,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_granule+0.5))
plt.colorbar(im9, ax=ax9, orientation="horizontal")
ax10 = fig.add_subplot(gs[4:8,7])
im10 = ax10.imshow(granule_rate_code_y[:n_granule,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_granule+0.5))

plt.colorbar(im10, ax=ax10, orientation="horizontal")
ax11 = fig.add_subplot(gs[4:8,8])
im11 = ax11.imshow(granule_phase_code_x[:n_granule,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_granule+0.5))

plt.colorbar(im11, ax=ax11, orientation="horizontal")
ax12 = fig.add_subplot(gs[4:8,9])
im12 = ax12.imshow(granule_phase_code_y[:n_granule,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_granule+0.5))

plt.colorbar(im12, ax=ax12, orientation="horizontal")
ax13 = fig.add_subplot(gs[4:8,10])
im13 = ax13.imshow(granule_polar_code_x[:n_granule,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_granule+0.5))

plt.colorbar(im13, ax=ax13, orientation="horizontal")
ax14 = fig.add_subplot(gs[4:8,11])
im14 = ax14.imshow(granule_polar_code_y[:n_granule,:], aspect="auto", cmap=colors, origin='lower', interpolation="none", extent=(0,20,0.5,n_granule+0.5))

plt.colorbar(im14, ax=ax14, orientation="horizontal")

for ax in [ax9, ax10, ax11, ax12, ax13, ax14]:
    ax.set_yticks(range(0,n_grid+1, 5))
    ax.set_ylim(ax1.get_ylim())


for axis in [ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14]:
    # axis.axis("off")
    # axis.axes.get_xaxis().set_visible(False)
    axis.axes.get_yaxis().set_visible(False)
    pass


# ax15 is a placeholder for a perceptron schematic
ax15 = fig.add_subplot(gs[8:12,:4])







"""
grid_rate_code_x = grid_rate_code[:4000,:,:]
grid_rate_code_y = grid_rate_code[4000:,:,:]

grid_phase_code_x = grid_phase_code[:4000,:,:]
grid_phase_code_y = grid_phase_code[4000:,:,:]

ax3 = fig.add_subplot(gs[0,1])
ax3.plot(grid_rate_code_x[:,0,0])
ax3.set_ylabel("Rate Code x")
ax3.set_title("Grid Cell Codes")
ax4 = fig.add_subplot(gs[1,1])
ax4.plot(grid_rate_code_y[:,0,0])
ax4.set_ylabel("Rate Code y")
ax5 = fig.add_subplot(gs[2,1])
ax5.plot(grid_phase_code_x[:,0,0])
ax5.set_ylabel("Phase Code x")
ax6 = fig.add_subplot(gs[3,1])
ax6.plot(grid_phase_code_y[:,0,0])
ax6.set_ylabel("Phase Code y")
"""