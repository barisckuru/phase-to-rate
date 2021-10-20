import matplotlib.pyplot as plt
import numpy as np
import grid_model
import matplotlib.gridspec as gridspec
import shelve
import os
from neural_coding import _code_maker, load_spikes

dir_name = os.path.dirname(__file__)  # Name of script dir
results_dir = os.path.join(dir_name, 'results', 'example_simulations')
fname_non_shuffled = 'grid-seed_duration_shuffling_tuning_10_2000_non-shuffled_tuned'
fname_shuffled = 'grid-seed_duration_shuffling_tuning_10_2000_shuffled_tuned'

non_shuffled = shelve.open(os.path.join(results_dir, fname_non_shuffled))
shuffled = shelve.open(os.path.join(results_dir, fname_shuffled))

plt.eventplot(non_shuffled['75']['grid_spikes'][100])

plt.eventplot(non_shuffled['75']['grid_spikes'][100])
plt.eventplot(non_shuffled['75']['granule_spikes'][100], linelengths=10, linewidth=3)

theta_lines = np.arange(0,2001,100)
vline_alpha = 0.3
linewidth=3
xlim = (-10, 2010)
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0,0])
ax1.eventplot(non_shuffled['75']['grid_spikes'][100], linelengths=1, linewidth=linewidth)
ax1.set_xlim(xlim)
ax1.set_ylabel("Grid Cell #")
ax1.vlines(x=theta_lines, ymin=-5, ymax=205, alpha=vline_alpha)
ax2 = fig.add_subplot(gs[1,0])
ax2.eventplot(non_shuffled['75']['granule_spikes'][100], linelengths=10, linewidth=linewidth)
ax2.set_xlim(xlim)
ax2.set_ylabel("Granule Cell #")
ax2.set_xlabel("Time (ms)")
ax2.vlines(x=theta_lines, ymin=-50, ymax=2050, alpha=vline_alpha)

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]
trajectories = [float(x) for x in list(non_shuffled.keys())]
n_samples = 20

# rate_n_phase(grid_spikes, trajectories, n_samples)