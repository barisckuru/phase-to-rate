"""
Simulate pydentate with grid cell inputs.

This is one of the most computationally intensive steps as it generates
all the raw data that is used for the perceptron, tempotron as well as
CA3 analysis. It allows for reproducibility through a variety of seeding
steps.
"""

from pydentate import neuron_tools
from grid_model import grid_simulate
import shelve
import copy
import time
import numpy as np
from pydentate_integrate import granule_simulate
import os

"""Setup"""
neuron_tools.load_compiled_mechanisms()


"""Parameters"""
grid_seeds = np.arange(1, 31, 1)

trajectories, p = [75], 100

# trajectories, p = [75], 100  # In cm
# trajectories, p = [74.5], 200
# trajectories, p = [74], 300
# trajectories, p = [73.5], 400
# trajectories, p = [73], 500
# trajectories, p = [72.5], 600
# trajectories, p = [72], 700
# trajectories, p = [71], 800
# trajectories, p = [70], 900
# trajectories, p = [69], 1000
# trajectories, p = [68], 1100
# trajectories, p = [67], 1200
# trajectories, p = [66], 1300
# trajectories, p = [65], 1400
# trajectories, p = [60], 1500
# trajectories, p = [30], 1600
# trajectories, p = [15], 1700

poisson_seeds = np.arange(p, p + 20, 1)
poisson_seeds = list(poisson_seeds)


# shuffling = "non-shuffled"
shuffling = "shuffled"
# poisson_reseeding = True  # Controls seeding between trajectories
speed_cm = 20
dur_ms = 2000
rate_scale = 5
n_grid = 200
pp_weight = 9e-4
network_type = "full"
# network_type = "no-feedback"
# network_type = "disinhibited"
# network_type = "no-feedforward"

print(network_type)

parameters = {}
parameters = {
    "dur_ms": dur_ms,
    "pp_weight": pp_weight,
    "speed": speed_cm,
    "n_grid": n_grid,
    "rate_scale": rate_scale,
    "poisson_seeds": poisson_seeds,
}

print("grid", grid_seeds, "poiss", poisson_seeds,
      "traj", trajectories, "dur", dur_ms)

for grid_seed in grid_seeds:
    save_dir = (f'/home/baris/results/{network_type}/seperate/seed_'
                + str(grid_seed) + '/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    output_path = (
        f"{grid_seed}_{trajectories}_{poisson_seeds[0]}-{poisson_seeds[-1]}_{dur_ms}"
    )

    storage = shelve.open(
        save_dir
        + """grid-seed_trajectory_poisson-seeds_
                          duration_shuffling_tuning_"""
        + output_path
        + "_"
        + shuffling
        + "_"
        + network_type,
        writeback=True,
    )

    grid_spikes, _ = grid_simulate(
        trajs=trajectories,
        dur_ms=dur_ms,
        grid_seed=grid_seed,
        poiss_seeds=poisson_seeds,
        shuffle=shuffling,
        n_grid=n_grid,
        speed_cm=speed_cm,
        rate_scale=rate_scale,
    )

    granule_spikes = {}
    for traj in trajectories:
        granule_spikes[traj] = {}
        for poisson_seed in poisson_seeds:
            granule_spikes[traj][poisson_seed] = {}
            granule_spikes_poiss = granule_simulate(
                grid_spikes[traj][poisson_seed],
                dur_ms=dur_ms,
                network_type=network_type,
                grid_seed=grid_seed,
                pp_weight=pp_weight,
            )
            granule_spikes[traj][poisson_seed] = granule_spikes_poiss

    storage["grid_spikes"] = copy.deepcopy(grid_spikes)
    storage["granule_spikes"] = copy.deepcopy(granule_spikes)
    storage["parameters"] = parameters
    print("seed " + str(grid_seed) + " completed")
    storage.close()

stop = time.time()
time_sec = stop - start
time_min = time_sec / 60
time_hour = time_min / 60
print("time, ", time_sec, " sec, ", time_min, " min, ", time_hour, "hour  ")
