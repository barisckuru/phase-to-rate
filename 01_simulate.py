"""
Simulate pydentate with grid cell inputs on linear trajectories.

This is one of the most computationally intensive steps as it generates
all the raw data that is used for the perceptron, tempotron as well as
CA3 analysis. It allows for reproducibility through a variety of seeding
steps.
"""

import shelve
import copy
import numpy as np
import os
from pydentate import neuron_tools
from phase_to_rate import grid_model
from phase_to_rate import pydentate_integrate

"""Setup"""
dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

neuron_tools.load_compiled_mechanisms()


"""Seeding and trajectories"""
grid_seeds = np.arange(1, 31, 1)

"""
All trajectories and their poisson seeds in the paper:
# trajectories, p = [75], 100
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
"""

trajectories, p = [75], 100

poisson_seeds = np.arange(p, p + 20, 1)
poisson_seeds = list(poisson_seeds)

"""Parameters"""
shuffling = "shuffled"  # "non-shuffled" or "shuffled"
network_type = "full"  # "full", "no-feedback", "no-feedforward" or "disinhibited"

parameters = {}
parameters = {
    "dur_ms": 2000,
    "pp_weight": 9e-4,
    "speed": 20,
    "n_grid": 200,
    "rate_scale": 5,
    "poisson_seeds": poisson_seeds,
}

"""IMPORTANT NOTE ON GC RATE ADJUSTMENT
TODO
"""

verbose = True

"""Start simulating each grid seed"""
for grid_seed in grid_seeds:
    file_name = ("grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_pp-weight_" +
        f"{grid_seed}_{trajectories}_{poisson_seeds[0]}-{poisson_seeds[-1]}_{parameters['dur_ms']}_{shuffling}_{network_type}_{parameters['pp_weight']}")
    file_path = os.path.join(results_dir, file_name)

    if verbose: print(f"Start simulating {file_name}")

    grid_spikes, _ = grid_model.grid_simulate(
        trajs=trajectories,
        dur_ms=parameters['dur_ms'],
        grid_seed=grid_seed,
        poiss_seeds=poisson_seeds,
        shuffle=shuffling,
        n_grid=parameters['n_grid'],
        speed_cm=parameters['speed'],
        rate_scale=parameters['rate_scale'],
    )

    granule_spikes = {}
    for traj in trajectories:
        granule_spikes[traj] = {}
        for poisson_seed in poisson_seeds:
            granule_spikes[traj][poisson_seed] = {}
            granule_spikes_poiss = pydentate_integrate.granule_simulate(
                grid_spikes[traj][poisson_seed],
                dur_ms=parameters['dur_ms'],
                network_type=network_type,
                grid_seed=grid_seed,
                pp_weight=parameters['pp_weight'],
            )
            granule_spikes[traj][poisson_seed] = granule_spikes_poiss

    storage = shelve.open(file_path, writeback=True)
    storage["grid_spikes"] = copy.deepcopy(grid_spikes)
    storage["granule_spikes"] = copy.deepcopy(granule_spikes)
    storage["parameters"] = parameters
    storage.close()
    print(f"Done simulating {file_name}")


