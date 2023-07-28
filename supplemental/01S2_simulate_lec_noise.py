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
import sys
import argparse
import scipy
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
from scipy import interpolate
import pdb

pr = argparse.ArgumentParser(description='Local pattern separation paradigm')
pr.add_argument('-grid_seed',
                type=int,
                help='Grid seed to run',
                default=1,
                dest='grid_seed')

pr.add_argument('-noise_scale',
                type=int,
                help='Scale of noise',
                default=200,
                dest='noise_scale')
args = pr.parse_args()

"""Seeding and trajectories"""
grid_seeds = [args.grid_seed]

trajectories, p = [75], 100

poisson_seeds = np.arange(p, p + 20, 1)
poisson_seeds = list(poisson_seeds)

"""LEC NOISE"""
def lec_noise(n_cells=20, p_seeds=[0]):
    x = np.linspace(0, 1, 2000)
    y = np.flip(scipy.signal.sawtooth(2 * np.pi * 40 * x))
    y += 1
    y = y / 2

    kernel_size = 20
    y_smoothed = scipy.ndimage.gaussian_filter1d(y, 5)
    phase_shift = 25
    y_smoothed_phase_shifted = y_smoothed[phase_shift:1000+phase_shift] * 50
    y_smoothed_phase_shifted -= y_smoothed_phase_shifted.min()
    x = np.linspace(0, 2000, 1000)
    
    # Interpolation
    f = interpolate.interp1d(x, y_smoothed_phase_shifted)
    x_new = np.linspace(0, 2000, 20000)
    
    y_new = f(x_new)

    asig = AnalogSignal(
        y_new,
        units=1 * pq.Hz,
        t_start=0 * pq.s,
        t_stop=2 * pq.s,
        sampling_period=0.0001* pq.s
    )

    spikes = []

    for ps in p_seeds:
        np.random.seed(ps+4442)
        curr_spikes = []
        for gc in range(n_cells):

            curr_train = (
                stg.inhomogeneous_poisson_process(
                    asig, refractory_period=0.001 * pq.s, as_array=True
                )
                * 1000)
            curr_spikes.append(curr_train)
        spikes.append(curr_spikes)

    return spikes
    
lec_spikes = lec_noise(n_cells=20, p_seeds=poisson_seeds)


# Create spikes
"""Setup"""
dirname = os.path.dirname(__file__)
results_dir = os.path.join(dirname, 'data')
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

neuron_tools.load_compiled_mechanisms()

"""Parameters"""
shuffling = "non-shuffled"  # "non-shuffled" or "shuffled"
network_type = "full"  # "full", "no-feedback", "no-feedforward" or "disinhibited"

parameters = {}
parameters = {
    "dur_ms": 2000,
    "pp_weight": 9e-4,
    "speed": 20,
    "n_grid": 200,
    "rate_scale": 5,
    "poisson_seeds": poisson_seeds,
    "noise_scale": args.noise_scale
}

"""IMPORTANT NOTE ON GC RATE ADJUSTMENT
TODO
"""

verbose = True

"""Start simulating each grid seed"""
for grid_seed in grid_seeds:
    file_name = ("grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_pp-weight_noise-scale_" +
        f"{grid_seed}_{trajectories}_{poisson_seeds[0]}-{poisson_seeds[-1]}_{parameters['dur_ms']}_{shuffling}_{network_type}_{parameters['pp_weight']}_{parameters['noise_scale']}")
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
        rate_scale=parameters['rate_scale']
    )

    granule_spikes = {}
    for traj in trajectories:
        granule_spikes[traj] = {}
        for idx, poisson_seed in enumerate(poisson_seeds):
            granule_spikes[traj][poisson_seed] = {}
            granule_spikes_poiss = pydentate_integrate.granule_simulate_lec_noise(
                grid_spikes[traj][poisson_seed],
                lec_spikes[idx],
                dur_ms=parameters['dur_ms'],
                network_type=network_type,
                grid_seed=grid_seed,
                pp_weight=parameters['pp_weight'],
                n_lec_synapses=parameters['noise_scale'],
            )
            granule_spikes[traj][poisson_seed] = granule_spikes_poiss
            
    storage = shelve.open(file_path, writeback=True)
    storage["grid_spikes"] = copy.deepcopy(grid_spikes)
    storage["lec_spikes"] = copy.deepcopy(lec_spikes)
    storage["granule_spikes"] = copy.deepcopy(granule_spikes)
    storage["parameters"] = parameters
    storage.close()
    print(f"Done simulating {file_name}")

# plt.xlim((0, 100))
# plt.ylim((0, 70))
# plt.xlabel("Spike time in theta bin")
# plt.ylabel("Number of Spikes")