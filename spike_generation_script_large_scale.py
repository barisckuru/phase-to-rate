"""
This script generates grid cell spikes, feeds them into pydentate to
generate granule cell spikes and saves both as time stamps.
This script is particularly designed to simulate a large number
of poisson seeds without exceeding memory.
"""


from pydentate import neuron_tools
from grid_model import grid_simulate
import shelve
import copy
import time
import numpy as np
from pydentate_integrate import granule_simulate
import os
import argparse
import pdb

pr = argparse.ArgumentParser(description='Simulate pydentae with grid cell inputs')
pr.add_argument('-gs',
                type=int,
                help='The grid seed',
                dest='grid_seed',
                required=True)
pr.add_argument('-ps',
                nargs=3,
                type=int,
                help='Start, stop and step of poisson seed range',
                dest='poisson_seeds',
                required=True)
pr.add_argument('-t',
                nargs=1,
                type=float,
                help='The trajectories',
                dest='trajectories',
                required=True)
pr.add_argument('-s',
                type=str,
                help='"shuffled" or "non-shuffled"',
                dest='shuffling',
                required=True)
pr.add_argument('-n',
                type=str,
                help='The network type, "full", "no-feedback", "no-feedforward" or "disinhibited"',
                dest='network_type',
                required=True)
args = pr.parse_args()

# Load neuron dll
neuron_tools.load_compiled_mechanisms()

"""Parameters"""
p = 1500
poisson_seed_start, poisson_seed_stop, poisson_seed_step = args.poisson_seeds
poisson_seeds = range(poisson_seed_start, poisson_seed_stop, poisson_seed_step)
grid_seed = args.grid_seed
trajectories = args.trajectories
shuffling = args.shuffling
speed_cm = 20
dur_ms = 2000  # TODO
rate_scale = 5
n_grid = 200
pp_weight = 9e-4
network_type = args.network_type

# Store some parameters
parameters = {}
parameters = {
    "dur_ms": dur_ms,
    "pp_weight": pp_weight,
    "speed": speed_cm,
    "n_grid": n_grid,
    "rate_scale": rate_scale,
    "poisson_seeds": poisson_seeds,
}

print("grid", grid_seed, "poiss", poisson_seeds,
      "traj", trajectories, "dur", dur_ms)

# Define the output directory, create if necessary
dir_name = os.path.dirname(__file__)  # Name of script dir
save_dir = os.path.join(dir_name, 'data', f'network_{network_type}', f'seed_{grid_seed}')

# if not os.path.isdir(save_dir):
os.makedirs(save_dir, exist_ok=True)

file_name_signature = "grid-seed_trajectories_poisson-seeds_duration_shuffling_tuning"
file_name_data = f"{grid_seed}_{trajectories}_{args.poisson_seeds}_{dur_ms}_{shuffling}_{network_type}"
file_name = f"{file_name_signature}_{file_name_data}"
shelve_loc = os.path.join(save_dir, file_name)

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
del _  # Delete unused output to free up memory

# Write grid_spikes, parameters and initialize storage for granule cells
with shelve.open(shelve_loc) as storage:
    # If grid spikes already exist on file, check data integrity
    if "grid_spikes" in storage.keys():
        k1 = list(storage['grid_spikes'].keys())[0]
        k2 = list(storage['grid_spikes'][k1].keys())[0]

        compare = storage['grid_spikes'][k1][k2][0] == grid_spikes[k1][k2][0]
        if not compare.all():
            raise ValueError("grid_spikes on file not equal simulated grid_spikes. Check storage integrity!")
    # If grid spikes do not already exist, write them to file
    else:
        storage["grid_spikes"] = copy.deepcopy(grid_spikes)
        storage["parameters"] = parameters

    # Create granule_spikes dict only if it does not already exist
    if not "granule_spikes" in storage.keys():
        storage["granule_spikes"] = {traj:{} for traj in trajectories}
        
# pdb.set_trace()
for traj in trajectories:
    for poisson_seed in poisson_seeds:
        # Skip if the poisson seed already exists on file
        with shelve.open(shelve_loc) as storage:
            if poisson_seed in storage["granule_spikes"][traj].keys():
                print(f"Poisson seed {poisson_seed} already on file! Skipped.")
                continue
        print(f"Current poisson seed: {poisson_seed}")
        granule_spikes_poiss = granule_simulate(
            grid_spikes[traj][poisson_seed],
            dur_ms=dur_ms,
            network_type=network_type,
            grid_seed=grid_seed,
            pp_weight=pp_weight,
        )

        # Write granule spikes from the current poisson seed to file
        with shelve.open(shelve_loc, writeback=True) as storage:
            storage["granule_spikes"][traj][poisson_seed] = copy.deepcopy(granule_spikes_poiss)

            

print("seed " + str(grid_seed) + " completed")
