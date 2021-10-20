#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:52:52 2021

@author: baris
"""


import numpy as np
import copy
import shelve
import os
import glob


def _spike_counter(spike_times, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms / bin_size_ms)
    n_cells = len(spike_times)
    counts = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, value in enumerate(spike_times):
            curr_ct = (
                (bin_size_ms * (i) < value) & (value < bin_size_ms * (i + 1))
            ).sum()
            counts[idx, i] = curr_ct
    return counts


def _phase_definer(spike_times, nan_fill=False, bin_size_ms=100, dur_ms=2000):
    n_bins = int(dur_ms / bin_size_ms)
    n_cells = len(spike_times)
    phases = np.zeros((n_cells, n_bins))
    for i in range(n_bins):
        for idx, val in enumerate(spike_times):
            curr_train = val[
                ((bin_size_ms * (i) < val) & (val < bin_size_ms * (i + 1)))
            ]
            if curr_train.size != 0:
                phases[idx, i] = np.mean(
                    curr_train % (bin_size_ms) / (bin_size_ms) * 2 * np.pi
                )
    if nan_fill is True:
        mean_phases = np.mean(phases[phases != 0])
        phases[phases == 0] = mean_phases
    return phases


def _code_maker(
    single_count, single_phase, phase_of_rate_code=np.pi / 4, rate_in_phase=1
):
    single_count = single_count.flatten("C")
    single_phase = single_phase.flatten("C")

    cts_for_phase = copy.deepcopy(single_count)
    # change rate code to mean of non zeros where it is nonzero
    # cts_for_phase[cts_for_phase!=0]=np.mean(cts_for_phase[cts_for_phase!=0])
    cts_for_phase[cts_for_phase != 0] = rate_in_phase

    # rate code with constant 45 deg phase
    rate_y = single_count * np.sin(phase_of_rate_code)
    rate_x = single_count * np.cos(phase_of_rate_code)
    rate_code = np.concatenate((rate_x, rate_y), axis=None)

    # phase code with phase and mean rate
    phase_y = cts_for_phase * np.sin(single_phase)
    phase_x = cts_for_phase * np.cos(single_phase)
    phase_code = np.concatenate((phase_x, phase_y), axis=None)

    # polar code with rate and phase
    polar_y = single_count * np.sin(single_phase)
    polar_x = single_count * np.cos(single_phase)
    polar_code = np.concatenate((polar_x, polar_y), axis=None)

    return rate_code, phase_code, polar_code


def rate_n_phase(spike_times,
                 trajectories,
                 n_samples,
                 bin_size_ms=100,
                 dur_ms=2000):
    """

    Generate spike counts and phases as well as different coding schemes.

    Parameters
    ----------
    spike_times : dict
        Spike times from different trajectories.
    trajectories : list
        List of trajectories.
    n_samples : int
        number of samples with different poisson seeds.
    bin_size_ms : int
        Time bin size in milliseconds. The default is 100.
    dur_ms : int, optional
        Duration of the simulation. The default is 2000.

    Returns
    -------
    counts : numpy array
        Spike counts in each time bin for each cell, sample and trajectory.
        [n_cell, n_bins, n_samples, n_traj]
    phases : numpy array
        Spike phases in each time bin for each cell, sample and trajectory.
        [n_cell, n_bins, n_samples, n_traj]
    rate_code : numpy array
        Flattened rate code for each sample and trajectory.
        [2*n_cell*n_bins, n_samples, n_traj]
    phase_code : numpy array
        Flattened phase code for each sample and trajectory.
        [2*n_cell*n_bins, n_samples, n_traj]
    polar_code : numpy array
        Flattened polar code for each sample and trajectory.
        [2*n_cell*n_bins, n_samples, n_traj]

    """
    n_bins = int(dur_ms / bin_size_ms)
    n_traj = len(trajectories)
    n_cell = len(spike_times[trajectories[0]][0])
    counts = np.empty((n_cell, n_bins, n_samples, n_traj))
    phases = np.empty((n_cell, n_bins, n_samples, n_traj))
    rate_code = np.empty((2 * n_cell * n_bins, n_samples, n_traj))
    phase_code = np.empty((2 * n_cell * n_bins, n_samples, n_traj))
    polar_code = np.empty((2 * n_cell * n_bins, n_samples, n_traj))

    for traj_idx, traj in enumerate(trajectories):
        spike_times_traj = spike_times[traj]
        for sample_idx in range(n_samples):
            spike_times_sample = spike_times_traj[sample_idx]
            single_count = _spike_counter(
                spike_times_sample, bin_size_ms=bin_size_ms, dur_ms=dur_ms
            )
            single_phase = _phase_definer(
                spike_times_sample, bin_size_ms=bin_size_ms, dur_ms=dur_ms
            )
            counts[:, :, sample_idx, traj_idx] = single_count
            phases[:, :, sample_idx, traj_idx] = single_phase
            s_rate_code, s_phase_code, s_polar_code = _code_maker(
                single_count, single_phase
            )
            rate_code[:, sample_idx, traj_idx] = s_rate_code
            phase_code[:, sample_idx, traj_idx] = s_phase_code
            polar_code[:, sample_idx, traj_idx] = s_polar_code
    return counts, phases, rate_code, phase_code, polar_code


def load_spikes(path, cell_type, trajectories, n_samples):
    """

    Load the spike times from the data generated by simulations.

    Parameters
    ----------
    path : str
        Data loading path.
    cell_type : str
        "grid" or "granule".
    trajectories : list
        List of trajectories.
    n_samples : int
        Number of samples.

    Raises
    ------
    Exception
        If cell type or n_samples is not valid.

    Returns
    -------
    spikes : dict
        returns loaded spikes from different trajectories.

    """
    if not os.path.exists(path):
        print(path)
        print('path does not exist')
        split = path.split("_")
        grid_seed = int(split[4])
        shuffling = split[6]
        dur_ms = split[5]
        _collect_spikes(grid_seed, shuffling, dur_ms)

    storage = shelve.open(path[:-4])
    spikes = {}
    for traj in trajectories:
        print(traj)
        requested_spikes = []
        traj_key = str(traj)
        poisson_seeds = storage[traj_key]["parameters"]["poisson_seeds"]
        if n_samples > len(poisson_seeds):
            raise Exception("Too much samples requested!")
        elif n_samples < 1:
            raise Exception("n_samples should be larger than 0!")
        else:
            poisson_seeds = poisson_seeds[0:n_samples]

        if cell_type == "grid":
            all_spikes = storage[traj_key]["grid_spikes"]
        elif cell_type == "granule":
            all_spikes = storage[traj_key]["granule_spikes"]
        else:
            raise Exception("Cell type does not exist!")
        for poisson in poisson_seeds:
            requested_spikes.append(all_spikes[poisson])
        spikes[traj] = requested_spikes
    storage.close()
    return spikes


def _collect_spikes(
    grid_seed,
    shuffling,
    dur_ms,
    trajectories,
    network_type,
    path="/home/baris/results/"
):

    collective_path = path + str(network_type) + "/collective/"
    separate_path = (path + str(network_type) + "/separate/" +
                     "seed_" + str(grid_seed) + "/")
    print(separate_path)

    for traj in trajectories:
        print(traj)
        file = glob.glob(os.path.join(
            separate_path, "*%s*_%s*.dat" % (traj, shuffling)))[0][0:-4]
        fname = f"{separate_path}*{traj}]*_{shuffling}*.dat"
        file = glob.glob(fname)[0][0:-4]
        print(file)
        storage_old = shelve.open(file)
        output_name = f"{grid_seed}_{dur_ms}"
        collective_storage = []
        collective_storage = (collective_path +
                              "grid-seed_duration_shuffling_tuning_")
        collective_storage = (collective_storage + output_name + "_" +
                              shuffling + "_" + network_type)
        storage = shelve.open(collective_storage, writeback=True)
        traj_key = str(traj)
        storage[traj_key] = {}
        storage[traj_key]["grid_spikes"] = copy.deepcopy(
            storage_old["grid_spikes"][traj]
        )
        storage[traj_key]["granule_spikes"] = copy.deepcopy(
            storage_old["granule_spikes"][traj]
        )
        storage[traj_key]["parameters"] = copy.deepcopy(
            storage_old["parameters"])
        storage.close()
        storage_old.close()


trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72,
                71, 70, 69, 68, 67, 66, 65, 60, 30, 15]

# for i in np.arange(1,11,1):
#     print(i)
#     _collect_spikes(i, 'non-shuffled', 2000, trajectories, 'disinhibited')