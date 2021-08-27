#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 10:50:07 2021

@author: baris
"""


import numpy as np
import random
from scipy import ndimage
from scipy.stats import skewnorm
from skimage.measure import profile_line
from scipy import interpolate
from neo.core import AnalogSignal
import quantities as pq
from elephant import spike_train_generation as stg
import copy
import pdb
import numba as nb


def _grid_maker(spacing, orientation, pos_peak,
                arr_size=200, sizexy=[1, 1], max_rate=1
                ):
    """
    Generate a 2D grid cell firing field.

    Returns a 2D numpy array

    Parameters
    ----------
    spacing : int
        The spacing of the grid cell
    orientation : int
        The orientation (angle) of the grid cell field
    pos_peak : list
        The position  (x,y) of the center of the grid field in the middle
        Also refered as phase of the grid cell
    arr_size : int
        Size of one dimension of the array representing the square field.
        (Resolution)
        The default is 200.
    sizexy : list
        Size of the grid field in meters [x,y]
        The default is [1,1].
    max_rate : int
        Max firing rate defined in grid cell firing field.
        The default is 1.
        Since rate is defined in phase precession model
        max rate could be adjusted here for model without phase prec

    Returns
    -------
        numpy array
        2D firing rate of a grid cell
    """
    arr_size = arr_size
    x, y = pos_peak
    pos_peak = np.array([x, y])
    lambda_spacing = spacing * (
        arr_size / 100
    )  # 100 required for conversion
    k = (4 * np.pi) / (lambda_spacing * np.sqrt(3))
    degrees = orientation
    theta = np.pi * (degrees / 180)
    meterx, metery = sizexy
    arrx = meterx * arr_size  # *arr_size for defining the 2d array size
    arry = metery * arr_size
    dims = np.array([arrx, arry])
    rate = np.ones(dims)
    # dist = np.ones(dims)
    # implementation of grid function
    # 3 ks for 3 cos gratings with different angles
    k1 = (
        (k / np.sqrt(2))
        * np.array(
            (
                np.cos(theta + (np.pi)/12) + np.sin(theta + (np.pi)/12),
                np.cos(theta + (np.pi)/12) - np.sin(theta + (np.pi)/12),
            )
        )
    ).reshape(
        2,
    )
    k2 = (
        (k/np.sqrt(2))
        * np.array(
            (
                np.cos(theta + (5*np.pi)/12) + np.sin(theta + (5*np.pi)/12),
                np.cos(theta + (5*np.pi)/12) - np.sin(theta + (5*np.pi)/12),
            )
        )
    ).reshape(
        2,
    )
    k3 = (
        (k/np.sqrt(2))
        * np.array(
            (
                np.cos(theta + (9*np.pi)/12) + np.sin(theta + (9*np.pi)/12),
                np.cos(theta + (9*np.pi)/12) - np.sin(theta + (9*np.pi)/12),
            )
        )
    ).reshape(
        2,
    )

    # .reshape is only need when function is in the loop
    # (shape somehow becomes (2,1) otherwise normal shape is already (2,)
    for i in range(dims[0]):
        for j in range(dims[1]):
            curr_dist = np.array([i, j] - pos_peak)
            rate[i, j] = (
                np.cos(np.dot(k1, curr_dist))
                + np.cos(np.dot(k2, curr_dist))
                + np.cos(np.dot(k3, curr_dist))
            ) / 3
    rate = (
        max_rate * 2 / 3 * (rate + 1 / 2)
    )
    return rate


def _grid_population(n_grid, seed, arr_size=200):
    """
    Generate a population of grid cells.

    Parameters
    ----------
    n_grid : int
        number of grid cells.
    seed : int
        seed to generate distint populations.
    arr_size : int
        Size of one dimension of the array representing the square field.
        (Resolution)
        The default is 200.

    Returns
    -------
    rate_grids : numpy nd array
        2D firing rate profile of grid cells.
    grid_spc : np array
        grid spacings.
    """
    # skewed normal distribution for grid_spc
    np.random.seed(seed)
    median_spc = 43
    spc_max = 100
    # Negative values are left skewed, positive values are right skewed.
    skewness = 6
    # Skewnorm function
    grid_spc = skewnorm.rvs(a=skewness, loc=spc_max, size=n_grid)
    grid_spc = grid_spc - min(
        grid_spc
    )
    # Standadize all the vlues between 0 and 1.
    # Shift the set so the minimum value is equal to zero.
    grid_spc = grid_spc / max(grid_spc)
    grid_spc = (
        grid_spc * spc_max
    )  # Multiply the standardized values by the maximum value.
    grid_spc = grid_spc + (median_spc - np.median(grid_spc))

    grid_ori = np.random.randint(
        0, high=60, size=[n_grid, 1]
    )  # uniform dist btw 0-60 degrees
    grid_phase = np.random.randint(
        0, high=(arr_size - 1), size=[n_grid, 2]
    )  # uniform dist grid phase
    # create a 3d array with grids for n_grid
    rate_grids = np.zeros((arr_size, arr_size, n_grid))  # empty array
    for i in range(n_grid):
        x = grid_phase[i][0]
        y = grid_phase[i][1]
        rate = _grid_maker(grid_spc[i], grid_ori[i], [x, y])
        rate_grids[:, :, i] = rate
    return rate_grids, grid_spc


def _draw_traj(
    all_grids,
    n_grid,
    par_trajs,
    arr_size=200,
    field_size_cm=100,
    dur_ms=2000,
    speed_cm=20,
):
    """Obtain the firing profile for simulated linear trajectories."""
    size2cm = int(arr_size / field_size_cm)
    dur_s = dur_ms / 1000
    traj_len_cm = int(dur_s * speed_cm)
    traj_len_dp = traj_len_cm * size2cm
    par_idc_cm = par_trajs
    par_idc = par_idc_cm * size2cm - 1
    n_traj = par_trajs.shape[0]
    # empty arrays
    traj = np.empty((n_grid, traj_len_dp))
    trajs = np.empty((n_grid, traj_len_dp, n_traj))

    # draw the trajectories
    for j in range(n_traj):
        idc = par_idc[j]
        for i in range(n_grid):
            traj[i, :] = profile_line(
                all_grids[:, :, i], (idc, 0), (idc, traj_len_dp - 1),
                mode="constant"
            )
            trajs[:, :, j] = traj

    return trajs


def _rate2dist(grids, spacings):
    """Convert rate arrays into linear distance arrays."""
    grid_dist = np.zeros((grids.shape[0], grids.shape[1], grids.shape[2]))
    for i in range(grids.shape[2]):
        grid = grids[:, :, i]
        spacing = spacings[i]
        trans_dist_2d = (
            (np.arccos(((grid * 3 / 2) - 1 / 2)) * np.sqrt(2))
            * np.sqrt(6)
            * spacing
            / (4*np.pi)
        )
        grid_dist[:, :, i] = (trans_dist_2d / (spacing / 2)) / 2
    return grid_dist


def _interp(arr, dur_s, def_dt_s=0.025, new_dt_s=0.002):
    """Interpolate the given array with new dt in seconds."""
    arr_len = arr.shape[1]
    t_arr = np.linspace(0, dur_s, arr_len)
    if (
        new_dt_s != def_dt_s
    ):  # if dt given is different than default_dt_s(0.025), then interpolate
        new_len = int(dur_s / new_dt_s)
        new_t_arr = np.linspace(0, dur_s, new_len)
        f = interpolate.interp1d(t_arr, arr, axis=1)
        interp_arr = f(new_t_arr)
    return interp_arr, new_t_arr


def _import_phase_dist(
        path="/home/baris/phase_coding/norm_grid_phase_dist.npz"):
    """Import default, non shuffled and saved phase distributions."""
    norm_n = np.load(path)["grid_norm_dist"]
    dt_s = 0.001
    dur_s = 0.1
    total_spikes_bin = 25
    phase_prof = (total_spikes_bin / np.sum(norm_n)) * norm_n / (dt_s)
    def_phase_asig = AnalogSignal(
        phase_prof,
        units=1 * pq.Hz,
        t_start=0 * pq.s,
        t_stop=dur_s * pq.s,
        sampling_period=dt_s * pq.s,
        sampling_interval=dt_s * pq.s,
    )
    return def_phase_asig


def _randomize_grid_spikes(arr, bin_size_ms, time_ms=2000):
    """Randomize the phases in time bins without affecting the rate code."""
    def_phase_asig = _import_phase_dist()
    randomized_grid = np.empty(0)
    n_bins = int(time_ms / bin_size_ms)
    for i in range(n_bins):
        curr_ct = ((bin_size_ms * i < arr) & (arr < bin_size_ms * (i+1))).sum()
        curr_train = (
            stg.inhomogeneous_poisson_process(
                def_phase_asig, refractory_period=0.001 * pq.s, as_array=True
            )
            * 1000
        )
        rand_spikes = np.array(random.sample(list(curr_train), k=curr_ct))
        spikes_ms = np.ones(rand_spikes.shape[0]) * (bin_size_ms*i)+rand_spikes
        randomized_grid = np.append(randomized_grid, np.array(spikes_ms))
    return np.sort(randomized_grid)


def _inhom_poiss(
    arr,
    n_traj,
    dur_s,
    shuffle,
    diff_seed=False,
    poiss_seed=0,
    dt_s=0.025,
    dur_ms=2000,
):
    """Generate spikes from a seeded inhomogeneous Poisson function.

    Parameters
    ----------
    shuffle : str
        option to shuffle phases of grid cells
    diff_seed : Boolean
        option to decide seeding inh poiss function
        with same or diff seeds.

    Returns
    -------
    Spike times in miliseconds
    """
    # length of the trajectory that mouse went
    np.random.seed(poiss_seed)
    n_cells = arr.shape[0]
    if shuffle == 'shuffled':
        shuffled = True
    elif shuffle == 'non-shuffled':
        shuffled = False
    else:
        raise ValueError('Shuffling is not defined correctly')

    spi_arr = np.zeros((n_traj, n_cells), dtype=np.ndarray)
    for grid_idc in range(n_cells):
        for i in range(n_traj):
            if diff_seed is True:
                np.random.seed(poiss_seed + grid_idc + (5 * i))
            elif diff_seed is False:
                np.random.seed(poiss_seed + grid_idc)

            rate_profile = arr[grid_idc, :, i]
            asig = AnalogSignal(
                rate_profile,
                units=1 * pq.Hz,
                t_start=0 * pq.s,
                t_stop=dur_s * pq.s,
                sampling_period=dt_s * pq.s,
                sampling_interval=dt_s * pq.s,
            )
            curr_train = (
                stg.inhomogeneous_poisson_process(
                    asig, refractory_period=0.001 * pq.s, as_array=True
                )
                * 1000
            )
            if shuffled is True:
                curr_train = _randomize_grid_spikes(curr_train,
                                                    100, time_ms=dur_ms)
            spi_arr[i, grid_idc] = np.array(curr_train)  # time conv to ms
    spi_arr = [[cell for cell in traj] for traj in spi_arr]
    return spi_arr


def _overall(dist_trajs, rate_trajs, shift_deg, T,
             n_traj, rate_scale, speed_cm, scaling_factor=5, dur_s=2):
    """
    Generate the overall oscillatory firing profile for simulated trajecotries.

    Parameters
    ----------
    dist_trajs : numpy nd array
        linear distance change on trajectories.
    rate_trajs : numpy nd array
        rate change on trajectories.
    shift_deg : int
        amount of phase precession in degrees.
    T : int
        period of theta oscillation.
    n_traj : int
        number of trajectories.
    rate_scale : int
        for adjusting the firing rate.
    speed_cm : int
        walking speed of mouse in centimeters,
        increases the firing rate.
     scaling_factor : int
        adjust the firing rate. The default is 5.
    dur_s : int
        duration of simulation in seconds. The default is 2.

    Returns
    -------
    overall : numpy nd array
        overall firing profile of grid cells with
        oscillations and direction of movement.

    """
    # infer the direction out of rate of change in the location
    direction = np.diff(dist_trajs, axis=1)
    # last element is same with the -1 element of diff array
    direction = np.concatenate((direction, direction[:, -1:, :]), axis=1)
    # values larger > or < 1 indicates the direction 1 or -1
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    # rate of change is inversed for the direction
    direction = -direction
    traj_dist_dir = dist_trajs * direction
    traj_dist_dir = ndimage.gaussian_filter1d(traj_dist_dir, sigma=1, axis=1)
    traj_dist_dir, dist_t_arr = _interp(traj_dist_dir, dur_s)
    factor = shift_deg / 360  # adjust the phase shift with a factor
    one_theta_phase = (2 * np.pi * (dist_t_arr % T) / T) % (2 * np.pi)
    theta_phase = np.repeat(one_theta_phase[np.newaxis, :], 200, axis=0)
    theta_phase = np.repeat(theta_phase[:, :, np.newaxis], n_traj, axis=2)
    firing_phase_dir = 2 * np.pi * (traj_dist_dir + 0.5) * factor
    phase_code_dir = np.exp(1.5 * np.cos(firing_phase_dir - theta_phase))
    # firing rate could be scaled by scaling factor or speed
    scaling_factor = 5
    constant_mv = 0.16
    overall = phase_code_dir * rate_trajs * speed_cm * constant_mv * scaling_factor
    return overall


def grid_simulate(
    trajs,
    dur_ms,
    grid_seed,
    poiss_seeds,
    shuffle,
    diff_seed,
    n_grid=200,
    speed_cm=20,
    rate_scale=1,
    arr_size=200,
    f=10,
    shift_deg=180,
    dt_s=0.002,
):
    """
    Simulate the activity of a population of grid cells.

    Returns spike times and spacings

    Parameters
    ----------
    trajs : int or numpy array
        Location of parallel trajectories to simulate
    dur_ms : int
        Duration of the simulation in milliseconds
    grid_seed : int
        Seed for the deterministic generation of a random grid cell population
    poiss_seeds : numpy array
        Seeds to simulate different trials via inhomegenous poisson function
    shuffle : str ("shuffled" or "non-shuffled")
        Shuffles the spike times in individual time bins
    diff_seed : Boolean
        Decides if inhomogenous poisson function is reseeded
        with a new set of seeds for different trajectories
    n_grid : int
        Number of grid cells in the population
    speed_cm : int
        Speed of the simulated mouse in centimeters
        Effects the firing rate of the cells
        Faster the mouse, higher the firing rate
    rate_scale : int
        Scale the firing rate of grid cells

    Returns
    -------
    grid_spikes : dict
        Each key is a poisson seed containing a list of trajectories which
        in turn contains a list of cells. This means:
        len(grid_spikes.keys()) == len(poiss_seeds)
        len(grid_spikes[key]) == len(trajs) and
        len(grid_spikes[key][0]) == n_grid
    spacings : numpy array
        Spacings of the grid cell population
    """
    dur_s = dur_ms / 1000
    T = 1 / f

    trajs = np.array(trajs)
    n_traj = trajs.shape[0]

    if type(poiss_seeds) is int:
        poiss_seeds = np.array([poiss_seeds])

    grid_spikes = {}

    grids, spacings = _grid_population(
        n_grid, seed=grid_seed, arr_size=arr_size
    )
    grid_dist = _rate2dist(grids, spacings)
    dist_trajs = _draw_traj(grid_dist, n_grid, trajs, dur_ms=dur_ms)
    rate_trajs = _draw_traj(grids, n_grid, trajs, dur_ms=dur_ms)
    rate_trajs, rate_t_arr = _interp(rate_trajs, dur_s)
    overall = _overall(
        dist_trajs, rate_trajs, shift_deg, T, n_traj, rate_scale, speed_cm
    )

    for idx, poiss_seed in enumerate(poiss_seeds):
        curr_grid_spikes = _inhom_poiss(
            overall,
            n_traj,
            dur_s,
            shuffle=shuffle,
            diff_seed=diff_seed,
            dt_s=dt_s,
            poiss_seed=poiss_seed,
        )
        grid_spikes[poiss_seed] = curr_grid_spikes

    return grid_spikes, spacings

if __name__ == '__main__':
    test_grids, _ = grid_simulate([75, 74.5, 74, 70], 2000, 1, np.array([150, 250, 350]), "non-shuffled", False)
