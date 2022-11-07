#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 13:28:49 2021

@author: baris
"""

from pydentate import net_tunedrev, neuron_tools
import numpy as np
import scipy.stats as stats
import pdb
from neuron import h, gui


def granule_simulate(
    grid_spikes,
    dur_ms=2000,
    network_type='full',
    grid_seed=1,
    pp_weight=9e-4,
    input_scale=1000,
    n_grid=200,
    n_granule=2000,
    n_mossy=60,
    n_basket=24,
    n_hipp=24
):
    """

    Simulate biophysically realistic model of the dentate gyrus (pyDentate).

    Parameters
    ----------
    grid_spikes : list
        Spike times of grid cell population.
    dur_ms : int
        Duration of the simulation.
    network_type : str
        Tuning of the network.
    grid_seed : int
        Seed for the grid cell population
        also seeds the dentate gyrus model.
    pp_weight : int
        Connection weight from perforant path
        from grid cell to dentate gyrus cells.
    input_scale : int
        Scale of the input. The default is 1000.
    n_grid : int
        Number of grid cells. The default is 200.
    n_granule : int
        Number granule cells. The default is 2000.
    n_mossy : int
        Number of mossy cells. The default is 60.
    n_basket : int
        Number of basket cells. The default is 24.
    n_hipp : int
        Number of Hillar perforant path cells. The default is 24.

    Raises
    ------
    ValueError
        If tuning of network is invalid.

    Returns
    -------
    granule_spikes : list
        Granule cell spike times in a list.

    """
    np.random.seed(grid_seed)
    # Randomly choose target cells for the PP lines
    gauss_gc = stats.norm(loc=1000, scale=input_scale)
    gauss_bc = stats.norm(loc=12, scale=(input_scale / float(n_granule)) * 24)
    pdf_gc = gauss_gc.pdf(np.arange(n_granule))
    pdf_gc = pdf_gc / pdf_gc.sum()
    pdf_bc = gauss_bc.pdf(np.arange(n_basket))
    pdf_bc = pdf_bc / pdf_bc.sum()
    GC_indices = np.arange(n_granule)
    start_idc = np.random.randint(0, n_granule - 1, size=n_grid)

    PP_to_GCs = []
    for x in start_idc:
        curr_idc = np.concatenate((GC_indices[x:n_granule], GC_indices[0:x]))
        PP_to_GCs.append(np.random.choice(curr_idc,
                                          size=100, replace=False, p=pdf_gc))

    PP_to_GCs = np.array(PP_to_GCs)

    BC_indices = np.arange(n_basket)
    start_idc = np.array(((start_idc / float(n_granule)) * 24), dtype=int)

    PP_to_BCs = []
    for x in start_idc:
        curr_idc = np.concatenate((BC_indices[x:n_basket], BC_indices[0:x]))
        PP_to_BCs.append(np.random.choice(curr_idc,
                                          size=1, replace=False, p=pdf_bc))

    PP_to_BCs = np.array(PP_to_BCs)

    nw = net_tunedrev.TunedNetwork(
        None,
        np.array(grid_spikes, dtype=object),
        np.array(PP_to_GCs),
        np.array(PP_to_BCs),
        network_type=network_type,
        pp_weight=pp_weight,
    )
    neuron_tools.run_neuron_simulator(t_stop=dur_ms)

    # granule_spikes = nw.populations[0].get_timestamps()
    granule_spikes = [x[0].as_numpy() for x in nw.populations[0].ap_counters]

    return granule_spikes



def granule_simulate_noisy(
    grid_spikes,
    dur_ms=2000,
    network_type='full',
    grid_seed=1,
    pp_weight=9e-4,
    input_scale=1000,
    n_grid=200,
    n_granule=2000,
    n_mossy=60,
    n_basket=24,
    n_hipp=24,
    noise_scale=0.05
):
    """
    Simulate biophysically realistic model of the dentate gyrus (pyDentate).

    Parameters
    ----------
    grid_spikes : list
        Spike times of grid cell population.
    dur_ms : int
        Duration of the simulation.
    network_type : str
        Tuning of the network.
    grid_seed : int
        Seed for the grid cell population
        also seeds the dentate gyrus model.
    pp_weight : int
        Connection weight from perforant path
        from grid cell to dentate gyrus cells.
    input_scale : int
        Scale of the input. The default is 1000.
    n_grid : int
        Number of grid cells. The default is 200.
    n_granule : int
        Number granule cells. The default is 2000.
    n_mossy : int
        Number of mossy cells. The default is 60.
    n_basket : int
        Number of basket cells. The default is 24.
    n_hipp : int
        Number of Hillar perforant path cells. The default is 24.

    Raises
    ------
    ValueError
        If tuning of network is invalid.

    Returns
    -------
    granule_spikes : list
        Granule cell spike times in a list.

    """
    np.random.seed(grid_seed)
    # Randomly choose target cells for the PP lines
    gauss_gc = stats.norm(loc=1000, scale=input_scale)
    gauss_bc = stats.norm(loc=12, scale=(input_scale / float(n_granule)) * 24)
    pdf_gc = gauss_gc.pdf(np.arange(n_granule))
    pdf_gc = pdf_gc / pdf_gc.sum()
    pdf_bc = gauss_bc.pdf(np.arange(n_basket))
    pdf_bc = pdf_bc / pdf_bc.sum()
    GC_indices = np.arange(n_granule)
    start_idc = np.random.randint(0, n_granule - 1, size=n_grid)

    PP_to_GCs = []
    for x in start_idc:
        curr_idc = np.concatenate((GC_indices[x:n_granule], GC_indices[0:x]))
        PP_to_GCs.append(np.random.choice(curr_idc,
                                          size=100, replace=False, p=pdf_gc))

    PP_to_GCs = np.array(PP_to_GCs)

    

    BC_indices = np.arange(n_basket)
    start_idc = np.array(((start_idc / float(n_granule)) * 24), dtype=int)

    PP_to_BCs = []
    for x in start_idc:
        curr_idc = np.concatenate((BC_indices[x:n_basket], BC_indices[0:x]))
        PP_to_BCs.append(np.random.choice(curr_idc,
                                          size=1, replace=False, p=pdf_bc))

    PP_to_BCs = np.array(PP_to_BCs)

    nw = net_tunedrev.TunedNetwork(
        None,
        np.array(grid_spikes, dtype=object),
        np.array(PP_to_GCs),
        np.array(PP_to_BCs),
        network_type=network_type,
        pp_weight=pp_weight,
    )
    dt = 0.1

    """CREATE NOISE"""
    noise = np.random.normal(0, noise_scale, size=(n_granule, int(dur_ms/dt)))
    noise_clamps = []
    noise_vectors = []
    time_vectors = []
    for idx, gc in enumerate(nw.populations[0]):
        ic = h.IClamp(gc.soma(0.5))
        ic.delay = 0.0
        ic.dur = 1e9
        tvec = h.Vector()
        vec = h.Vector()
        vec.from_python(noise[idx])
        vec.play(ic._ref_amp, dt)
        noise_clamps.append(ic)
        noise_vectors.append(vec)
        time_vectors.append(tvec)
        # pdb.set_trace()
    nw.populations[0].voltage_recording(range(2000))

    neuron_tools.run_neuron_simulator(t_stop=dur_ms, dt_sim=dt)
    granule_spikes = [x[0].as_numpy() for x in nw.populations[0].ap_counters]


    return granule_spikes




