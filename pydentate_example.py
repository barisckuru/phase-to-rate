"""This script shows how pyDentate could be used after the
refactoring and gives an entry point to prototype its usage with
the grid model."""

from neuron import h, gui
from pyDentate import net_tunedrev, neuron_tools, inputs
import numpy as np
import scipy.stats as stats

"""Connectivity and time stamp creation still happens on script side.
TODO DMK Create a pydentate module that creates the connectivity."""
input_seed = 100
run = 1
input_scale = 1000
input_frequency = 10  # Irrelevant for grid model

np.random.seed(input_seed+run)
# Randomly choose target cells for the PP lines

gauss_gc = stats.norm(loc=1000, scale=input_scale)
gauss_bc = stats.norm(loc=12, scale=(input_scale/2000.0)*24)
pdf_gc = gauss_gc.pdf(np.arange(2000))
pdf_gc = pdf_gc/pdf_gc.sum()
pdf_bc = gauss_bc.pdf(np.arange(24))
pdf_bc = pdf_bc/pdf_bc.sum()
GC_indices = np.arange(2000)
start_idc = np.random.randint(0, 1999, size=400)

PP_to_GCs = []
for x in start_idc:
    curr_idc = np.concatenate((GC_indices[x:2000], GC_indices[0:x]))
    PP_to_GCs.append(np.random.choice(curr_idc, size=100, replace=False,
                                        p=pdf_gc))

PP_to_GCs = np.array(PP_to_GCs)
PP_to_GCs = PP_to_GCs[0:24]

BC_indices = np.arange(24)
start_idc = np.array(((start_idc/2000.0)*24), dtype=int)

PP_to_BCs = []
for x in start_idc:
    curr_idc = np.concatenate((BC_indices[x:24], BC_indices[0:x]))
    PP_to_BCs.append(np.random.choice(curr_idc, size=1, replace=False,
                                        p=pdf_bc))

PP_to_BCs = np.array(PP_to_BCs)
PP_to_BCs = PP_to_BCs[0:24]

# TODO write inputs.gaussian_connectivity_gc_bc()
# PP_to_GCs = inputs.gaussian_connectivity(n_pre=400, n_post=[2000, 24], n_syn=[100,1])


print(PP_to_GCs.shape)

# Generate temporal patterns for the 100 PP inputs
# TODO replace with 
temporal_patterns = inputs.inhom_poiss(modulation_rate=input_frequency)
temporal_patterns[0:24]