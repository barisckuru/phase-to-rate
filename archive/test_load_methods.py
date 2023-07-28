# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 11:21:10 2023

@author: Daniel
"""

from phase_to_rate.neural_coding import load_spikes, load_spikes_DMK, load_spikes_DMK_plus_lec, rate_n_phase
import shelve

path_new = r'C:\Users\Daniel\repos\phase-to-rate\data\noise_lec_identical\grid-seed_trajectory_poisson-seeds_duration_shuffling_tuning_pp-weight_noise-scale_340_[75, 74.5, 74, 73.5, 73]_100-109_2000_non-shuffled_full_0.0009_200'

path_old = r'C:\Users\Daniel\repos\phase-to-rate\data\main\full\collective\grid-seed_duration_shuffling_tuning_1_2000_non-shuffled_full'

shelve_old = shelve.open(path_old)

new_read = load_spikes_DMK(path_new, "grid", [75, 74.5, 74, 73.5, 73], 10)

old_read = load_spikes(path_old, 'grid', [75, 74.5, 74, 73.5, 73], 20)

new_counts = [len(x) for x in new_read[75][0]]

old_counts = [len(x) for x in old_read[75][0]]

