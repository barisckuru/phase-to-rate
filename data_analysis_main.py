#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:19:20 2021

@author: baris
"""

from neural_coding_new import load_spikes, rate_n_phase



path = '/home/baris/results/grid-seed_duration_shuffling_tuning_1_2000_shuffled_tuned'


grid_spikes = load_spikes(path, 'grid', [75, 70], 20)


granule_spikes = load_spikes(path, 'granule', [75, 70], 20)

spikes.keys()

type(spikes[75])

len(grid_spikes[75][0])

len(spikes[70])
