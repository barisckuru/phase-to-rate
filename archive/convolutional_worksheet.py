#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 16:32:14 2021

@author: baris
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os, glob
from rate_n_phase_code_gra import phase_code, overall_spike_ct
import time
import copy
from neuron import h, gui  # gui necessary for some parameters to h namespace
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
from result_processor import all_codes
import pandas as pd
from scipy import stats



path1 = '/home/baris/results/grid_mixed_input/new/'



n_samples = 20
grid_s = 420

#nonshuffled 75-74.5-74-73.5 

grid_rate_ns = np.empty((16,n_samples,8000))
gra_rate_ns = np.empty((16,n_samples,80000))
grid_phase_ns = np.empty((16,n_samples,8000))
gra_phase_ns = np.empty((16,n_samples,80000))
npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*non*75.0*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)

    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[0,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[1,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[2,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[3,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[0,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[1,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[2,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[3,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[0,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[1,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[2,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[3,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[0,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[1,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[2,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[3,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    if ct==(int(n_samples/5)):
        break
   
#shuffled 75-74.5-74-73.5 

grid_rate_s = np.empty((16,n_samples,8000))
gra_rate_s = np.empty((16,n_samples,80000))
grid_phase_s = np.empty((16,n_samples,8000))
gra_phase_s = np.empty((16,n_samples,80000))
npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*75.0*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[0,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[1,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[2,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[3,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[0,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[1,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[2,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[3,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[0,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[1,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[2,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[3,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[0,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[1,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[2,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[3,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break


#nonshuffled 73-72.5-72-71.5

npzfiles = []
ct=0
for file in sorted(glob.glob(os.path.join(path1,'*non*73.0*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[4,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[5,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[6,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[7,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[4,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[5,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[6,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[7,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[4,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[5,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[6,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[7,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[4,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[5,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[6,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[7,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    if ct==(int(n_samples/5)):
        break 
   
#shuffled  73.72.5-72-71.5


npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*73.0*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[4,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[5,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[6,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[7,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[4,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[5,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[6,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[7,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[4,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[5,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[6,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[7,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[4,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[5,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[6,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[7,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break


#nonshuffled 71-70.5-70-69

npzfiles = []
ct=0
for file in sorted(glob.glob(os.path.join(path1,'*non*69*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[8,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[9,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[10,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[11,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[8,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[9,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[10,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[11,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[8,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[9,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[10,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[11,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[8,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[9,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[10,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[11,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    if ct==(int(n_samples/5)):
        break 
   
#shuffled  71-70.5-70-69


npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*69*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[8,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[9,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[10,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[11,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[8,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[9,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[10,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[11,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[8,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[9,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[10,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[11,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[8,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[9,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[10,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[11,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break


#nonshuffled 68-67-66-65

npzfiles = []
ct=0
for file in sorted(glob.glob(os.path.join(path1,'*non*68*seed'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    
    grid_rate_ns[12,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[13,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[14,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[15,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    
    gra_rate_ns[12,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[13,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[14,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[15,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    
    grid_phase_ns[12,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[13,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[14,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[15,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]

    gra_phase_ns[12,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[13,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[14,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[15,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    if ct==(int(n_samples/5)):
        break 
   
#shuffled  68-67-66-65


npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*68*seed'+str(grid_s)+'_2000ms.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,1]
    grid_rate_s[12,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[13,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[14,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[15,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,1]
    gra_rate_s[12,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[13,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[14,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[15,5*ct:5+5*ct,:] = t735_gra

    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,1]
    grid_phase_s[12,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[13,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[14,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[15,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,1]
    gra_phase_s[12,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[13,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[14,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[15,5*ct:5+5*ct,:] = t735_gra_phase
    ct+=1
    if ct==(int(n_samples/5)):
        break
    
   
    
   
    
   
grid_rate_s.shape    

sample_set = np.vstack((grid_rate_ns[0,:,:], grid_rate_ns[5,:,:]))
    
labels = np.append(np.zeros(20),np.ones(20))
    

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=3)
decision_tree = decision_tree.fit(sample_set, labels)
r = export_text(decision_tree)
print(r)
    
   
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
iris = load_iris()
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(iris.data, iris.target)
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)
   
    
   
#parameters
lr=1e-4
n_samples = 20
grid_s = 420


#labels, output for training the network, 5 for each trajectory



def label(n_poiss):
    a = np.tile([1, 0], (n_poiss,1))
    b = np.tile([0, 1], (n_poiss,1))
    labels = np.vstack((a,b))
    labels = torch.FloatTensor(labels) 
    out_len = labels.shape[1]
    return labels, out_len

#BUILD THE NETWORK

class Net(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)
    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        return y

#TRAIN THE NETWORK

def train_net(net, train_data, test_data, labels, labels_test, n_iter, lr):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    track_loss = []
    loss_fn = nn.MSELoss(reduction='mean')
    loss_fn_test = nn.MSELoss(reduction='mean')
    test_loss=[]
    for i in range(n_iter):
        train_data = torch.FloatTensor(train_data)
        out = net(train_data)
        loss = torch.sqrt(loss_fn(out, labels))
        test_data = torch.FloatTensor(test_data)
        out_test = net(test_data)
        loss_test = torch.sqrt(loss_fn_test(out_test, labels_test))
        # Compute gradients
        optimizer.zero_grad()
        loss.backward()
    
        # Update weights
        optimizer.step()
    
        # Store current value of loss
        track_loss.append(loss.item())  # .item() needed to transform the tensor output of loss_fn to a scalar
        test_loss.append(loss_test.item())
        # print(loss_test)
        # Track progress
        if (i + 1) % (n_iter // 5) == 0:
          print(f'iteration {i + 1}/{n_iter} | loss: {loss.item():.3f}')

    return track_loss, test_loss, out


   
    
def leave_one_out(code, lr=1e-3, n_iter=50000, th=0.2, grid_seeds = np.array([grid_s]), perc_seed=420):
    _75 = code[0,:,:]
    _65 = code [6,:,:]
    trajs = ' 75 vs 71 '
    code_name = 'Granule Rate shuffled'
    inp_len = code.shape[2]
    plt.close('all')
    fig, ax = plt.subplots()
    for i in range(int(_75.shape[0])):
        train_75 = np.vstack((_75[:i,:], _75[i+1:,:]))
        train_65 = np.vstack((_65[:i,:], _65[i+1:,:]))
        train = np.vstack((train_75, train_65))
        print(train.shape)
        test = np.vstack((_75[i,:], _65[i,:]))
        labels, out_len = label(int(train.shape[0]/2))
        labels_test, out_len_test = label(int(test.shape[0]/2))
        torch.manual_seed(perc_seed+i)
        net = Net(inp_len, out_len)
        print(n_iter)
        track_loss, test_loss, out = train_net(net, train, test, labels, labels_test, n_iter=n_iter, lr=lr)
        ax.plot(track_loss, 'b-')
        ax.plot(test_loss, 'k-')
    ax.set_title(code_name + trajs+'  blue=train & black=test \n n_train='+str(train.shape[0])+' n_test='+str(test.shape[0])+' lr='+str(lr))
    print(i)
    
 
leave_one_out(grid_rate_s)  
# leave_one_out(grid_phase_s)      
# leave_one_out(gra_rate_s) 
# leave_one_out(gra_phase_s) 


first = np.empty((20, 8000))
second = np.empty((20, 8000))


for i in range(20):
    np.random.seed(seed=100)
    signal = np.random.rand(8000)
    noise_seed = np.random.randint(8000)
    np.random.seed(seed=noise_seed+i)
    noise = np.random.rand(8000)/2
    
    first[i, :] = signal+noise
    
    np.random.seed(seed=101)
    signal_2 = np.random.rand(8000)
    noise_seed_2 = np.random.randint(8000)
    np.random.seed(seed=noise_seed_2+i)
    noise_2 = np.random.rand(8000)/2
    
    second[i, :] = signal_2+noise_2
    
    
load = np.load(npzfiles[0], allow_pickle=True)

grid_spikes = load['grid_spikes_sim'][0]


n_samples = 20
grid_s = 420

#nonshuffled 75-74.5-74-73.5 

grid_rate_ns = np.empty((16,n_samples,8000))
gra_rate_ns = np.empty((16,n_samples,80000))
grid_phase_ns = np.empty((16,n_samples,8000))
gra_phase_ns = np.empty((16,n_samples,80000))
npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*non*'+str(grid_s)+'_2000ms.npz'))):
    npzfiles.append(file)
    for i in 
    load = np.load(file, allow_pickle=True)
    



sample_set = np.empty((2000,200, 15, 20))

def counter(grid_spikes):
    n_poiss= grid_spikes.shape[0]
    for i in range(n_poiss):
       binned = binned_ct(grid_spikes[i],1 )
       sample_set[:,:,0,i] = binned[0]
       sample_set[:,:,0,i] = binned[0]



def binned_ct(arr, bin_size_ms, time_ms=2000):
    n_bins = int(time_ms/bin_size_ms)
    n_cells = arr.shape[0] 
    n_traj = arr.shape[1]
    counts = np.zeros((n_bins, n_cells, n_traj))
    for i in range(n_bins):
        for index, value in np.ndenumerate(arr):
            curr_ct = ((bin_size_ms*(i) < value) & (value < bin_size_ms*(i+1))).sum()
            counts[i][index] = curr_ct
            #search and count the number of spikes in the each bin range
    return counts




binned_matrix = binned_ct(grid_spikes, 1)

plt.figure()
plt.imshow(binned_matrix[:,:,0])


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 15)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
