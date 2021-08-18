#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 10:33:01 2021

@author: baris
"""



import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import os, glob
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim






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


   
def label_leave1out(n_poiss, idx_leave_out):
    a = np.tile([1, 0], (n_poiss,1))
    b = np.tile([0, 1], (n_poiss,1))
    labels = np.vstack((a,b))
    labels = torch.FloatTensor(labels) 
    labels = np.delete(labels, idx_leave_out, axis=0)
    out_len = labels.shape[1]
    return labels, out_len


def leave_one_out(code, lr=1e-4, n_iter=10000, grid_seeds = np.array([grid_s]), perc_seed=420):
    _75 = code[0,:,:]
    _65 = code [15,:,:]
    trajs = ' 75 vs 65 '
    code_name = 'Grid Rate shuffled'
    inp_len = code.shape[2]
    plt.close('all')
    fig, ax = plt.subplots()
    whole = np.vstack((_75, _65))
    n_sample = int(whole.shape[0])
    for i in range(n_sample):
        train = np.vstack((whole[:i,:], whole[i+1:,:]))
        print(train.shape)
        test = whole[i,:]
        if i < ((n_sample/2)-1):
            label_test = torch.FloatTensor([1,0]) 
        else:
            label_test = torch.FloatTensor([0,1]) 
        labels, out_len = label_leave1out(int(n_sample/2), i)
        print(labels.shape)
        torch.manual_seed(perc_seed+i)
        net = Net(inp_len, out_len)
        print(n_iter)
        track_loss, test_loss, out = train_net(net, train, test, labels, label_test, n_iter=n_iter, lr=lr)
        ax.plot(track_loss, 'b-')
        ax.plot(test_loss, 'k-')
    ax.set_title(code_name + trajs+'  blue=train & black=test \n n_train='+str(train.shape[0])+' n_test=1 lr='+str(lr))
    print(i)
    
 
leave_one_out(grid_rate_s)  
# leave_one_out(grid_phase_ns)      
# leave_one_out(gra_rate_s) 
# leave_one_out(gra_phase_s) 





def correlation (path, n_grid_seed, fname, plot='no'):
    # grid_rate, grid_phase, grid_complex, gra_rate, gra_phase, gra_complex = all_codes(path)
    
    #one time bin
    grid_rate = np.concatenate((grid_rate[:,:,:,1000:1200], grid_rate[:,:,:,5000:5200]), axis=3)
    grid_phase = np.concatenate((grid_phase[:,:,:,1000:1200], grid_phase[:,:,:,5000:5200]), axis=3)
    grid_complex = np.concatenate((grid_complex[:,:,:,1000:1200], grid_complex[:,:,:,5000:5200]), axis=3)
    gra_rate = np.concatenate((gra_rate[:,:,:,10000:12000], gra_rate[:,:,:,50000:52000]), axis=3)
    gra_phase = np.concatenate((gra_phase[:,:,:,10000:12000], gra_phase[:,:,:,50000:52000]), axis=3)
    gra_complex = np.concatenate((gra_complex[:,:,:,10000:12000], gra_complex[:,:,:,50000:52000]), axis=3)
    n_gra = 2000
    n_grid = 200
    n_bin = 20 

    n_comp = int(grid_rate.shape[0]*(grid_rate.shape[0]-1)/2)
    rate_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    rate_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    phase_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    phase_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    complex_grid_corr = np.zeros((n_grid_seed,5,n_comp))
    complex_gra_corr = np.zeros((n_grid_seed,5,n_comp))
    
    ###sorting - color code
    trajectories = np.array([75, 74.5, 74, 73.5, 73, 72.5, 72, 71.5, 71, 70, 65, 60])
    diff = np.subtract.outer(trajectories, trajectories)
    diff = diff[np.triu_indices(12,1)]
    sort = np.argsort(diff, kind='stable')

    
    mean_rate, mean_phase, mean_complex = binned_mean(path,1)
    m_rate_grid = mean_rate[0]
    m_rate_gra = mean_rate[1]
    m_phase_grid = mean_phase[0]
    m_phase_gra = mean_phase[1]
    m_complex_grid = mean_complex[0]
    m_complex_gra = mean_complex[1]
    
    sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=2.5,color_codes=True)
    cmap = sns.color_palette('coolwarm', as_cmap=True)
    fig, (ax_rate, ax_phase, ax_complex) = plt.subplots(1,3, sharey=True, gridspec_kw={'width_ratios':[1,1,1]}, figsize=(15,5))
    ax_rate.plot(m_rate_grid,m_rate_gra, 'k--', linewidth=3)
    ax_phase.plot(m_phase_grid,m_phase_gra, 'k--', linewidth=3)
    ax_complex.plot(m_complex_grid,m_complex_gra, 'k--', linewidth=3)
    for grid in range(n_grid_seed):
        for poiss in range(5):
            grid_rate_corr = pearson_r(grid_rate[:,grid,poiss,:], grid_rate[:,grid,poiss,:])[sort]
            grid_phase_corr = pearson_r(grid_phase[:,grid,poiss,:], grid_phase[:,grid,poiss,:])[sort]
            grid_complex_corr = pearson_r(grid_complex[:,grid,poiss,:], grid_complex[:,grid,poiss,:])[sort]
            gra_rate_corr = pearson_r(gra_rate[:,grid,poiss,:], gra_rate[:,grid,poiss,:])[sort]
            gra_phase_corr = pearson_r(gra_phase[:,grid,poiss,:], gra_phase[:,grid,poiss,:])[sort]
            gra_complex_corr = pearson_r(gra_complex[:,grid,poiss,:], gra_complex[:,grid,poiss,:])[sort]
            rate_grid_corr[grid, poiss, :] = grid_rate_corr
            rate_gra_corr[grid, poiss, :] = gra_rate_corr
            phase_grid_corr[grid, poiss, :] = grid_phase_corr
            phase_gra_corr[grid, poiss, :] = gra_phase_corr
            complex_grid_corr[grid, poiss, :] = grid_complex_corr
            complex_gra_corr[grid, poiss, :] = gra_complex_corr
            t = diff[sort]
            
            if plot=='yes':
                im2 = ax_phase.scatter(phase_grid_corr[grid, poiss, :], phase_gra_corr[grid, poiss, :], c=t, s=30, cmap=cmap)
                im3 = ax_complex.scatter(complex_grid_corr[grid, poiss, :], complex_gra_corr[grid, poiss, :], c=t, s=30, cmap=cmap)
                im1 = ax_rate.scatter(rate_grid_corr[grid, poiss, :], rate_gra_corr[grid, poiss, :], c=t, s=30, cmap=cmap)
                # for one bin
                # ax_rate.set_title('Rate')
                ax_rate.set_xlabel('$R_{in}$')
                ax_rate.set_ylabel('$R_{out}$')
                ax_rate.set_aspect('equal')
                ax_rate.set_xlim(-0.15,1)
                ax_rate.set_ylim(-0.10,1)
                ax_rate.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'g-', linewidth=1)
                # ax_rate.tick_params(labelbottom= False)
                # ax_phase.set_title('Phase')
                ax_phase.set_xlabel('$R_{in}$')
                ax_phase.set_aspect('equal')
                ax_phase.set_xlim(-0.15,1)
                ax_phase.set_ylim(-0.10,1)
                ax_phase.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'g-', linewidth=1)
                # ax_phase.tick_params(labelbottom= False)
                # ax_complex.set_title('Polar')
                ax_complex.set_xlabel('$R_{in}$')
                ax_complex.set_aspect('equal')
                ax_complex.set_xlim(-0.15,1)
                ax_complex.set_ylim(-0.10,1)
                ax_complex.plot(np.arange(0,1.1,0.1),np.arange(0,1.1,0.1),'g-', linewidth=1)
                # ax_complex.tick_params(labelbottom= False)

                
                fig.subplots_adjust(right=0.8)
                cax = fig.add_axes([0.83,0.30,0.01,0.40])
                cbar = fig.colorbar(im3, cax=cax)
                cbar.set_label('distance (cm)', labelpad=20, rotation=270)
                save_dir = '/home/baris/figures/'
                fig.savefig(save_dir+'Rin_Rout_codes_'+fname+'.eps', dpi=200)
                fig.savefig(save_dir+'Rin_Rout_codes_'+fname+'.png', dpi=200)
            else:
                pass
            
    return rate_grid_corr, phase_grid_corr, complex_grid_corr, rate_gra_corr, phase_gra_corr, complex_gra_corr




'TEST'



# # def test_signals():
# first = np.empty((20, 8000))
# second = np.empty((20, 8000))
# for i in range(20):
#     np.random.seed(seed=100)
#     signal = np.random.rand(8000)
#     noise_seed = np.random.randint(8000)
#     np.random.seed(seed=noise_seed+i)
#     noise = np.random.rand(8000)*5
    
#     first[i, :] = signal/2 + noise
    
#     np.random.seed(seed=101)
#     signal_2 = np.random.rand(8000)
#     noise_seed_2 = np.random.randint(8000)
#     np.random.seed(seed=noise_seed_2+10+i)
#     noise_2 = np.random.rand(8000)*5
    
#     second[i, :] = signal_2/2 + noise_2
#     # return first, second

# # first, second = test_signals()

# def leave_one_out_test(first, second, lr=1e-4, n_iter=10000, grid_seeds = np.array([grid_s]), perc_seed=420):
#     _75 = first
#     _65 = second
#     trajs = ' 2 Test Signals'
#     code_name = ' Random noise in each sample'
#     inp_len = first.shape[1]
#     plt.close('all')
#     fig, ax = plt.subplots()
#     whole = np.vstack((_75, _65))
#     n_sample = int(whole.shape[0])
#     for i in range(n_sample):
#         train = np.vstack((whole[:i,:], whole[i+1:,:]))
#         print(train.shape)
#         test = whole[i,:]
#         if i < ((n_sample/2)-1):
#             label_test = torch.FloatTensor([1,0]) 
#         else:
#             label_test = torch.FloatTensor([0,1]) 
#         labels, out_len = label_leave1out(int(n_sample/2), i)
#         print(labels.shape)
#         torch.manual_seed(perc_seed+i)
#         net = Net(inp_len, out_len)
#         print(n_iter)
#         track_loss, test_loss, out = train_net(net, train, test, labels, label_test, n_iter=n_iter, lr=lr)
#         # if i==10:
#         ax.plot(track_loss, 'b-')
#         ax.plot(test_loss, 'k-')
#     ax.set_title(code_name + trajs+'  blue=train & black=test \n n_train='+str(train.shape[0])+' n_test='+str(test.shape[0])+' lr='+str(lr))
#     print(i)
    

# leave_one_out_test(first, second)  








    

#old
# def leave_one_out(code, lr=1e-3, n_iter=50000, grid_seeds = np.array([grid_s]), perc_seed=420):
#     _75 = code[0,:,:]
#     _65 = code [6,:,:]
#     trajs = ' 75 vs 71 '
#     code_name = 'Granule Rate shuffled'
#     inp_len = code.shape[2]
#     plt.close('all')
#     fig, ax = plt.subplots()
#     for i in range(int(_75.shape[0])):
#         train_75 = np.vstack((_75[:i,:], _75[i+1:,:]))
#         train_65 = np.vstack((_65[:i,:], _65[i+1:,:]))
#         train = np.vstack((train_75, train_65))
#         print(train.shape)
#         test = np.vstack((_75[i,:], _65[i,:]))
#         labels, out_len = label(int(train.shape[0]/2))
#         labels_test, out_len_test = label(int(test.shape[0]/2))
#         torch.manual_seed(perc_seed+i)
#         net = Net(inp_len, out_len)
#         print(n_iter)
#         track_loss, test_loss, out = train_net(net, train, test, labels, labels_test, n_iter=n_iter, lr=lr)
#         ax.plot(track_loss, 'b-')
#         ax.plot(test_loss, 'k-')
#     ax.set_title(code_name + trajs+'  blue=train & black=test \n n_train='+str(train.shape[0])+' n_test='+str(test.shape[0])+' lr='+str(lr))
#     print(i)

 
    