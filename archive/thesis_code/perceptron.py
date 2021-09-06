#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:50:35 2021

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
import matplotlib
from matplotlib.ticker import FormatStrFormatter, NullFormatter

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

def train_net(net, train_data, labels, n_iter=1000, lr=1e-4):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    track_loss = []
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    for i in range(n_iter):
        out = net(train_data)
        loss = torch.sqrt(loss_fn(out, labels))
        # Compute gradients
        optimizer.zero_grad()
        loss.backward()
    
        # Update weights
        optimizer.step()
    
        # Store current value of loss
        track_loss.append(loss.item())  # .item() needed to transform the tensor output of loss_fn to a scalar
        
        # Track progress
        if (i + 1) % (n_iter // 5) == 0:
          print(f'iteration {i + 1}/{n_iter} | loss: {loss.item():.3f}')

    return track_loss, out


def all_ths(path1, n_grid_seed, n_traj):
    # path1 = '/home/baris/results/perceptron_th_n_codes/2000_th/'
    npzfiles = []
    all_th = np.zeros((6,n_grid_seed,n_traj))
    i = 1
    for file in sorted(glob.glob(os.path.join(path1,'*.npz'))):
        npzfiles.append(file)
        load = np.load(file, allow_pickle=True)
        all_th[:,:,(2*(i-1)):(2*i)] = load['stacked_th']
        i+=1
    return all_th



def perceptron(code, c, lr=5e-4, n_iter=10000, th=0.2):
    
    n_traj, n_grid_seed, n_poiss, inp_len = code.shape
    
    n_grid_seed = 20
    
    grid_seeds = np.arange(510,530,1)
    perc_seeds = grid_seeds-100

    #threshold crossing points
    th_cross = np.zeros((n_grid_seed,2))
    labels, out_len = label(n_poiss)
    #Into tensor
    code_sim = torch.FloatTensor(np.concatenate((code[0], code[c]), axis=1))
    code_diff = torch.FloatTensor(np.concatenate((code[0], code[c+1]), axis=1))
    
    for i in range(n_grid_seed):
        perc_seed = perc_seeds[i]
        torch.manual_seed(perc_seed)
        net_sim = Net(inp_len, out_len)
        train_loss_sim, out_sim = train_net(net_sim, code_sim[i], labels, n_iter=n_iter, lr=lr)
        torch.manual_seed(perc_seed)
        net_diff = Net(inp_len, out_len)
        train_loss_diff, out_diff = train_net(net_diff, code_diff[i], labels, n_iter=n_iter, lr=lr)
        th_cross[i,0] = np.argmax(np.array(train_loss_sim) < th)
        th_cross[i,1] = np.argmax(np.array(train_loss_diff) < th)
    return th_cross


   


def shuffle_perceptron(shuffled_code, nonshuffled_code, lr=5e-5, n_iter=5000, th=0.2, grid_seeds = np.array([520])):
    
    n_sample, inp_len = shuffled_code.shape
    n_poiss = int(n_sample/2)
    perc_seeds = grid_seeds-100
    n_grid_seed = grid_seeds.shape[0]
    #threshold crossing points
    shuffled_th_cross = np.zeros(n_grid_seed)
    nonshuffled_th_cross = np.zeros(n_grid_seed)
    labels, out_len = label(n_poiss)
    #Into tensor
    shuffled_code = torch.FloatTensor(shuffled_code)
    nonshuffled_code = torch.FloatTensor(nonshuffled_code)
    
    for i in range(n_grid_seed):
        perc_seed = perc_seeds[i]
        torch.manual_seed(perc_seed)
        net_shuffled = Net(inp_len, out_len)
        train_loss_shuffled, out_shuffled = train_net(net_shuffled, shuffled_code, labels, n_iter=n_iter, lr=lr)
        torch.manual_seed(perc_seed)
        net_nonshuffled = Net(inp_len, out_len)
        train_loss_nonshuffled, out_nonshuffled = train_net(net_nonshuffled, nonshuffled_code, labels, n_iter=n_iter, lr=lr)
        shuffled_th_cross[i] = np.argmax(np.array(train_loss_shuffled) < th)
        nonshuffled_th_cross[i] = np.argmax(np.array(train_loss_nonshuffled) < th)
    return shuffled_th_cross, nonshuffled_th_cross

#Parameters for perceptron


path1 = '/home/baris/results/grid_mixed_input/diff_poiss/full/75-74.5-74-73.5/'

'nonshuffled_full_diff_poiss_420-439_traj_75.0-74.5-74.0-73.5_net-seed421_2000ms.npz'



#nonshuffled 

grid_rate_ns = np.empty((8,20,8000))
gra_rate_ns = np.empty((8,20,80000))
grid_phase_ns = np.empty((8,20,8000))
gra_phase_ns = np.empty((8,20,80000))
npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*non*75.0*421*.npz'))):
    npzfiles.append(file)
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,0]
    
    grid_rate_ns[0,5*ct:5+5*ct,:] = t75_grid
    grid_rate_ns[1,5*ct:5+5*ct,:] = t745_grid
    grid_rate_ns[2,5*ct:5+5*ct,:] = t74_grid
    grid_rate_ns[3,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,0]
    
    gra_rate_ns[0,5*ct:5+5*ct,:] = t75_gra
    gra_rate_ns[1,5*ct:5+5*ct,:] = t745_gra
    gra_rate_ns[2,5*ct:5+5*ct,:] = t74_gra
    gra_rate_ns[3,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,0]
    
    grid_phase_ns[0,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_ns[1,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_ns[2,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_ns[3,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,0]

    gra_phase_ns[0,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_ns[1,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_ns[2,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_ns[3,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    
   
#shuffled 

grid_rate_s = np.empty((8,20,8000))
gra_rate_s = np.empty((8,20,80000))
grid_phase_s = np.empty((8,20,8000))
gra_phase_s = np.empty((8,20,80000))
npzfiles = []
ct=0

for file in sorted(glob.glob(os.path.join(path1,'*shuffled*75.0*421*.npz'))):
    if not 'non' in file:
        npzfiles.append(file)
    else:
        continue
    load = np.load(file, allow_pickle=True)
    t75_grid = load['grid_rate_code'][0:5,:,0]
    t745_grid = load['grid_rate_code'][5:10,:,0]
    t74_grid = load['grid_rate_code'][0:5,:,1]
    t735_grid = load['grid_rate_code'][5:10,:,0]
    
    grid_rate_s[0,5*ct:5+5*ct,:] = t75_grid
    grid_rate_s[1,5*ct:5+5*ct,:] = t745_grid
    grid_rate_s[2,5*ct:5+5*ct,:] = t74_grid
    grid_rate_s[3,5*ct:5+5*ct,:] = t735_grid
    
    t75_gra = load['gra_rate_code'][0:5,:,0]
    t745_gra = load['gra_rate_code'][5:10,:,0]
    t74_gra = load['gra_rate_code'][0:5,:,1]
    t735_gra = load['gra_rate_code'][5:10,:,0]
    
    gra_rate_s[0,5*ct:5+5*ct,:] = t75_gra
    gra_rate_s[1,5*ct:5+5*ct,:] = t745_gra
    gra_rate_s[2,5*ct:5+5*ct,:] = t74_gra
    gra_rate_s[3,5*ct:5+5*ct,:] = t735_gra


    t75_grid_phase = load['grid_phase_code'][0:5,:,0]
    t745_grid_phase = load['grid_phase_code'][5:10,:,0]
    t74_grid_phase = load['grid_phase_code'][0:5,:,1]
    t735_grid_phase = load['grid_phase_code'][5:10,:,0]
    
    grid_phase_s[0,5*ct:5+5*ct,:] = t75_grid_phase
    grid_phase_s[1,5*ct:5+5*ct,:] = t745_grid_phase
    grid_phase_s[2,5*ct:5+5*ct,:] = t74_grid_phase
    grid_phase_s[3,5*ct:5+5*ct,:] = t735_grid_phase    
    
    t75_gra_phase  = load['gra_phase_code'][0:5,:,0]
    t745_gra_phase = load['gra_phase_code'][5:10,:,0]
    t74_gra_phase = load['gra_phase_code'][0:5,:,1]
    t735_gra_phase = load['gra_phase_code'][5:10,:,0]

    gra_phase_s[0,5*ct:5+5*ct,:] = t75_gra_phase
    gra_phase_s[1,5*ct:5+5*ct,:] = t745_gra_phase
    gra_phase_s[2,5*ct:5+5*ct,:] = t74_gra_phase
    gra_phase_s[3,5*ct:5+5*ct,:] = t735_gra_phase
   
    ct+=1
    
    
trj1 = 0
trj2 = 1   
    
shuffled_grid_input = np.vstack((grid_rate_s[trj1,:,:], grid_rate_s[trj2,:,:]))
nonshuffled_grid_input = np.vstack((grid_rate_ns[trj1,:,:], grid_rate_ns[trj2,:,:]))    
    
shuffled_gra_input = np.vstack((gra_rate_s[trj1,:,:], gra_rate_s[trj2,:,:]))
nonshuffled_gra_input = np.vstack((gra_rate_ns[trj1,:,:], gra_rate_ns[trj2,:,:]))

grid_s_th, grid_ns_th = shuffle_perceptron(shuffled_grid_input, nonshuffled_grid_input)
gra_s_th, gra_ns_th = shuffle_perceptron(shuffled_gra_input, nonshuffled_gra_input)







def time_cut(code, dur):
    def_dur = 2000
    def_n_bin =  20
    bin_size = 100
    arr_len = code.shape[3]
    n_bin = int(dur/bin_size)
    n_cell = int(arr_len/(def_n_bin*2))
    cut_len = int(def_dur/dur)
    half_len = int(arr_len/2)
    cut_code = np.concatenate((code[:,:,:,:(n_bin*n_cell)], code[:,:,:,(half_len):(half_len+n_bin*n_cell)]), axis=3)
    return cut_code

grid_rate, grid_phase, grid_complex, gra_rate, gra_phase, gra_complex = all_codes()
dur = 200
grid_rate = time_cut(grid_rate, dur)
grid_phase = time_cut(grid_phase, dur)
grid_complex = time_cut(grid_complex, dur)
gra_rate = time_cut(gra_rate, dur)
gra_phase = time_cut(gra_phase, dur)
gra_complex = time_cut(gra_complex, dur)

n_iter = 10000
th = 0.2
lr = 5e-4
traj = 10
th_grid_rate = perceptron(grid_rate, c=traj, lr=lr, n_iter=n_iter)
th_grid_phase = perceptron(grid_phase, c=traj, lr=lr, n_iter=n_iter)
th_grid_complex = perceptron(grid_complex, c=traj, lr=lr, n_iter=n_iter)
th_gra_rate = perceptron(gra_rate, c=traj, lr=lr, n_iter=n_iter)
th_gra_phase = perceptron(gra_phase, c=traj, lr=lr, n_iter=n_iter)
th_gra_complex = perceptron(gra_complex, c=traj, lr=lr, n_iter=n_iter)


#for plotting single data points
th_norm_r = (th_gra_rate/th_grid_rate)
th_norm_p = (th_gra_phase/th_grid_phase)
th_norm_c = (th_gra_complex/th_grid_complex)
speed_r = 1/th_norm_r
speed_p = 1/th_norm_p
speed_c = 1/th_norm_c
#flattened for plotting th barplot 
th_norm_rate = (th_gra_rate/th_grid_rate).flatten('F')
th_norm_phase = (th_gra_phase/th_grid_phase).flatten('F')
th_norm_complex = (th_gra_complex/th_grid_complex).flatten('F')
speed_norm_rate = 1/th_norm_rate
speed_norm_phase = 1/th_norm_phase
speed_norm_complex = 1/th_norm_complex


ths = np.concatenate((th_norm_rate,th_norm_phase,th_norm_complex))
speeds = np.concatenate((speed_norm_rate,speed_norm_phase,speed_norm_complex))



n_grid_seed = 20
trajectories = np.array([75, 74.5, 74, 73.5, 73, 72.5, 72, 71.5, 71, 70, 65, 60])
n_traj = trajectories.shape[0]
distance = 75-trajectories
#BARPLOT encoding speed all trajectories
stacked_th = np.stack((th_grid_rate, th_grid_phase, th_grid_complex, th_gra_rate, th_gra_phase, th_gra_complex))
save_dir = '/home/baris/results/perceptron_th_n_codes/20_grid_seeds/lr_5-4/200_th/'
fname = str(n_grid_seed)+'_seed_lr_'+str(lr)+'_'+str(dur)+'_th_6'
np.savez(save_dir+fname, stacked_th=stacked_th)


ths_2000 = all_ths('/home/baris/results/perceptron_th_n_codes/20_grid_seeds/lr_5-4/2000_th/', n_grid_seed, n_traj)
# ths_1000 = all_ths('/home/baris/results/perceptron_th_n_codes/20_grid_seeds/1000_th/')
# ths_500 = all_ths('/home/baris/results/perceptron_th_n_codes/500_th/')
ths_200 = all_ths('/home/baris/results/perceptron_th_n_codes/20_grid_seeds/lr_5-4/200_th/', n_grid_seed, n_traj)

# ths = ths_2000

data = np.zeros((122, 40, 30))

writer = pd.ExcelWriter('thresholds_200ms.xlsx', engine='xlsxwriter')

codes = ['grid_rate', 'grid_phase', 'grid_polar', 'gra_rate', 'gra_phase', 'gra_polar']

for i in range(6):
    code = codes[i]
    df = pd.DataFrame(ths_200[i,:,:])
    df.columns = ['75-75','75-74.5', '75-74', '75-73.5', '75-73', '75-72.5', '75-72',
                  '75-71.5', '75-71', '75-70', '75-65', '75-60']
    df.to_excel(writer, sheet_name=code)

writer.save()

def ext_ths(ths_time, cell):
    ths = ths_time
    th_grid_r = ths[0,:,1:]
    th_grid_p = ths[1,:,1:]
    th_grid_c = ths[2,:,1:]
    th_gra_r = ths[3,:,1:]
    th_gra_p = ths[4,:,1:]
    th_gra_c = ths[5,:,1:]
    th_norm_r = (th_gra_r/th_grid_r)
    th_norm_p = (th_gra_p/th_grid_p)
    th_norm_c = (th_gra_c/th_grid_c)
    if cell == 'grid':
        speed_r = 1/th_grid_r
        speed_p = 1/th_grid_p
        speed_c = 1/th_grid_c
    elif cell == 'granule':
        speed_r = 1/th_gra_r
        speed_p = 1/th_gra_p
        speed_c = 1/th_gra_c
    else:
        speed_r = 1/th_norm_r
        speed_p = 1/th_norm_p
        speed_c = 1/th_norm_c
    mean_r = np.mean(speed_r, axis=0)
    mean_p = np.mean(speed_p, axis=0)
    mean_c = np.mean(speed_c, axis=0)
    # std_r = np.std(speed_r, axis=0)
    # std_p = np.std(speed_p, axis=0)
    # std_c = np.std(speed_c, axis=0)
    sem_r = stats.sem(speed_r, axis=0)
    sem_p = stats.sem(speed_p, axis=0)
    sem_c = stats.sem(speed_c, axis=0)
    return mean_r, mean_p, mean_c, sem_r, sem_p, sem_c
    


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class Labeloffset():
    def __init__(self,  ax, label="", axis="y"):
        self.axis = {"y":ax.yaxis, "x":ax.xaxis}[axis]
        self.label=label
        ax.callbacks.connect(axis+'lim_changed', self.update)
        ax.figure.canvas.draw()
        self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + " "+ fmt.get_offset() )




plt.close('all')

sns.reset_orig()
sns.color_palette('deep')
sns.set(context='paper',style='whitegrid', palette='deep', font='Arial',font_scale=1.5,color_codes=True)
import pylab as plott
params = {'legend.fontsize':15, 'legend.handlelength':1.5}
plott.rcParams.update(params)

fig2, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize=(10,15))


cell = 'n'
dur= 2000
# mean_r, mean_p, mean_c, sem_r, sem_p, sem_c = ext_ths(ths_2000, cell)
# mean_r, mean_p, mean_c, sem_r, sem_p, sem_c = ext_ths(ths_1000)
# mean_r, mean_p, mean_c, sem_r, sem_p, sem_c = ext_ths(ths_500)
mean_r, mean_p, mean_c, sem_r, sem_p, sem_c = ext_ths(ths_200, cell)


# plt.title('Normalized (granule/grid) Perceptron Encoding Speed 2000ms')
if cell =='grid' or cell =='granule':
    cell_name = cell.capitalize()+' Codes in ' +str(dur)+ ' ms'
    speed = 'speed'
else:
    cell_name = 'Normalized (granule/grid) in '+str(dur)+' ms'
    speed = 'speed'
    
ax = ax6
    
ax.set_xscale('log')
# ax.tick_params(axis='x', which='minor')

# ax.xaxis.set_minor_formatter(NullFormatter()) 


x_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
ax.xaxis.set_major_locator(x_major)
x_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
ax.xaxis.set_minor_locator(y_minor)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.grid(True, axis='x', which='minor')
ax.yaxis.offsetText.set_visible(False)

ax.set_title(cell_name)
ax.set_xlabel('distance (cm)')
# ax.set_ylabel(speed)
ax.ticklabel_format(axis = 'y', style='sci', scilimits=(0,0))
ax.set_aspect('auto')
ax.errorbar(distance[1:], mean_r, yerr=sem_r, capsize=3.5, label='rate', linewidth=2)
ax.errorbar(distance[1:], mean_p, yerr=sem_p, capsize=3.5, label='phase', linewidth=2)
ax.errorbar(distance[1:], mean_c, yerr=sem_c, capsize=3.5, label='polar', linewidth=2, linestyle='--')
ax.legend()

formatter = mticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((0,0))
ax.yaxis.set_major_formatter(formatter)

lo = Labeloffset(ax, label=speed, axis="y")



parameters = ('lr = 5*$10^{-4}$,  20 grid seeds, different Poisson seeds, error bars = SEM' + 
          '\n$N$ = number of epochs until RMSE reached a threshold of 0.2 ')
# ax1.annotate(parameters, (0,0), (0, -37), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)
plt.tight_layout(pad=1, h_pad=2)
fig2.set_size_inches(12,14)




save_dir = '/home/baris/figures/'


fig2.savefig(save_dir+'all10_encoding_speed_dur_'+str(dur)+'_ngrid_'+str(n_grid_seed)+'_lr_'+str(lr)+'.eps', dpi=200)
fig2.savefig(save_dir+'all10_encoding_speed_dur_'+str(dur)+'_ngrid_'+str(n_grid_seed)+'_lr_'+str(lr)+'.png', dpi=200)





n_seeds = 10
trajectories = np.array([75, 74.5, 74, 73.5, 73, 72.5, 72, 71.5, 71, 70, 65, 60])
n_traj = trajectories.shape[0]
distance = 75-trajectories
hue = np.tile(np.tile(distance[1:], n_seeds), 3)
hue = (['similar']*n_seeds+ ['distinct']*n_seeds)*3
code = ['rate']*n_seeds*n_traj +['phase']*n_seeds*n_traj+['complex']*n_seeds*n_traj
seeds = np.tile(np.arange(1,11,1), 6)

df = pd.DataFrame({'encoding speed': speeds,
                    'trajectories': pd.Categorical(hue), 
                    'code': pd.Categorical(code),
                    'seeds': pd.Categorical(seeds)})

df['trajectories'] = pd.Categorical(df['trajectories'], categories=['similar','distinct'], ordered=True)

sns.set(context='paper',style='whitegrid', palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)
plt.figure()
sns.lineplot(x='trajectories', y='encoding speed', hue='code', ci="sd", markers="o",err_kws={'capsize':2},
              data=df, err_style="bars", legend="full", sort=False)
plt.title('Normalized Perceptron Encoding Speed')
parameters = ('normalized speed = granule/grid,   lr = 5*$10^{-5}$,  10 grid seeds, different Poisson seeds' + 
              '\n$speed$ = 1/number of epochs until RMSE reached a threshold of 0.2 ')
plt.annotate(parameters, (0,0), (0, -37), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)



#BARPLOT encoding speed 3 trajectories
n_seeds = 10
hue = (['similar']*n_seeds+ ['distinct']*n_seeds)*3
code = ['rate']*n_seeds*2 +['phase']*n_seeds*2+['complex']*n_seeds*2
seeds = np.tile(np.arange(1,11,1), 6)

df = pd.DataFrame({'encoding speed': speeds,
                    'trajectories': pd.Categorical(hue), 
                    'code': pd.Categorical(code),
                    'seeds': pd.Categorical(seeds)})

df['trajectories'] = pd.Categorical(df['trajectories'], categories=['similar','distinct'], ordered=True)

sns.set(context='paper',style='whitegrid', palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)
plt.figure()
sns.lineplot(x='trajectories', y='encoding speed', hue='code', ci="sd", markers="o",err_kws={'capsize':2},
              data=df, err_style="bars", legend="full", sort=False)
plt.title('Normalized Perceptron Encoding Speed')
parameters = ('normalized speed = granule/grid,   lr = 5*$10^{-5}$,  10 grid seeds, different Poisson seeds' + 
              '\n$speed$ = 1/number of epochs until RMSE reached a threshold of 0.2 ')
plt.annotate(parameters, (0,0), (0, -37), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)


# avg_speed = avg_speed.T.reset_index()

# p = avg_speed.plot(x="trajectories", kind='line', yerr=std_speed)

# avg_speed["trajectories"]

# c = avg_speed["complex"].T

# c.plot(x=["similar", "distinct"])

# avg_speed.plot(x = "complex")

# plt.figure()
# plt.plot()


plt.close('all')
ax1 = sns.barplot(x='code', y='encoding speed', hue='trajectories', data=df, order=['rate', 'phase', 'complex'])
plt.title('Normalized encoding speed')
parameters = ('lr = 5*$10^{-4}$,  10 Grid seeds, 5 Poisson seeds,'+ 
              '  $speed$ = 1/number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

width = ax1.patches[0].get_width()

for idx in range(n_seeds):
    loc = width/2
    rate_sim = speed_r[idx,0]
    rate_diff = speed_r[idx,1]
    phase_sim = speed_p[idx,0]
    phase_diff = speed_p[idx,1]
    complex_sim = speed_c[idx,0]
    complex_diff = speed_c[idx,1]
    sns.lineplot(x=[-loc,loc], y=[rate_diff, rate_sim], color='k')
    sns.lineplot(x=[1-loc,1+loc], y=[phase_diff, phase_sim], color='k')
    sns.lineplot(x=[2-loc,2+loc], y=[complex_diff, complex_sim], color='k')

#gra threshold cross
plt.figure()
ax1 = sns.barplot(x='code', y='threshold cross', hue='trajectories', data=gra_df, order=['rate', 'phase', 'complex'])
plt.title('Threshold Crossing for Granule Codes in 2000ms')
parameters = ('lr = 5*$10^{-3}$,  10 Grid seeds, 5 Poisson seeds,'+ 
              '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

width = ax1.patches[0].get_width()
for idx in range(n_seeds):
    loc = width/2
    rate_sim = th_gra[idx,0]
    rate_diff = th_gra[idx,1]
    phase_sim = th_gra[idx,2]
    phase_diff = th_gra[idx,3]
    complex_sim = th_gra[idx,4]
    complex_diff = th_gra[idx,5]
    sns.lineplot(x=[-loc,loc], y=[rate_diff, rate_sim], color='k')
    sns.lineplot(x=[1-loc,1+loc], y=[phase_diff, phase_sim], color='k')
    sns.lineplot(x=[2-loc,2+loc], y=[complex_diff, complex_sim], color='k')
    
    
width = ax1.patches[0].get_width()
for idx in range(n_seeds):
    loc = width/2
    rate_sim = th_gra[idx,0]
    rate_diff = th_gra[idx,1]
    phase_sim = th_gra[idx,2]
    phase_diff = th_gra[idx,3]
    complex_sim = th_gra[idx,4]
    complex_diff = th_gra[idx,5]
    sns.lineplot(x=[-loc,loc], y=[rate_diff, rate_sim], color='r')
    sns.lineplot(x=[1-loc,1+loc], y=[phase_diff, phase_sim], color='b')
    sns.lineplot(x=[2-loc,2+loc], y=[complex_diff, complex_sim], color='g')
    
    

th_grid_rate


a = np.array([[810., 188.],
        [890., 216.],
        [720., 232.],
        [828., 219.],
        [773., 215.],
        [752., 185.],
        [773., 196.],
        [744., 195.],
        [757., 191.],
        [666., 232.]])
Out[266]: 
array([[55., 17.],
        [55., 18.],
        [54., 19.],
        [55., 18.],
        [54., 18.],
        [55., 18.],
        [55., 18.],
        [55., 18.],
        [54., 17.],
        [54., 19.]])
'''

'''results for same poiss rate
array([[1121.,  203.],
        [1149.,  213.],
        [1193.,  224.],
        [1129.,  208.],
        [1122.,  208.],
        [1130.,  209.],
        [1147.,  200.],
        [1121.,  197.],
        [1120.,  198.],
        [1144.,  216.]])'''