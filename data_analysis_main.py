#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 14:19:20 2021

@author: baris
"""

from neural_coding import load_spikes, rate_n_phase
from perceptron import run_perceptron
import numpy as np
import matplotlib.pyplot as plt
import shelve

trajectories = [75, 74.5, 74, 73.5, 73, 72.5, 72, 71, 70, 69, 68, 67, 66, 65, 60]
n_samples = 20
grid_seed = 1

lr = 1e-3
n_iter = 10000


# non-shuffled
path = "/home/baris/results/collective/grid-seed_duration_shuffling_tuning_"+str(grid_seed)+"_2000_non-shuffled_tuned"


grid_spikes = load_spikes(path, "grid", trajectories, n_samples)
granule_spikes = load_spikes(path, "granule", trajectories, n_samples)

(
    grid_counts,
    grid_phases,
    grid_rate_code,
    grid_phase_code,
    grid_polar_code,
) = rate_n_phase(grid_spikes, trajectories, n_samples)

(
    granule_counts,
    granule_phases,
    granule_rate_code,
    granule_phase_code,
    granule_polar_code,
) = rate_n_phase(granule_spikes, trajectories, n_samples)




# shuffled
s_path = "/home/baris/results/collective/grid-seed_duration_shuffling_tuning_"+str(grid_seed)+"_2000_shuffled_tuned"

s_grid_spikes = load_spikes(s_path, "grid", trajectories, n_samples)
s_granule_spikes = load_spikes(s_path, "granule", trajectories, n_samples)

(
    s_grid_counts,
    s_grid_phases,
    s_grid_rate_code,
    s_grid_phase_code,
    s_grid_polar_code,
) = rate_n_phase(s_grid_spikes, trajectories, n_samples)

(
    s_granule_counts,
    s_granule_phases,
    s_granule_rate_code,
    s_granule_phase_code,
    s_granule_polar_code,
) = rate_n_phase(s_granule_spikes, trajectories, n_samples)



all_codes = {'shuffled':{},
             'non-shuffled': {}}
all_codes['shuffled'] = {'grid': {},
                         'granule': {}}
all_codes['non-shuffled'] = {'grid': {},
                         'granule': {}}
all_codes['non-shuffled']['grid'] = {'rate': grid_rate_code,
                 'phase': grid_phase_code}
all_codes['shuffled']['grid'] = {'rate': s_grid_rate_code,
                 'phase': s_grid_phase_code}
all_codes['non-shuffled']['granule'] = {'rate': granule_rate_code,
                 'phase': granule_phase_code}
all_codes['shuffled']['granule'] = {'rate': s_granule_rate_code,
                 'phase': s_granule_phase_code}


# all_codes['non-shuffled']['grid'] = {'rate': grid_rate_code,
#                  'phase': grid_phase_code,
#                  'polar': grid_polar_code}
# all_codes['shuffled']['grid'] = {'rate': s_grid_rate_code,
#                  'phase': s_grid_phase_code,
#                  'polar': s_grid_polar_code}
# all_codes['non-shuffled']['granule'] = {'rate': granule_rate_code,
#                  'phase': granule_phase_code,
#                  'polar': granule_polar_code}
# all_codes['shuffled']['granule'] = {'rate': s_granule_rate_code,
#                  'phase': s_granule_phase_code,
#                  'polar': s_granule_polar_code}




def thresholds(code, trajectories, lr=1e-3, n_iter=2000):
    ths = []
    losses = []
    s_ths = []
    s_losses = []
    for traj_idx, traj in enumerate(trajectories):
        perceptron_input = np.hstack((code[:,:,0], code[:,:,traj_idx+1]))
        th_cross, train_loss = run_perceptron(perceptron_input, grid_seed, lr=lr, n_iter=n_iter)
        ths.append(th_cross)
        losses.append(train_loss)
        
        if traj_idx == (len(trajectories)-2):
            break
    return ths, losses
   



threshold = {}    
for shuffling in all_codes:
    threshold[shuffling] = {}
    for cell in all_codes[shuffling]:
        threshold[shuffling][cell] = {}
        for code in all_codes[shuffling][cell]:
            print(shuffling)
            print(cell)
            print(code)
            if code == 'phase':
                learning = 1e-3
            else:
                learning = lr
            ths, losses = thresholds(all_codes[shuffling][cell][code], trajectories, lr=learning, n_iter=n_iter)
            threshold[shuffling][cell][code] = ths
    





fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, sharex='col') #sharey='row',

fig1.suptitle('Perceptron learning speed - '+str(n_samples)+
              ' samples \n grid seed= '+str(grid_seed)+
              ', rate lr = '+format(lr, '.1E') + ', phase lr = '+format(learning, '.1E'))
xticks = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 15]
ax1.plot(xticks, 1/np.array(threshold['shuffled']['grid']['rate']), 'k-', label='shuffled')
ax1.plot(xticks, 1/np.array(threshold['non-shuffled']['grid']['rate']), 'b--', label='non-shuffled')
ax1.legend()
ax1.set_title('grid rate code')
ax1.set_ylabel('Speed (1/N)')

ax2.plot(xticks, 1/np.array(threshold['shuffled']['grid']['phase']), 'k-', label='shuffled')
ax2.plot(xticks, 1/np.array(threshold['non-shuffled']['grid']['phase']), 'b--', label='non-shuffled')
ax2.legend()
ax2.set_title('grid phase code')



ax3.plot(xticks, 1/np.array(threshold['shuffled']['granule']['rate']), 'k-', label='shuffled')
ax3.plot(xticks, 1/np.array(threshold['non-shuffled']['granule']['rate']), 'b--', label='non-shuffled')
ax3.legend()
ax3.set_xlabel('distance (cm)')
ax3.set_title('granule rate code')
ax3.set_ylabel('Speed (1/N)')


ax4.plot(xticks, 1/np.array(threshold['shuffled']['granule']['phase']), 'k-', label='shuffled')
ax4.plot(xticks, 1/np.array(threshold['non-shuffled']['granule']['phase']), 'b--', label='non-shuffled')
ax4.legend()
ax4.set_title('granule phase code')
ax4.set_xlabel('distance (cm)')
plt.tight_layout()


# fig1, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, sharey='row', sharex='col')

# fig1.suptitle('Perceptron learning speed - '+str(n_samples)+' samples \n grid seed= '+str(grid_seed)+', lr = '+format(lr, '.1E'))
# xticks = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 15]
# ax1.plot(xticks, 1/np.array(threshold['shuffled']['grid']['rate']), 'k-', label='shuffled')
# ax1.plot(xticks, 1/np.array(threshold['non-shuffled']['grid']['rate']), 'b--', label='non-shuffled')
# ax1.legend()
# ax1.set_title('grid rate code')
# ax1.set_ylabel('Speed (1/N)')

# ax2.plot(xticks, 1/np.array(threshold['shuffled']['grid']['phase']), 'k-', label='shuffled')
# ax2.plot(xticks, 1/np.array(threshold['non-shuffled']['grid']['phase']), 'b--', label='non-shuffled')
# ax2.legend()
# ax2.set_title('grid phase code')



# ax3.plot(xticks, 1/np.array(threshold['shuffled']['granule']['rate']), 'k-', label='shuffled')
# ax3.plot(xticks, 1/np.array(threshold['non-shuffled']['granule']['rate']), 'b--', label='non-shuffled')
# ax3.legend()
# ax3.set_xlabel('distance (cm)')
# ax3.set_title('granule rate code')
# ax3.set_ylabel('Speed (1/N)')


# ax4.plot(xticks, 1/np.array(threshold['shuffled']['granule']['phase']), 'k-', label='shuffled')
# ax4.plot(xticks, 1/np.array(threshold['non-shuffled']['granule']['phase']), 'b--', label='non-shuffled')
# ax4.legend()
# ax4.set_title('granule phase code')
# ax4.set_xlabel('distance (cm)')



# ax5.plot(xticks, 1/np.array(threshold['shuffled']['grid']['polar']), 'k-', label='shuffled')
# ax5.plot(xticks, 1/np.array(threshold['non-shuffled']['grid']['polar']), 'b--', label='non-shuffled')
# ax5.legend()
# ax5.set_xlabel('distance (cm)')
# ax5.set_title('granule rate code')
# ax5.set_ylabel('Speed (1/N)')


# ax6.plot(xticks, 1/np.array(threshold['shuffled']['granule']['polar']), 'k-', label='shuffled')
# ax6.plot(xticks, 1/np.array(threshold['non-shuffled']['granule']['polar']), 'b--', label='non-shuffled')
# ax6.legend()
# ax6.set_title('granule phase code')
# ax6.set_xlabel('distance (cm)')


plt.tight_layout()

# save_dir = '/home/baris/figures/'
# fig1.savefig(save_dir+'perceptron_speed_'+str(n_samples)+'_samples_grid_seed_'+str(grid_s)+'.png', dpi=200)



# for traj_idx, traj in enumerate(trajectories):
#     perceptron_input = np.hstack((granule_rate_code[:,:,0], granule_rate_code[:,:,traj_idx+1]))
#     th_cross, train_loss = run_perceptron(perceptron_input, grid_seed, lr=1e-3, n_iter=2000)
#     ths.append(th_cross)
#     losses.append(train_loss)
    
#     s_perceptron_input = np.hstack((s_granule_rate_code[:,:,0], s_granule_rate_code[:,:,traj_idx+1]))
#     s_th_cross, s_train_loss = run_perceptron(s_perceptron_input, grid_seed, lr=1e-3, n_iter=2000)
#     s_ths.append(s_th_cross)
#     s_losses.append(s_train_loss)
    
#     grid_perceptron_input = np.hstack((grid_rate_code[:,:,0], grid_rate_code[:,:,traj_idx+1]))
#     grid_th_cross, train_loss = run_perceptron(grid_perceptron_input, grid_seed, lr=1e-3, n_iter=2000)
#     grid_ths.append(th_cross)
#     grid_losses.append(train_loss)
    
#     s_grid_perceptron_input = np.hstack((s_grid_rate_code[:,:,0], s_grid_rate_code[:,:,traj_idx+1]))
#     s_grid_th_cross, s_train_loss = run_perceptron(s_grid_perceptron_input, grid_seed, lr=1e-3, n_iter=2000)
#     s_grid_ths.append(s_th_cross)
#     s_grid_losses.append(s_train_loss)    
    
    
    
    
    
#     p_perceptron_input = np.hstack((granule_phase_code[:,:,0], granule_phase_code[:,:,traj_idx+1]))
#     p_th_cross, p_train_loss = run_perceptron(p_perceptron_input, grid_seed, lr=1e-3, n_iter=5000)
#     ths.append(p_th_cross)
#     losses.append(p_train_loss)
    
#     p_s_perceptron_input = np.hstack((s_granule_phase_code[:,:,0], s_granule_phase_code[:,:,traj_idx+1]))
#     p_s_th_cross, p_s_train_loss = run_perceptron(p_s_perceptron_input, grid_seed, lr=1e-3, n_iter=5000)
#     p_s_ths.append(p_s_th_cross)
#     p_s_losses.append(p_s_train_loss)

#     p_grid_perceptron_input = np.hstack((grid_phase_code[:,:,0], grid_phase_code[:,:,traj_idx+1]))
#     p_grid_th_cross, train_loss = run_perceptron(p_grid_perceptron_input, grid_seed, lr=1e-3, n_iter=2000)
#     p_grid_ths.append(th_cross)
#     p_grid_losses.append(train_loss)
    
#     p_s_grid_perceptron_input = np.hstack((s_grid_phase_code[:,:,0], s_grid_phase_code[:,:,traj_idx+1]))
#     p_s_grid_th_cross, s_train_loss = run_perceptron(s_perceptron_input, grid_seed, lr=1e-3, n_iter=2000)
#     p_s_grid_ths.append(s_th_cross)
#     p_s_grid_losses.append(s_train_loss)    

# stacked = np.hstack((granule_rate_code[:,:,0], granule_rate_code[:,:,1]))




# transposed = np.transpose(perceptron_input, (1,0))





# spikes.keys()

# type(spikes[75])

# len(grid_spikes[75][0])

# len(spikes[70])


# granule_75 = granule_spikes[75][19]

# grid_75 = grid_spikes[75][19]

# import matplotlib.pyplot as plt

# plt.figure()
# plt.eventplot(granule_75)
# plt.figure()
# plt.eventplot(grid_75)
