#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:31:20 2020

@author: baris
"""

inp_len = n_bin*n_grid
lr = 8e-2

rate_code_sim = np.zeros((sample_size, 2*inp_len, n_sampleset))
rate_code_diff = np.zeros((sample_size, 2*inp_len, n_sampleset))
phase_code_sim = np.zeros((sample_size, 2*inp_len, n_sampleset))
phase_code_diff = np.zeros((sample_size, 2*inp_len, n_sampleset))
complex_code_sim = np.zeros((sample_size, 2*inp_len, n_sampleset))
complex_code_diff = np.zeros((sample_size, 2*inp_len, n_sampleset))
#thresholg crossing points
rate_th_cross_sim = []
rate_th_cross_diff = []
phase_th_cross_sim = []
phase_th_cross_diff = []
complex_th_cross_sim = []
complex_th_cross_diff = []
#RMSE loss in each epoch
rate_rmse_sim = []
rate_rmse_diff = []
phase_rmse_sim = []
phase_rmse_diff = []
complex_rmse_sim = []
complex_rmse_diff = []




# #Input generation
# #Rate trajs with phase info; oscillations implemented in the rate profile
# grid_phases_sim,  gra_phases_sim, grid_spikes_sim, gra_spikes_sim, rate_trajs_sim, dt_s, theta_phase_sim, phase_code_dir_sim, overall_dir_sim = phase_code(sim_traj, dur_ms, grid_seeds[idx], poiss_seeds, pp_weight)
# grid_phases_diff, gra_phases_diff, grid_spikes_diff, gra_spikes_diff, rate_trajs_diff, dt_s, theta_phase_diff, phase_code_dir_diff, overall_dir_diff = phase_code(diff_traj, dur_ms, grid_seeds[idx], poiss_seeds, pp_weight)
# #grid and granule spike counts \ rate codes
# grid_sim_traj_cts, gra_sim_traj_cts = overall_spike_ct(grid_spikes_sim, gra_spikes_sim, dur_ms, poiss_seeds, n_traj=n_traj)
# grid_diff_traj_cts, gra_diff_traj_cts = overall_spike_ct(grid_spikes_diff, gra_spikes_diff, dur_ms, poiss_seeds, n_traj=n_traj)

# #change rate code to mean of non zeros where it is nonzero
# cts_for_phase_sim = copy.deepcopy(grid_sim_traj_cts)
# cts_for_phase_sim[cts_for_phase_sim!=0]=np.mean(cts_for_phase_sim[cts_for_phase_sim!=0]) #was 1
# cts_for_phase_diff = copy.deepcopy(grid_diff_traj_cts)
# cts_for_phase_diff[cts_for_phase_diff!=0]=np.mean(cts_for_phase_diff[cts_for_phase_diff!=0])

# #rate code with constant 45 deg phase
# phase_of_rate_code = np.pi/4
# rate_y_sim = grid_sim_traj_cts*np.sin(phase_of_rate_code)
# rate_x_sim = grid_sim_traj_cts*np.cos(phase_of_rate_code)
# rate_sim =  np.concatenate((rate_y_sim, rate_x_sim), axis=1)
# rate_y_diff = grid_diff_traj_cts*np.sin(phase_of_rate_code)
# rate_x_diff = grid_diff_traj_cts*np.cos(phase_of_rate_code)
# rate_diff =  np.concatenate((rate_y_diff, rate_x_diff), axis=1)

# #phase code with phase and mean rate 
# phase_y_sim = cts_for_phase_sim*np.sin(grid_phases_sim)
# phase_x_sim = cts_for_phase_sim*np.cos(grid_phases_sim)
# phase_sim =  np.concatenate((phase_y_sim, phase_x_sim), axis=1)
# phase_y_diff = cts_for_phase_diff*np.sin(grid_phases_diff)
# phase_x_diff = cts_for_phase_diff*np.cos(grid_phases_diff)
# phase_diff =  np.concatenate((phase_y_diff, phase_x_diff), axis=1)
# #complex code with rate and phase
# complex_sim_y = grid_sim_traj_cts*np.sin(grid_phases_sim)
# complex_sim_x = grid_sim_traj_cts*np.cos(grid_phases_sim)
# complex_sim = np.concatenate((complex_sim_y, complex_sim_x), axis=1)
# complex_diff_y = grid_diff_traj_cts*np.sin(grid_phases_diff)
# complex_diff_x = grid_diff_traj_cts*np.cos(grid_phases_diff)
# complex_diff = np.concatenate((complex_diff_y, complex_diff_x), axis=1)

fig, ax = plt.subplots()
ax.plot(rate_code_sim[1,:,0], alpha=0.3)
ax.plot(phase_code_sim[1,:,0], alpha=0.3)
ax.plot(complex_code_sim[1,:,0], alpha=0.3)
ax.legend(("Rate", "Phase", "Complex"))

fig, ax = plt.subplots()
ax.plot(rate_trajs_sim[0,:,0], alpha=0.3)
ax.plot(complex_code_sim[1,:,0], alpha=0.3)
ax.legend(("Rate", "Phase", "Complex"))

np.mean(rate_trajs_sim[0,:,0])
np.mean(overall_dir_sim[0,:,0])/np.mean(rate_trajs_sim[0,:,0])

np.mean(overall_dir_sim[:,:,1])/np.mean(rate_trajs_sim[:,:,1])

#fill arrays to save the data
rate_code_sim[:,:,idx] = rate_sim
rate_code_diff[:,:,idx] = rate_diff
phase_code_sim[:,:,idx] = phase_sim
phase_code_diff[:,:,idx] = phase_diff
complex_code_sim[:,:,idx] = complex_sim
complex_code_diff[:,:,idx] = complex_diff

print('data done!')

#Into tensor
rate_sim = torch.FloatTensor(rate_sim)
rate_diff = torch.FloatTensor(rate_diff)
phase_sim = torch.FloatTensor(phase_sim)
phase_diff = torch.FloatTensor(phase_diff)
complex_sim = torch.FloatTensor(complex_sim)
complex_diff = torch.FloatTensor(complex_diff)

#initate the network with diff types of inputs and plot the change in loss
#rate code
torch.manual_seed(perc_seed)
net_rate_sim = Net(inp_len*2, out_len)
rate_train_loss_sim, rate_out_sim = train_net(net_rate_sim, rate_sim, labels, n_iter=n_iter, lr=lr)
rate_rmse_sim.append(rate_train_loss_sim)
rate_th_cross_sim.append(np.argmax(np.array(rate_train_loss_sim) < 0.2))
if perc_seed == perc_seeds[0]:
    ax1.plot(rate_train_loss_sim, 'b-', label=str(sim_traj[0])+'cm vs '+str(sim_traj[1])+'cm')
else:
    ax1.plot(rate_train_loss_sim, 'b-')
        
torch.manual_seed(perc_seed)
net_rate_diff = Net(inp_len*2, out_len)
rate_train_loss_diff, rate_out_diff = train_net(net_rate_diff, rate_diff, labels, n_iter=n_iter, lr=lr)
rate_rmse_diff.append(rate_train_loss_diff)
rate_th_cross_diff.append(np.argmax(np.array(rate_train_loss_diff) < 0.2))

#phase code        
torch.manual_seed(perc_seed)
net_phase_sim = Net(inp_len*2, out_len)
phase_train_loss_sim, out_sim = train_net(net_phase_sim, phase_sim, labels, n_iter=n_iter, lr=lr)
phase_rmse_sim.append(phase_train_loss_sim)
phase_th_cross_sim.append(np.argmax(np.array(phase_train_loss_sim) < 0.2))

torch.manual_seed(perc_seed)
net_phase_diff = Net(inp_len*2, out_len)
phase_train_loss_diff, out_diff = train_net(net_phase_diff, phase_diff, labels, n_iter=n_iter, lr=lr)
phase_rmse_diff.append(phase_train_loss_diff)
phase_th_cross_diff.append(np.argmax(np.array(phase_train_loss_diff) < 0.2))

#complex code
torch.manual_seed(perc_seed)
net_complex_sim = Net(inp_len*2, out_len)
complex_train_loss_sim, complex_out_sim = train_net(net_complex_sim, complex_sim, labels, n_iter=n_iter, lr=lr)
complex_rmse_sim.append(complex_train_loss_sim)
complex_th_cross_sim.append(np.argmax(np.array(complex_train_loss_sim) < 0.2))

torch.manual_seed(perc_seed)
net_complex_diff = Net(inp_len*2, out_len)
complex_train_loss_diff, complex_out_diff = train_net(net_complex_diff, complex_diff, labels, n_iter=n_iter, lr=lr)
complex_rmse_diff.append(complex_train_loss_diff)
complex_th_cross_diff.append(np.argmax(np.array(complex_train_loss_diff) < 0.2))

save_dir = '/home/baris/repo/perceptron_results/'
fname = 'grid_rate_n_phase_perceptron_net-seed'+str(perc_seeds)+'_'+str(dur_ms)+'ms'

np.savez(save_dir+fname, 
 rate_code_sim = rate_code_sim,
 rate_code_diff = rate_code_diff,
 phase_code_sim = phase_code_sim,
 phase_code_diff = phase_code_diff,
 complex_code_sim = complex_code_sim,
 complex_code_diff = complex_code_diff,
 
 grid_spikes_sim = grid_spikes_sim,
 grid_spikes_diff = grid_spikes_diff,
 gra_spikes_sim = gra_spikes_sim,
 gra_spikes_diff = gra_spikes_diff,
 
 rate_rmse_sim = rate_rmse_sim,
 rate_rmse_diff = rate_rmse_diff,
 phase_rmse_sim = phase_rmse_sim,
 phase_rmse_diff = phase_rmse_diff,
 complex_rmse_sim = complex_rmse_sim,
 complex_rmse_diff = complex_rmse_diff,
 
 rate_th_cross_sim=rate_th_cross_sim, 
 rate_th_cross_diff=rate_th_cross_diff,
 phase_th_cross_sim=phase_th_cross_sim, 
 phase_th_cross_diff=phase_th_cross_diff,
 complex_th_cross_diff=complex_th_cross_diff, 
 complex_th_cross_sim=complex_th_cross_sim,

 n_grid = n_grid, 
 max_rate = max_rate,
 dur_ms = dur_ms,
 bin_size = bin_size,
 n_bin = n_bin,
 dur_s = dur_s,
 speed_cm = speed_cm,
 field_size_cm = field_size_cm,
 traj_size_cm = traj_size_cm,
 inp_len = inp_len,
 lr = lr,
 n_iter = n_iter,
 sample_size = sample_size,
 n_sampleset = n_sampleset,
 labels = labels,
 sim_traj = sim_traj,
 diff_traj = diff_traj,
 grid_seeds = grid_seeds,
 poiss_seeds = poiss_seeds,
 perc_seeds = perc_seeds)