# -*- coding: utf-8 -*-
"""
Figure 1 demonstrates the grid cell model with phase precession and the
shuffling feature.
"""

import matplotlib.pyplot as plt
import numpy as np
import grid_model
import matplotlib.gridspec as gridspec

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

spacing = 50
orientation = 30
pos_peak = [20, 20]
arr_size = 200
dur_ms = 2000
dt_s = 0.002

trajectories = [75.0]
simulation_result = grid_model.grid_simulate(
    trajectories,
    dur_ms=dur_ms,
    grid_seed=1,
    poiss_seeds=[1],
    shuffle='non-shuffled',
    n_grid=200,
    speed_cm=20,
    rate_scale=5,
    arr_size=arr_size,
    f=10,
    shift_deg=180,
<<<<<<< HEAD
    dt_s=0.001,
=======
    dt_s=dt_s,
>>>>>>> 5887f8a0c0c1aebf6c85bad742693598fed6cba3
    large_output=True
)

grid_length_cm = 100
grid_axes_ticks = np.arange(0,grid_length_cm,grid_length_cm/arr_size)

trajectories_y = 100 - np.array(trajectories)
trajectories_xstart = 0
trajectories_xend= 40

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(6,2)
ax1 = fig.add_subplot(gs[:3,0])
ax1.imshow(simulation_result[3][:,:,1], extent=[0,100,0,100]) # TODO UNITS
for trajectory in trajectories_y:
    ax1.plot([trajectories_xstart, trajectories_xend], [trajectory]*2,
             linewidth=3)
ax2 = fig.add_subplot(gs[0:2,1])
x = np.arange(0,dur_ms,dt_s*1000)
ax2.plot(x,simulation_result[4][1,:,0])
ax2.plot(x,simulation_result[4][1,:,1])
ax2.plot(x,simulation_result[4][1,:,2])
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Frequency (Hz)")
ax2.legend(("Traj1", "Traj2", "Traj3"))
ax3 = fig.add_subplot(gs[2:3,1])
ax3.eventplot(simulation_result[0][trajectories[0]][1][1], lineoffsets=3, color=colors[0])
ax3.eventplot(simulation_result[0][trajectories[1]][1][1], lineoffsets=2, color=colors[1])
ax3.eventplot(simulation_result[0][trajectories[2]][1][1], lineoffsets=1, color=colors[2])
ax3.set_ylabel("Trajectory")
ax3.set_xlabel("Time (ms)")
ax3.get_yaxis().set_visible(False)
xlim=[-10, 2010]
ax2.set_xlim((xlim))
ax3.set_xlim((xlim))




# =============================================================================
# =============================================================================
# # Phase-Location Diagram
# =============================================================================
# =============================================================================

import sys
sys.path.insert(0, '//home/baris/phase_coding/archive/thesis_code')

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
from grid_short_traj import grid_maker, grid_population, draw_traj
from scipy import interpolate
from skimage.measure import profile_line
from elephant import spike_train_generation as stg
from neo.core import AnalogSignal
import quantities as pq
from scipy import ndimage

def interp(arr, def_dur_s, new_dt_s):
    arr_len = arr.shape[0]
    t_arr = np.linspace(0, def_dur_s, arr_len)
    new_len = int(def_dur_s/new_dt_s)
    new_t_arr = np.linspace(0, def_dur_s, new_len)
    interp_arr = np.interp(new_t_arr, t_arr, arr)
    return interp_arr, new_t_arr

def inhom_poiss(rate_profile, dur_s, poiss_seeds, dt_s=0.002):
    rate_profile = rate_profile
    asig = AnalogSignal(rate_profile,
                            units=1*pq.Hz,
                            t_start=0*pq.s,
                            t_stop=dur_s*pq.s,
                            sampling_period=dt_s*pq.s,
                            sampling_interval=dt_s*pq.s)
    n_poiss = poiss_seeds.shape[0]
    spi_arr = np.zeros((n_poiss), dtype = np.ndarray)
    for idx, seed in enumerate(poiss_seeds):
        np.random.seed(seed)
        curr_train = stg.inhomogeneous_poisson_process(asig)
        spi_arr[idx] = np.array(curr_train.times) #time conv to ms
    return spi_arr


def overall(spacing, center, orientation, dur):
    spacing = spacing
    center = center
    orientation = orientation
    field_size_m = 1
    field_size_cm = field_size_m*100
    arr_size = 200
    max_rate = 1
    new_dt_s = 0.002
    dur_s = dur
    f=10
    T = 1/f
    arr = grid_maker(spacing, orientation, center, arr_size, [field_size_m,field_size_m], max_rate)
    
    
    trans_dist_2d = (np.arccos(((arr*3/(2*max_rate))-1/2))*np.sqrt(2))*np.sqrt(6)*spacing/(4*np.pi)
    trans_norm_2d = (trans_dist_2d/(spacing/2))
    
    
    traj_loc = profile_line(trans_norm_2d, (99,0), (99,(40*dur)-1), mode='constant')
    traj_rate = profile_line(arr, (99,0), (99,(40*dur)-1))
    
    
    
    def_dt = dur_s/traj_loc.shape[0]
    traj_rate, rate_t_arr = interp(traj_rate, dur_s, new_dt_s)
    def_time = np.arange(0, dur_s, def_dt)
    time_hr = rate_t_arr
    shift = 0
    theta = (np.sin(f*2*np.pi*def_time+shift)+1)/2
    theta_hr = (np.sin(f*2*np.pi*time_hr+shift)+1)/2
    phase = (2*np.pi*(def_time%T)/T + shift)%(2*np.pi)
    phase_hr = (2*np.pi*(time_hr%T)/T + shift)%(2*np.pi)
    
    
    #direction
    direction = np.diff(traj_loc)
    direction = np.append(direction, direction[-1])
    direction[direction < 0] = -1
    direction[direction > 0] = 1
    direction[direction == 0] = 1
    direction = -direction
    
    traj_loc_dir = traj_loc*direction
    traj_loc_dir = ndimage.gaussian_filter(traj_loc_dir, sigma=1)
    traj_loc_dir, loc_t_arr = interp(traj_loc_dir, dur_s, new_dt_s)
    loc_hr = np.arange(0, field_size_cm, field_size_cm/traj_loc_dir.shape[0])
    
    factor = 180/360
    firing_phase_dir = 2*np.pi*(traj_loc_dir+0.5)*factor
    phase_code_dir = np.exp(1.5*np.cos(firing_phase_dir-phase_hr))
    overall_dir = phase_code_dir*traj_rate*0.16*20*5
    return overall_dir, arr, time_hr

##############################################


def precession_spikes(overall, dur):
    dt_s = 0.002
    t_sec = dur
    norm_overall = overall_dir
    asig = AnalogSignal(norm_overall,
                                        units=1*pq.Hz,
                                        t_start=0*pq.s,
                                        t_stop=t_sec*pq.s,
                                        sampling_period=dt_s*pq.s,
                                        sampling_interval=dt_s*pq.s)
    
    time_bin_size = 0.1
    T=time_bin_size
    times = np.arange(0, 5+time_bin_size, time_bin_size) 
    
    n_time_bins = int(t_sec/time_bin_size)
    bins_size_deg = 7.2
    phase_norm_fact = 360/bins_size_deg
    n_phase_bins = int(720/bins_size_deg)
    
    phases = [ [] for _ in range(n_time_bins) ]
    n = 1000
    for i in range(n):
        train = np.array(stg.inhomogeneous_poisson_process(asig).times)
        # phase_deg.append(train%T/(t_sec*T)*360)
        for j, time in enumerate(times):
            if j == times.shape[0]-1:
                break
            curr_train = train[np.logical_and(train > time, train < times[j+1])]
            # if curr_train != []:
            if curr_train.size>0:
                phases[j] += list(curr_train%(T)/(T)*360)
                phases[j] += list(curr_train%(T)/(T)*360+360)
                
    
    counts = np.empty((n_phase_bins, n_time_bins))
                      
    for i in range(n_phase_bins):
        for j, phases_in_time in enumerate(phases):
            phases_in_time = np.array(phases_in_time) 
            counts[i][j] = ((bins_size_deg*(i) < phases_in_time) & (phases_in_time < bins_size_deg*(i+1))).sum()
    f=10
    norm_freq = counts*phase_norm_fact*f/n
    norm_freq = ndimage.gaussian_filter(norm_freq, sigma=[1,1])
    return norm_freq

##############################
'phase precession figures'
spacing = 40
center = [100,100]
orientation = 25
dur = 5

overall_dir, rate, _ = overall(spacing, center, orientation, dur)
spike_phases = precession_spikes(overall_dir, dur)

cmap = sns.color_palette('RdYlBu_r', as_cmap=True)
cmap2 = sns.color_palette('RdYlBu_r', as_cmap=True)
f2, (ax1, ax2) = plt.subplots(2,1, sharex=False, gridspec_kw={'height_ratios':[1,2]}, figsize=(7,9))
# f2.tight_layout(pad=0.1)
im2 = ax2.imshow(spike_phases, aspect=1/7, cmap=cmap, extent=[0,100,720,0])
ax2.set_ylim((0,720))
#, extent=[0,100,0,720]
# ax2.title('Mean Firing Frequency ('+str(n)+ ' trials)\n spacing=' + str(spacing)+' cm'
#           +'    phase='+str(center[0])+'    orientation='+str(orientation))
im1 = ax1.imshow(rate[50:150,:], cmap=cmap2)
#, extent=[0,100,80,20]
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_title('\u0394=' + str(spacing)+'       [$x_{0}$ $y_{0}$]='+str(100-np.array(center))+'       \u03A8='+str(orientation), loc='center')
         # \n phase precession=240deg, spacing='+str(spacing)+'cm, timebin=0.100s, phasebin=1deg')
ax2.set_xlabel('Position (cm)')
ax2.set_ylabel('Theta phase (deg)')
f2.subplots_adjust(right=0.8)
# cax = f2.add_axes([0.60,0.16,0.015,0.35])
cax = f2.add_axes([0.80,0.16,0.025,0.35])
cbar = f2.colorbar(im2, cax=cax)
cbar.set_label('Hz', labelpad=15, rotation=270)

save_dir = '/home/baris/figures/'
f2.savefig(save_dir+'phase_prec_spacing_center_orientation_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.eps', dpi=200)
f2.savefig(save_dir+'phase_prec_spacing_center_orientation_'+str(spacing)+'_'+str(center[0])+'_'+str(orientation)+'.png', dpi=200)
#######################################################


# =============================================================================
# =============================================================================
# # Phase-Location Diagram
# =============================================================================
# =============================================================================













