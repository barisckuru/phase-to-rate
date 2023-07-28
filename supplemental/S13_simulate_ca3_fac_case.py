# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:33:57 2022

@author: oliver braganza
"""
import numpy as np
import matplotlib as mpl
import pickle
from load_spikes_olli2 import load_data
from brian2 import *

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#def run_brian(index_dict,spike_time_dict):
gseeds = 20              # number of grid seeds (10)
n_poisson = 10          # number of poisson seeds (10)
simulation_time = 2     # simulation time in s (2)
symmetricSTDP = True    # if False runs assymmetric STDP kernel

''' cross-correlograms '''
from elephant.conversion import BinnedSpikeTrain
from elephant.spike_train_correlation import cross_correlation_histogram
from viziphant.spike_train_correlation import plot_cross_correlation_histogram
from neo.core import SpikeTrain
from quantities import s
from itertools import combinations

def crosscorrelogram(brian_spikemonitor, ms_window):
    show_bins = ms_window//10
    data = brian_spikemonitor
    pop_dict = data.event_trains()
    cell_list = []
    for key in pop_dict:
        train = pop_dict[key]
        neo_train = SpikeTrain(train/second*s, t_stop=simulation_time)
        cell_list.append(neo_train)
    binned_spike_trains = BinnedSpikeTrain(cell_list, n_bins=ms_window)
    pop_hist = np.zeros(2*show_bins+1)
    for a, b in combinations(binned_spike_trains, 2):         
        hist, lags = cross_correlation_histogram(
               a, b, window=[-show_bins,show_bins], border_correction=True)
        pop_hist = pop_hist + hist
    pop_hist = pop_hist/np.nanmean(pop_hist)
    return pop_hist[:,0],lags

#==============================================================================
# CA3 model with mossy fiber facilitation
#==============================================================================

def get_synapses(stimulus, neuron, tau_inact, A_SE, U_SE, tau_rec, tau_facil=None):
    """
    stimulus -- input stimulus
    neuron -- target neuron
    tau_inact -- inactivation time constant
    A_SE -- absolute synaptic strength
    U_SE -- utilization of synaptic efficacy
    tau_rec -- recovery time constant
    tau_facil -- facilitation time constant (optional)
    """

    synapses_eqs = """
    dx/dt =  z/tau_rec   : 1 (clock-driven) # recovered
    dy/dt = -y/tau_inact : 1 (clock-driven) # active
    A_SE : ampere
    U_SE : 1
    tau_inact : second
    tau_rec : second
    z = 1 - x - y : 1 # inactive
    e_syn_post = A_SE*y : ampere (summed)
    """

    if tau_facil:
        synapses_eqs += """
        du/dt = -u/tau_facil : 1 (clock-driven)
        tau_facil : second
        """

        synapses_action = """
        u += U_SE*(1-u)
        y += u*x # important: update y first
        x += -u*x
        """
    else:
        synapses_action = """
        y += U_SE*x # important: update y first
        x += -U_SE*x
        """

    synapses = Synapses(stimulus,
                        neuron,
                        model=synapses_eqs,
                        on_pre=synapses_action,
                        method="exponential_euler")
    synapses.connect(p=0.035) 

    # start fully recovered
    synapses.x = 1

    synapses.tau_inact = tau_inact
    synapses.A_SE = A_SE
    synapses.U_SE = U_SE
    synapses.tau_rec = tau_rec

    if tau_facil:
        synapses.tau_facil = tau_facil

    return synapses


start_scope()

''' CA3 Pyramidal cells '''
N = 600                         # number of neurons
taum = 45*ms                    # membrane time constant45
vt = -48*mV                     # threshold
vr = -54*mV                     # reset voltage after a spike
El = -67*mV                     # resting potential
R_in = 150*Mohm
gmax = 1                        # max exc. conductance
dApre = .1
if symmetricSTDP:
    dApost = dApre
else:
    dApost = -dApre
dApost *= gmax
dApre *= gmax

''' CA3 Interneurons (BC values from Neuroelectro'''
I_N = 60
I_taum = 14*ms                   # interneuron time constant
I_vt = -37*mV
I_El = -52*mV
I_R_in = 112*Mohm
w_I = 60 * pA
w_E = 40 * pA
tau_s = 15*ms
homeostasis = 0.05


tauSTDP = 20*ms
taupre = taupost = tauSTDP

eqs_neurons = '''e_syn : ampere
                 e_synCA3 : ampere
                 i_syn : ampere
                 dv/dt = (El - v + (R_in*(e_syn+e_synCA3-i_syn))) / taum : volt '''
#InputCells = SpikeGeneratorGroup(2000,index_list,spike_time_list*ms)
InputCells = SpikeGeneratorGroup(2000,[],[]*ms)

CA3neurons = NeuronGroup(N, eqs_neurons, threshold='v>vt', reset='v = vr', 
                         method='exact')
CA3neurons.v = El

FFInterneurons = NeuronGroup(I_N,'''
                             e_syn : ampere
                             dv/dt = (I_El-v+(I_R_in*e_syn)) / I_taum : volt''',
                             threshold='v>I_vt',
                             reset='v = vr', 
                             refractory=5*ms)
FFInterneurons.v = I_El
#S_I_in = get_synapses(InputCells, FFInterneurons, 15*ms, 8*nA, 0.03, 130*ms, 530*ms) 
S_I_in = Synapses(InputCells, FFInterneurons, model='''
                  dw/dt =  -w/tau_s : ampere (clock-driven)
                  e_syn_post = w : ampere (summed)
                  ''',
                  on_pre='w += w_E')

#Synapses(InputCells, FFInterneurons, on_pre='v+=5 *mV')
#S_I_out = get_synapses(FFInterneurons, CA3neurons, 15*ms, 2*nA, 0.03, 130*ms)
S_I_out = Synapses(FFInterneurons, CA3neurons, model='''
                  dw/dt =  -w/tau_s : ampere (clock-driven)
                  i_syn_post = w : ampere (summed)
                  ''',
                  on_pre='w += w_I',
                  delay=5*ms)
S_I_in.connect(p=0.7)       # 40-50 ints per GC (Acsady1998)
S_I_out.connect(p=0.1)           # random connectivity

#Inputs = Synapses(InputCells, CA3neurons, on_pre='v+=15 *mV')

# Input_group, Output_group, inact_tau, max_strength, fractional_strength, rec_tau, facil_tau
Inputs = get_synapses(InputCells, CA3neurons, 30*ms, 5*nA, 0.03, 130*ms, 530*ms)

#Inputs.connect(p=0.035)     # ~ 70GCs per CA3 cell

S = Synapses(CA3neurons, CA3neurons,
             '''
             Wp : 1
             dw/dt = -w/tau_s : 1 (clock-driven)
             dApre/dt = -Apre / taupre : 1 (event-driven)
             dApost/dt = -Apost / taupost : 1 (event-driven)
             e_synCA3_post = w*w_E : ampere (summed)
             ''',
             on_pre='''
                    Apre += dApre
                    Wp = clip(Wp + Apost, 0, gmax)
                    w += Wp
                    ''',
             on_post='''Apost += dApost
                    Wp = clip(Wp + Apre, 0, gmax)
                    Wp -= Wp*homeostasis
                   ''',
             )
S.connect(p=0.01)                                           #  1% connectivity
S.Wp = 0                                                    # initial weight =0
mon = StateMonitor(S, 'Wp', record=True)
s_mon_Inp = SpikeMonitor(InputCells)
s_mon = SpikeMonitor(CA3neurons)
r_mon = PopulationRateMonitor(CA3neurons)
Is_mon = SpikeMonitor(FFInterneurons)
I_mon = PopulationRateMonitor(FFInterneurons)
monitors = [mon, s_mon_Inp, s_mon, r_mon, Is_mon, I_mon]
net = Network(collect())
net.add(monitors)
net.store()

# net.run(simulation_time*second, report='text') 
corr_dict = {} 
condition_dict = {}
print('starting case')
for condition in ['full','noFB']:
    print(condition)
    condition_dict[condition] = {}
    for gseed in range(11,gseeds+11):
        print(gseed)
        condition_dict[condition][gseed] = {}
        index_dict,spike_time_dict = load_data(condition,gseed,n_poisson)  
        final_weights = []
        CA3_spikes = []
        GC_spikes = []
        I_spikes = []
        for p,index_array in index_dict.items():
            spike_time_array = spike_time_dict[p]
            #print('poisson seed',p)
            net.restore()
            InputCells.set_spikes(index_array,spike_time_array*ms)
            net.run(simulation_time*second, report='text')
            CA3_spikes.append(np.array(s_mon.count))
            GC_spikes.append(np.array(s_mon_Inp.count))
            I_spikes.append(np.array(Is_mon.count))
            final_weights.append(np.array(mon.Wp.T[-1]))
            #CA3_spikes,final_weights,s_mon,mon,s_mon_I,r_mon
            ''' CA3_spikes is a list(pseeds) of arrays(cells) of total spikes per CA3 cell
            final_weights is a list(pseeds) of arrays(cells) of final weights across cell'''
            condition_dict[condition][gseed]['GC_spikes'] = GC_spikes
            condition_dict[condition][gseed]['CA3_spikes'] = CA3_spikes
            condition_dict[condition][gseed]['weights'] = final_weights
            condition_dict[condition][gseed]['I_spikes'] = I_spikes
        # weight/time mean over cells,pseeds
    
    corr_dict[condition]={}
    cc_hist,lags = crosscorrelogram(s_mon, 500)
    corr_dict[condition]['CA3']=[cc_hist,lags]
    plt.plot(lags,cc_hist,c='k')
    cc_hist,lags = crosscorrelogram(s_mon_Inp, 500)
    corr_dict[condition]['GC']=[cc_hist,lags]
with open('corr_dict.pkl', 'wb') as f:
    pickle.dump(corr_dict, f)

    
with open('condition_dict.pkl', 'wb') as f:
    pickle.dump(condition_dict, f)


f1, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
f1.subplots_adjust(hspace=.2, wspace=.2, left=0.1, right=0.9)

ax1.plot(s_mon_Inp.t/ms, s_mon_Inp.i, '|k', alpha=0.2)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('GC index')
ax1.text(0,0,"mean: %.2f Hz"%(s_mon_Inp.num_spikes/2000/simulation_time))

ax2.plot(s_mon.t/ms, s_mon.i, '|k', alpha=0.2)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('CA3 index')
ax2.text(0,0,"mean %.2f Hz"%(s_mon.num_spikes/600/simulation_time))

ax3.plot(Is_mon.t/ms, Is_mon.i, '|k', alpha=0.2)
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Interneuron index')
ax3.text(0,0,"mean %.2f Hz"%(Is_mon.num_spikes/60/simulation_time))

f1.savefig('RasterPlots.pdf')

f2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
f2.subplots_adjust(hspace=.6, wspace=.6, left=0.1, right=0.9)

ax1.plot(S.Wp, '.k')
ax1.set_ylabel('Weight')
ax1.set_xlabel('Synapse index')

ax2.hist(S.Wp, 20, log=True)
ax2.set_xlabel('Weight')

#ax3.plot(mon.t/second, mon.w.T/mon.w.T[0], c='b', alpha =0.2)
ax3.plot(mon.t/second, mon.Wp.T, c='grey', alpha=0.1)
ax3.plot(mon.t/second, np.mean(mon.Wp.T, axis=1), c='k')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Weight')

ax4.plot(r_mon.t/second, r_mon.smooth_rate(width=1*ms)/Hz)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('rate')


f3, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
f3.subplots_adjust(hspace=.2, wspace=.2, left=0.1, right=0.9)

x=corr_dict[condition]['CA3'][1]*4

ax1.plot(x, corr_dict['full']['GC'][0], c='b')
ax1.plot(x, corr_dict['noFB']['GC'][0], c='b', alpha=0.5)
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('GC crosscorr')

ax2.plot(x, corr_dict['full']['CA3'][0], c='g')
ax2.plot(x, corr_dict['noFB']['CA3'][0], c='g', alpha=0.5)
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('CA3 crosscorr')

f3.savefig('Crosscorrelograms_MF_fac and CA3_input.pdf')


