"""
Created on Tue Mar  1 13:33:57 2022

@author: oliver braganza
"""
import pandas as pd
import os 
import shelve
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from load_spikes_olli2 import load_data
from brian2 import *

gseeds = 3              # number of grid seeds (30)
n_poisson = 4           # number of poisson seeds (20)
simulation_time = 2     # simulation time in s (2)
symmetricSTDP = True    # if False runs assymmetric STDP kernel 
start_scope()

''' CA3 Pyramidal cells '''
N = 600                         # number of neurons
taum = 45*ms                    # membrane time constant45
vt = -48*mV                     # threshold
vr = -54*mV                     # reset voltage after a spike
El = -67*mV                     # resting potential
gmax = 1                      # max exc. conductance
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


tauSTDP_list = [50,40]#,30,20,10]  # CA3-CA3 plasticity timewindow
CA3_inh_list =[0.5,1]#,1.5,2,2.5,3,3.5,4,4.5,5]

#print('predicted simulation time: %d hours'%3.6*n_poisson*gseeds*len(tauSTDP_list)*len(CA3_inh_list)/3600)
# tau_w = 1000*ms

eqs_neurons = ''' dv/dt = (El - v) / taum : volt'''
#InputCells = SpikeGeneratorGroup(2000,index_list,spike_time_list*ms)
InputCells = SpikeGeneratorGroup(2000,[],[]*ms)

CA3neurons = NeuronGroup(N, eqs_neurons, threshold='v>vt', reset='v = vr', 
                         method='euler')
CA3neurons.v = El
FFInterneurons = NeuronGroup(I_N, 'dv/dt = (I_El-v) / I_taum : volt', threshold='v>I_vt', 
                          reset='v = vr', refractory=5*ms)
FFInterneurons.v = I_El
S_I_in = Synapses(InputCells, FFInterneurons, on_pre='v+= 5*mV')
S_I_in.connect(p=0.7)       # 40-50 ints per GC (Acsady1998)
S_I_out = Synapses(FFInterneurons, CA3neurons, on_pre='v-=inh', delay=5*ms)
S_I_out.connect(p=0.1)           # random connectivity

Inputs = Synapses(InputCells, CA3neurons, on_pre='v+=15 *mV')
Inputs.connect(p=0.035)     # ~ 70GCs per CA3 cell

S = Synapses(CA3neurons, CA3neurons,
             '''
             w : 1
             dApre/dt = -Apre / taupre : 1 (event-driven)
             dApost/dt = -Apost / taupost : 1 (event-driven)
             ''',
             on_pre='''
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)''',
             on_post='''Apost += dApost
                    w = clip(w + Apre, 0, gmax)''',
             )
S.connect(p=0.01)                                           #  1% connectivity
S.w = '0.0'                                                 # initial weight =0
mon = StateMonitor(S, 'w', record=True)
s_mon_Inp = SpikeMonitor(InputCells)
s_mon = SpikeMonitor(CA3neurons)
r_mon = PopulationRateMonitor(CA3neurons)
Is_mon = SpikeMonitor(FFInterneurons)
I_mon = PopulationRateMonitor(FFInterneurons)
monitors = [mon, s_mon_Inp, s_mon, r_mon, Is_mon, I_mon]
net = Network(collect())
net.add(monitors)
net.store()

#net.run(simulation_time*second, report='text')
 
condition_dict = {}
for tauSTDP in tauSTDP_list:
    taupre = taupost = tauSTDP*ms
    condition_dict[tauSTDP] = {}
    print('tauSTDP: %d'%tauSTDP)
    for inhibition in CA3_inh_list:
        condition_dict[tauSTDP][inhibition]={}
        print('inhibition: %d'%inhibition)
        for condition in ['full','noFB']:
            condition_dict[tauSTDP][inhibition][condition] = {}
            print('condition:'+ condition )
            for gseed in range(1,gseeds+1):
                print('grid seed: %d'%gseed)
                condition_dict[tauSTDP][inhibition][condition][gseed] = {}
                index_dict,spike_time_dict = load_data(condition,gseed)  
                final_weights = []
                CA3_spikes = []
                GC_spikes = []
                I_spikes = []
                for p,index_array in index_dict.items():
                    spike_time_array = spike_time_dict[p]
                    #print('poisson seed',p)
                    net.restore()
                    inh = inhibition*mV
                    InputCells.set_spikes(index_array,spike_time_array*ms)
                    net.run(simulation_time*second, report='text')
                    CA3_spikes.append(np.array(s_mon.count))
                    GC_spikes.append(np.array(s_mon_Inp.count))
                    I_spikes.append(np.array(Is_mon.count))
                    final_weights.append(np.array(mon.w.T[-1]))
                    #CA3_spikes,final_weights,s_mon,mon,s_mon_I,r_mon
                    ''' CA3_spikes is a list(pseeds) of arrays(cells) of total spikes per CA3 cell
                    final_weights is a list(pseeds) of arrays(cells) of final weights across cell'''
                    condition_dict[tauSTDP][inhibition][condition][gseed]['GC_spikes'] = GC_spikes
                    condition_dict[tauSTDP][inhibition][condition][gseed]['CA3_spikes'] = CA3_spikes
                    condition_dict[tauSTDP][inhibition][condition][gseed]['weights'] = final_weights
                    condition_dict[tauSTDP][inhibition][condition][gseed]['I_spikes'] = I_spikes    # weight/time mean over cells,pseeds

with open('sweep_condition_dict.pkl', 'wb') as f:
    pickle.dump(condition_dict, f)

