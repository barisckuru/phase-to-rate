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
from datetime import datetime
#from S12_simulate_ca3_fac_case import get_synapses


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

from load_spikes_olli2 import load_data
from brian2 import *

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

gseeds = 4              # number of grid seeds (30)
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
R_in = 150*Mohm
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
I_R_in = 112*Mohm
#w_I = 30 * pA
w_E = 40 * pA
tau_s = 15*ms
homeostasis = 0.2


tauSTDP_list = [50,40,30,20,10]  # CA3-CA3 plasticity timewindow
CA3_inh_list =[20,30,40,50,60,70,80,90]

#print('predicted simulation time: %d hours'%3.6*n_poisson*gseeds*len(tauSTDP_list)*len(CA3_inh_list)/3600)
# tau_w = 1000*ms


eqs_neurons = '''e_syn : ampere
                 e_synCA3 : ampere
                 i_syn : ampere
                 dv/dt = (El - v + (R_in*(e_syn+e_synCA3-i_syn))) / taum : volt '''#InputCells = SpikeGeneratorGroup(2000,index_list,spike_time_list*ms)
InputCells = SpikeGeneratorGroup(2000,[],[]*ms)

CA3neurons = NeuronGroup(N, eqs_neurons, threshold='v>vt', reset='v = vr', 
                         method='euler')
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

S_I_out = Synapses(FFInterneurons, CA3neurons, model='''
                  dw/dt =  -w/tau_s : ampere (clock-driven)
                  i_syn_post = w : ampere (summed)
                  w_I : ampere
                  ''',
                  on_pre='w += w_I',
                  delay=5*ms)
S_I_out.connect(p=0.1)           # random connectivity
#Synapses(InputCells, FFInterneurons, on_pre='v+=5 *mV')
#S_I_out = get_synapses(FFInterneurons, CA3neurons, 15*ms, 2*nA, 0.03, 130*ms)


S_I_in.connect(p=0.7)       # 40-50 ints per GC (Acsady1998)
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

#net.run(simulation_time*second, report='text')
#n_poisson = 4 
condition_dict = {}
print('starting parameter sweeps')
for tauSTDP in tauSTDP_list:
    taupre = taupost = tauSTDP*ms
    condition_dict[tauSTDP] = {}
    print('tauSTDP: %d'%tauSTDP)
    now = datetime.datetime.now()
    print(now.time())
    for inhibition in CA3_inh_list:
        condition_dict[tauSTDP][inhibition]={}
        print('inhibition: %d'%inhibition)
        for condition in ['full','noFB']:
            condition_dict[tauSTDP][inhibition][condition] = {}
            print('condition:'+ condition )
            for gseed in range(11,11+gseeds):
                print('grid seed: %d'%gseed)
                condition_dict[tauSTDP][inhibition][condition][gseed] = {}
                index_dict,spike_time_dict = load_data(condition,gseed,n_poisson)  
                final_weights = []
                CA3_spikes = []
                GC_spikes = []
                I_spikes = []
                for p,index_array in index_dict.items():
                    start_scope()
                    spike_time_array = spike_time_dict[p]
                    #print('poisson seed',p)
                    net.restore()
                    w_I = inhibition*pA
                    S_I_out.w_I = w_I
                    InputCells.set_spikes(index_array,spike_time_array*ms)
                    net.run(simulation_time*second, report=None)
                    CA3_spikes.append(np.array(s_mon.count))
                    GC_spikes.append(np.array(s_mon_Inp.count))
                    I_spikes.append(np.array(Is_mon.count))
                    final_weights.append(np.array(mon.Wp.T[-1]))
                    #CA3_spikes,final_weights,s_mon,mon,s_mon_I,r_mon
                    ''' CA3_spikes is a list(pseeds) of arrays(cells) of total spikes per CA3 cell
                    final_weights is a list(pseeds) of arrays(cells) of final weights across cell'''
                    condition_dict[tauSTDP][inhibition][condition][gseed]['GC_spikes'] = GC_spikes
                    condition_dict[tauSTDP][inhibition][condition][gseed]['CA3_spikes'] = CA3_spikes
                    condition_dict[tauSTDP][inhibition][condition][gseed]['weights'] = final_weights
                    condition_dict[tauSTDP][inhibition][condition][gseed]['I_spikes'] = I_spikes    # weight/time mean over cells,pseeds

with open('sweep_condition_dict2.pkl', 'wb') as f:
    pickle.dump(condition_dict, f)

