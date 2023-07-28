# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:33:57 2022

@author: oliver braganza
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from brian2 import (
    NeuronGroup,
    Synapses,
    SpikeGeneratorGroup,
    SpikeMonitor,
    StateMonitor,
    Network,
    start_scope,
    collect
)
from brian2 import ms, mV, pA, nA, Mohm, Hz, second
#from brian2 import run

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

simulation_time = 0.5     # simulation time in s (2)
synapse = 'Int-CA3Pyr'
#synapse = 'MF-Int'
#synapse = 'CA3Pyr-CA3Pyr'
#synapse = 'MF-CA3Pyr'
#synapse = 'Ex-Inh'



#==============================================================================
# mossy fiber facilitation
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
                        method="exact")
    synapses.connect() 

    # start fully recovered
    synapses.x = 1
    synapses.y = 0

    synapses.tau_inact = tau_inact
    synapses.A_SE = A_SE
    synapses.U_SE = U_SE
    synapses.tau_rec = tau_rec

    if tau_facil:
        synapses.tau_facil = tau_facil

    return synapses

def get_stimulus(start, stop, frequency):
    """
    start -- start time of stimulus
    stop -- stop time of stimulus
    frequency -- frequency of stimulus
    """

    times = np.arange(start / ms, stop / ms, 1 / (frequency / Hz) * 1e3) * ms
    stimulus = SpikeGeneratorGroup(2, [0] * len(times), times)

    return stimulus

start_scope()

''' CA3 Pyramidal cells '''
tau_m = 45*ms                    # membrane time constant45
vt = -48*mV                     # threshold
vr = -54*mV                     # reset voltage after a spike
El = -67*mV                     # resting potential
R_in = 150*Mohm


''' CA3 Interneurons (BC values from Neuroelectro'''
I_tau_m = 14*ms                   # interneuron time constant
I_vt = -37*mV
I_El = -52*mV
I_R_in = 112*Mohm
w_I = 300 * pA
w_E = 40 * pA
tau_s = 5*ms


eqs_Pyr_neuron = '''e_syn : ampere
                    i_syn : ampere
                    dv/dt = (El - v)/tau_m + (R_in*(e_syn - i_syn))/tau_m : volt '''

eqs_Int_neuron = '''e_syn : ampere
                    dv/dt = (I_El - v)/I_tau_m + (I_R_in*e_syn)/I_tau_m : volt '''

ISI = 50
start = 100
stop = start+ 3*ISI
freq = 1000//ISI
InputCell = get_stimulus(start * ms, stop * ms, freq * Hz)

Pyr_neuron = NeuronGroup(1, eqs_Pyr_neuron, threshold='v>vt', reset='v = vr', 
                         method='exact')
Pyr_neuron.v = El

Int_neuron = NeuronGroup(1, eqs_Int_neuron, threshold='v>I_vt', reset='v = vr', 
                         method='exact')
Int_neuron.v = I_El

if synapse == 'Ex-Inh':
    # Note stoichiometry factors (30 & 6) are included to approximate different population sizes
    InputsI = Synapses(InputCell, Int_neuron, model='''
                      dw/dt =  -w/tau_s : ampere (clock-driven)
                      e_syn_post = w : ampere (summed)
                      ''',
                      on_pre='w += 30*w_E')
    InputsP = get_synapses(InputCell, Pyr_neuron, tau_inact=30*ms, A_SE=5*nA, U_SE=0.03, tau_rec=130*ms, tau_facil=530*ms)
    ItoP = Synapses(Int_neuron, Pyr_neuron, model='''
                      dw/dt =  -w/tau_s : ampere (clock-driven)
                      i_syn_post = w : ampere (summed)
                      ''',
                      on_pre='w += 6*w_I',
                      delay=5*ms)
    ItoP.connect()
    InputsI.connect()
    #InputsP.connect()

    syn_monPi = StateMonitor(ItoP, 'i_syn_post', record=True)
    syn_monPe = StateMonitor(InputsP, 'e_syn_post', record=True)
    monI = StateMonitor(Int_neuron, 'v', record=True)
    mon = StateMonitor(Pyr_neuron,['v'], record=True)
    s_mon_Inp = SpikeMonitor(InputCell)
    monitors = [mon, monI, s_mon_Inp, syn_monPe, syn_monPi]
else:
# Input_group, Output_group, inact_tau, max_strength, fractional_strength, rec_tau, facil_tau
    if synapse == 'MF-CA3Pyr':
        Inputs = get_synapses(InputCell, Pyr_neuron, tau_inact=30*ms, A_SE=5*nA, U_SE=0.03, tau_rec=130*ms, tau_facil=530*ms)
        #Inputs.connect()
        #Pyr_neuron.i_syn = 0*pA
        syn_mon = StateMonitor(Pyr_neuron, 'e_syn', record=True)
    
    elif synapse == 'MF-Int':
        Inputs = Synapses(InputCell, Int_neuron, model='''
                          dw/dt =  -w/tau_s : ampere (clock-driven)
                          e_syn_post = w : ampere (summed)
                          ''',
                          on_pre='w += w_E')
        Inputs.connect()
        syn_mon = StateMonitor(Inputs, 'e_syn_post', record=True)
    
    elif synapse == 'Int-CA3Pyr':
        Inputs = Synapses(InputCell, Pyr_neuron, model='''
                          dw/dt =  -w/tau_s : ampere (clock-driven)
                          i_syn_post = w : ampere (summed)
                          ''',
                          on_pre='w += w_I')
        Inputs.connect()
        syn_mon = StateMonitor(Inputs, 'i_syn_post', record=True)
    
    elif synapse == 'CA3Pyr-CA3Pyr':
        Inputs = Synapses(InputCell, Pyr_neuron, model='''
                          dw/dt =  -w/tau_s : ampere (clock-driven)
                          e_syn_post = w : ampere (summed)
                          ''',
                          on_pre='w += w_E')
        Inputs.connect()
        syn_mon = StateMonitor(Inputs, 'e_syn_post', record=True)
    
        
    mon = StateMonitor(Pyr_neuron,['v'], record=True)
    s_mon_Inp = SpikeMonitor(InputCell)
    #syn_mon = StateMonitor(Inputs, ['e_syn_post','i_syn_post'], record=True)
    monitors = [mon, s_mon_Inp, syn_mon]

net = Network(collect())
net.add(monitors)
net.store()

net.run(simulation_time*second, report='text') 

if synapse == 'Ex-Inh':
    
    f1, ((ax1, ax2, ax3, ax4)) = plt.subplots(4, 1, figsize=(4, 9), sharex=True)
    f1.subplots_adjust(hspace=.2, wspace=.2, left=0.1, right=0.9)
    v_int = monI[0].v/mV
    e_current = -syn_monPe[0].e_syn_post/pA
    i_current = syn_monPi[0].i_syn_post/pA
    
    ax1.plot(s_mon_Inp.t/ms, s_mon_Inp.i, '|k')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Inputs')
    
    ax2.plot(monI.t/ms, v_int, 'k')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Int. syn. current (pA)')
    
    ax3.plot(syn_monPe.t/ms, e_current, 'k')
    ax3.plot(syn_monPe.t/ms, i_current, 'k')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('syn. currents (pA)')
    
    ax4.plot(mon[0].t/ms, mon[0].v/mV, 'k')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('v (mV)')
    
    f1.savefig('Facilitation.pdf')
else:    

    f1, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(4, 9), sharex=True)
    f1.subplots_adjust(hspace=.2, wspace=.2, left=0.1, right=0.9)
    
    f1.suptitle('synapse')
    if synapse == 'MF-CA3Pyr':
        current = -syn_mon[0].e_syn/pA
    elif synapse == 'MF-Int':
        current = -syn_mon[0].e_syn_post/pA
    elif synapse == 'Int-CA3Pyr':
        current = syn_mon[0].i_syn_post/pA
    elif synapse == 'CA3Pyr-CA3Pyr':
        current = -syn_mon[0].e_syn_post/pA
    
    
    ax1.plot(s_mon_Inp.t/ms, s_mon_Inp.i, '|k')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Inputs')
    
    ax2.plot(syn_mon.t/ms, current, 'k')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('syn. current (pA)')
    
    ax3.plot(mon[0].t/ms, mon[0].v/mV, 'k')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('v (mV)')
    
    f1.savefig('Facilitation.pdf')


