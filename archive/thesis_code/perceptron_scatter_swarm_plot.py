#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:10:46 2020

@author: bariskuru
"""

import seaborn as sns, numpy as np
import matplotlib.pyplot as plt

sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)




load = np.load('rate_n_phase_perceptron_norm_200ms_10000_iter_0.1_lr.npz', allow_pickle=True) 
load2 = np.load('complex_rate_n_phase_perceptron_norm_200ms_10000_iter_0.1_lr.npz', allow_pickle=True)


rate_th_cross_sim = 1/load['rate_th_cross_sim']
rate_th_cross_diff = 1/load['rate_th_cross_diff']
phase_th_cross_sim = 1/load['phase_th_cross_sim']
phase_th_cross_diff = 1/load['phase_th_cross_diff']
rate_phase_th_cross_diff = 1/load['rate_phase_th_cross_diff']
rate_phase_th_cross_sim = 1/load['rate_phase_th_cross_sim']
complex_th_sim = 1/load2['rate_phase_th_cross_sim']
complex_th_diff = 1/load2['rate_phase_th_cross_diff']

phase_div_rate_sim = np.mean(phase_th_cross_sim/rate_th_cross_sim)
phase_div_rate_sim_sd = np.std(phase_th_cross_sim/rate_th_cross_sim)
phase_div_rate_diff = np.mean(phase_th_cross_diff/rate_th_cross_diff)
phase_div_rate_diff_sd = np.std(phase_th_cross_diff/rate_th_cross_diff)

phaserate_div_rate_sim = np.mean(rate_phase_th_cross_sim/rate_th_cross_sim)
phaserate_div_rate_sim_sd = np.std(rate_phase_th_cross_sim/rate_th_cross_sim)
phaserate_div_rate_diff = np.mean(rate_phase_th_cross_diff/rate_th_cross_diff)
phaserate_div_rate_diff_sd = np.std(rate_phase_th_cross_diff/rate_th_cross_diff)


print(phase_div_rate_sim_sd)
print(phase_div_rate_diff_sd)
print(phaserate_div_rate_sim_sd)
print(phaserate_div_rate_diff_sd)

codes_sim_rate = np.array([1.404022505171842, 1.9323523155598052, 1.9486420446947548, 2.1840740559705742, 3.610007553446395 ])
codes_diff_rate = np.array([2.3253118260564465, 2.212299126261121, 2.0744256027830157, 1.3746606618529944, 1.4538658536919962])

rate_sim = np.array([0.7396897809622345, 0.7520905384791152, 0.7847187580658658, 0.8137329086235277, 1.006543586763082])
rate_diff = np.array([0.7381005325200718, 0.7605001986062314, 0.7730398750149322, 0.6872663138313297, 0.8029616506385884])


codes_sim_rate_sd = np.array([0.2842087718147501, 0.5581694310716737, 0.640995321635472, 0.7599530054782124, 1.5766815901282467])
codes_diff_rate_sd = np.array([0.41632775561108937, 0.36986468163652586, 0.640995321635472, 0.19320625097812344, 0.33622615846673487])
rate_sim_sd = np.array([0.13864021391814005, 0.2236673556713449, 0.18084979563918252, 0.22400946192019036, 0.2891647601939379])
rate_diff_sd = np.array([0.13558053179918558, 0.1592256235398166, 0.2301861772804394, 0.1731833909905886, 0.17005201715440216])



rate_th_cross_sim = 1/np.array([6485, 6292, 6652])
mean_rate_sim = np.mean(rate_th_cross_sim)
sd_rate_sim = np.std(rate_th_cross_sim)
phase_th_cross_sim = 1/np.array([663, 661, 709])
mean_phase_sim = np.mean(phase_th_cross_sim)
sd_phase_sim = np.std(phase_th_cross_sim)
complex_th_cross_sim = 1/np.array([1717, 2581, 2715])
mean_complex_sim = np.mean(complex_th_cross_sim)
sd_complex_sim = np.std(complex_th_cross_sim)

rate_th_cross_diff = 1/np.array([2693, 1314, 2839])
mean_rate_diff = np.mean(rate_th_cross_diff)
sd_rate_diff = np.std(rate_th_cross_diff)
phase_th_cross_diff = 1/np.array([289, 313, 316])
mean_phase_diff= np.mean(phase_th_cross_diff)
sd_phase_diff = np.std(phase_th_cross_diff)
complex_th_cross_diff = 1/np.array([877, 1234, 1035])
mean_complex_diff = np.mean(complex_th_cross_diff)
sd_complex_diff = np.std(complex_th_cross_diff)

means_sim = np.vstack((mean_rate_sim, mean_phase_sim, mean_complex_sim ))
sds_sim= np.vstack((sd_rate_sim, sd_phase_sim, sd_complex_sim ))
means_diff= np.vstack ((mean_rate_diff,mean_phase_diff, mean_complex_diff))
sds_diff = np.vstack((sd_rate_diff, mean_phase_diff, mean_complex_diff))
data_sim = np.vstack((rate_th_cross_sim, phase_th_cross_sim, complex_th_sim)).T
data_diff = np.vstack((rate_th_cross_diff, phase_th_cross_diff, complex_th_diff)).T

xaxis = np.arange(1,4,1)
plt.errorbar(xaxis, means_sim, yerr=sds_sim, fmt='-', capsize=4, label='75cm vs 74.5cm')
plt.errorbar(xaxis, means_diff, yerr=sds_diff, fmt='-', capsize=4, label='75cm vs 60cm')
plt.title('Mean Learning Speeds in 2000ms')
plt.ylabel('Mean Speed ($1/N_E$)')
plt.xticks([1, 2, 3], ['rate', 'phase', 'complex'])
plt.legend()

parameters = ('lr = $10^{-4}$, Error Bars = SD, 3 Grid seeds, 5 Poisson seeds,'+ 
              '  $N_E$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)


ax = sns.swarmplot(data=data_sim)
plt.title('Perceptron Learning Speed | Similar Traj | 2000ms ')
plt.ylabel('Speed ($1/N_E$)')
plt.xticks([0, 1, 2], ['rate code', 'phase code', 'complex code'])
parameters = ('learning rate = 5* $10^{-3}$ , 3 grid seeds'+ 
              ', 5 Poisson seeds, $N_E$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

plt.figure()
ax2 = sns.swarmplot(data=data_diff)
plt.title('Perceptron Learning Speed | Distinct Traj | 2000ms ')
plt.ylabel('Speed ($1/N_E$)')
plt.xticks([0, 1, 2], ['rate code', 'phase code', 'complex code'])
parameters = ('learning rate = 5* $10^{-3}$ , 3 grid seeds'+ 
              ', 5 Poisson seeds, $N_E$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)













ax = sns.swarmplot(data=data)
plt.title('Perceptron Learning Speed | Dissimilar | 2000ms ')
plt.ylabel('Speed ($1/N_E$)')
plt.xticks([0, 1, 2, 3, 4, 5], ['rate similar', 'rate distinct', 'phase similar', 'phase distinct', 'rate*phase similar', 'rate*phase distinct'])
parameters = ('learning rate = $10^{-2}$ \n 20 seeds for grid & network generation'+ 
              ', 5 Poisson seeds \n $N_E$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)


xaxis = np.arange(1,6,1)

plt.errorbar(xaxis, codes_sim_rate, yerr=codes_sim_rate_sd, fmt='-', capsize=4, label='75cm vs 74.5cm')
plt.errorbar(xaxis, codes_diff_rate, yerr=codes_diff_rate_sd, fmt='-', capsize=4, label='75cm vs 60cm')
              
plt.title('Mean Ratio of PhaseCode/RateCode Learning Speeds')
plt.ylabel('Mean Phase($1/N_E$)/Rate($1/N_E$)')
plt.xticks([0, 1, 2, 3, 4], ['2000ms', '1000ms', '800ms', '400ms', '200ms'])
plt.legend()

parameters = ('learning rate = $10^{-4}$,      Error Bars = SD \n 5 seeds for grid & network generation'+ 
              '\n $N_E$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)



plt.errorbar(xaxis, rate_sim, yerr=rate_sim_sd, fmt='-', capsize=4, label='75cm vs 74.5cm')
plt.errorbar(xaxis, rate_diff, yerr=rate_diff_sd, fmt='-', capsize=4, label='75cm vs 60cm')
plt.title('Mean Ratio of Phase*RateCode/RateCode Learning Speeds')
plt.ylabel('Mean Phase($1/N_E$)/Rate($1/N_E$)')
plt.xticks([0, 1, 2, 3, 4], ['2000ms', '1000ms', '800ms', '400ms', '200ms'])
plt.legend()

parameters = ('learning rate = $10^{-4}$,      Error Bars = SD \n 5 seeds for grid & network generation'+ 
              '\n $N_E$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)



xs = np.ones(20)

xs = np.arange()

plt.scatter(phase_th_cross_sim, rate_th_cross_sim)
plt.scatter(phase_th_cross_diff, rate_th_cross_diff)
plt.title('Perceptron Learning Speed | 800ms ')
plt.ylabel('Rate Code Learning Speed ($1/N_E$)')
plt.xlabel('Phase Code Learning Speed ($1/N_E$)')
plt.legend(['Similar Trajectories (75-74.5cm)', 'Distinct Trajectories (75-60cm)'])
parameters = ('learning rate = $10^{-4}$ \n 20 seeds for grid & network generation'+ 
              ', 5 Poisson seeds \n $N_E$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -30), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)





plt.scatter(phase_th_cross_sim, rate_th_cross_sim)
plt.scatter(phase_th_cross_diff, rate_th_cross_diff)
plt.title('Threshold Crossing (Epochs)')
plt.ylabel('Epochs')
plt.xticks([1, 2, 3, 4, 5, 6], ['rate similar', 'rate distinct', 'phase similar', 'phase distinct'])







plt.plot(data)
plt.title('Threshold Crossing (Epochs)')
plt.ylabel('Epochs')
plt.xlabel('Trials with different seeds')
plt.legend(['rate similar', 'rate distinct', 'phase similar', 'phase distinct'])


plt.scatter(xs, data)
plt.title('Threshold Crossing (Epochs)')
plt.ylabel('Epochs')
plt.xlabel('Trials with different seeds')
plt.legend(['rate similar', 'rate distinct', 'phase similar', 'phase distinct'])


