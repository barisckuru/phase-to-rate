import seaborn as sns, numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd


data = np.load("rate_n_phase_codes_perceptron_2000ms_net-seeds_410-419.npz", allow_pickle=True)

n_seeds = 10
th_grid = data['grid_th_cross'].T.reshape(6*n_seeds)
th_gra = data['gra_th_cross'].T.reshape(6*n_seeds)
speed_grid = 1/th_grid
speed_gra = 1/th_gra

hue = (['similar']*n_seeds+ ['distinct']*n_seeds)*3
code = ['rate']*n_seeds*2 +['phase']*n_seeds*2+['complex']*n_seeds*2


grid_df = pd.DataFrame({'threshold cross': th_grid,
                   'encoding speed': speed_grid,
                   'trajectories': pd.Categorical(hue), 
                   'code': pd.Categorical(code)})

gra_df = pd.DataFrame({'threshold cross': th_gra,
                   'encoding speed': speed_gra,
                   'trajectories': pd.Categorical(hue), 
                   'code': pd.Categorical(code)})


"BARPLOT"

n_seeds = 10
th_grid = data['grid_th_cross']
th_gra = data['gra_th_cross']
speed_grid = 1/th_grid
speed_gra = 1/th_gra

# lr_grid = data['lr_grid']
# lr_gra = data['lr_gra']

#grid th cross
plt.close('all')
ax1 = sns.barplot(x='code', y='threshold cross', hue='trajectories', data=grid_df, order=['rate', 'phase', 'complex'])
plt.title('Threshold Crossing for Grid Codes in 2000ms')
parameters = ('lr = 5*$10^{-4}$,  10 Grid seeds, 5 Poisson seeds,'+ 
              '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

width = ax1.patches[0].get_width()

for idx in range(n_seeds):
    loc = width/2
    rate_sim = th_grid[idx,0]
    rate_diff = th_grid[idx,1]
    phase_sim = th_grid[idx,2]
    phase_diff = th_grid[idx,3]
    complex_sim = th_grid[idx,4]
    complex_diff = th_grid[idx,5]
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


# plt.figure()

# ax2 = sns.barplot(x='code', y='encoding speed', hue='trajectories', data=df, order=['rate', 'phase', 'complex'])
# plt.title('Encoding Speed in 2000ms')
# parameters = ('lr = 5*$10^{-4}$,  9 Grid seeds, 5 Poisson seeds,'+ 
#               '  $encoding speed$ = 1 / number of epochs until RMSE reached a threshold of 0.2')
# plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)
# width = ax1.patches[0].get_width()
# for idx, val  in enumerate(rate_th_sim):
#     loc = width/2
#     rate_sim = 1/val[0]
#     rate_diff = 1/rate_th_diff[idx][0]
#     phase_sim = 1/phase_th_sim[idx][0]
#     phase_diff = 1/phase_th_diff[idx][0]
#     complex_sim = 1/complex_th_sim[idx][0]
#     complex_diff = 1/complex_th_diff[idx][0]
#     sns.lineplot(x=[-loc,loc], y=[rate_diff, rate_sim], color='k')
#     sns.lineplot(x=[1-loc,1+loc], y=[phase_diff, phase_sim], color='k')
#     sns.lineplot(x=[2-loc,2+loc], y=[complex_diff, complex_sim], color='k')
    
    
    
#  "VIOLINPLOT"

# plt.close('all')
# sns.violinplot(x='code', y='threshold cross', hue='trajectories', data=df)
# plt.title('Threshold Crossing in 500ms')
# parameters = ('lr = 5*$10^{-4}$,  9 Grid seeds, 5 Poisson seeds,'+ 
#               '  $threshold cross$ = number of epochs until RMSE reached a threshold of 0.2')
# plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)


# plt.figure()

# sns.violinplot(x='code', y='encoding speed', hue='trajectories', data=df)
# plt.title('Encoding Speed in 500ms')
# parameters = ('lr = 5*$10^{-4}$,  9 Grid seeds, 5 Poisson seeds,'+ 
#               '  $encoding speed$ = 1 / number of epochs until RMSE reached a threshold of 0.2')
# plt.annotate(parameters, (0,0), (0, -35), xycoords='axes fraction', textcoords='offset points', va='top', fontsize=12)

   
    

# fname = 'perceptron_threshold_net_'+str(seed_4s[0])
# np.savez(fname,
#          rate_th_cross_sim=rate_th_cross_sim, 
#          rate_th_cross_diff=rate_th_cross_diff,
#          phase_th_cross_sim=phase_th_cross_sim, 
#          phase_th_cross_diff=phase_th_cross_diff,
#          complex_th_cross_diff=complex_th_cross_diff, 
#          complex_th_cross_sim=complex_th_cross_sim)
        


# sns.set(context='paper',style='whitegrid',palette='colorblind', font='Arial',font_scale=1.5,color_codes=True)

# tips = sns.load_dataset("tips")

# rate_th_sim = []  
# rate_th_diff = []
# phase_th_sim = []
# phase_th_diff = []
# complex_th_sim = []
# complex_th_diff = [] 

# npzfiles = []
# for file in glob.glob("*.npz"):
#     npzfiles.append(file)
#     load = np.load(file, allow_pickle=True)
#     rate_th_sim.append(load['rate_th_cross_sim'])
#     rate_th_diff.append(load['rate_th_cross_diff'])
#     phase_th_sim.append(load['phase_th_cross_sim'])
#     phase_th_diff.append(load['phase_th_cross_diff'])
#     complex_th_sim.append(load['complex_th_cross_sim'])
#     complex_th_diff.append(load['complex_th_cross_diff'])
    
