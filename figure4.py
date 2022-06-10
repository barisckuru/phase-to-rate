# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 20:31:08 2022

@author: Daniel
"""

import sqlite3
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_ind
import numpy as np
import scipy.optimize
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.anova import AnovaRM
import scikit_posthocs as sp
import shelve
from tempotron.main import Tempotron

"""Load data"""
dirname = os.path.dirname(__file__)
db_path = os.path.join(
    dirname, 'data', 'tempotron_thresholds_mean.db')
con = sqlite3.connect(db_path)
cur = con.cursor()

rows = cur.execute(f"""SELECT * FROM tempotron_run WHERE n_cells=2000 AND (tempotron_seed=91 OR tempotron_seed=95)""")
data = []
for x in rows:
    data.append(x)
    print(x)

df = pd.DataFrame(data=data, columns=['tempotron_seed', 'epochs', 'time', 'Vrest', 'tau',
             'tau_s', 'threshold', 'learning_rate', 'n_cells',
             'trajectory_one', 'trajectory_two', 'pre_accuracy',
             'trained_accuracy', 'pre_loss', 
             'trained_loss', 'delta_loss', 'distance', 'grid_seed', 'duration', 
             'shuffling', 'network', 'cell_type', 'file_id'])

rate_path = os.path.join(dirname, 'data', 'mean_rates.csv')
df_mean_rates = pd.read_csv(rate_path, index_col=0)

"""Preprocessing"""
df = df.drop_duplicates()

to_exclude = ((df_mean_rates['mean_rate'] < 0.20) | (df_mean_rates['mean_rate'] > 0.30))
grid_seeds_excluded = df_mean_rates[to_exclude]['grid_seed'].unique()
df.drop(df[df['grid_seed'].isin(grid_seeds_excluded)].index, inplace=True)
df_mean_rates.drop(df_mean_rates[df_mean_rates['grid_seed'].isin(grid_seeds_excluded)].index, inplace=True)

df['mean_rate'] = np.nan

for index, row in df_mean_rates.iterrows():
    target_rows = ((df['grid_seed'] == row['grid_seed']) &
                   (df['shuffling'] == row['shuffling']) &
                   (df['network'] == row['network'])
                  )
    if target_rows.sum() > 1:
        raise ValueError()
    df.loc[target_rows, 'mean_rate'] = row['mean_rate']

def exp_decay(x, tau, init):
    return init*np.e**(-x/tau)

files = {}
taus = []
for idx in df.index:
    fid = df.loc[idx]['file_id']
    key = f"{fid}_{df.loc[idx]['shuffling']}_{df.loc[idx]['network']}"
    file_path = db_path = os.path.join(
        dirname, 'data', 'arrays', fid+'.npy')
    curr_data = np.load(file_path)
    popt, pcov = scipy.optimize.curve_fit(exp_decay, np.arange(200), curr_data[1,:], p0=[100, 300])
    fit_tau, fit_init = popt
    files[key] = curr_data
    taus.append(fit_tau)
df['decay_tau'] = 1 / np.array(taus)

a = df[(df['shuffling'] == 'non-shuffled')].sort_values('grid_seed')["decay_tau"]
b = df[(df['shuffling'] == 'shuffled')].sort_values('grid_seed')["decay_tau"]
print(ttest_rel(a, b, alternative='less'))

formula = 'decay_tau ~ C(network) + C(shuffling) + C(network):C(shuffling)'
model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2)

aovrm = AnovaRM(df, 'decay_tau', 'grid_seed', within=['shuffling', 'network'])
res = aovrm.fit()

print(res)

shuffled = df[df['shuffling'] == 'shuffled'].sort_values(['grid_seed', 'network'])
non_shuffled = df[df['shuffling'] == 'non-shuffled'].sort_values(['grid_seed', 'network'])

non_shuffled['decay_tau_norm'] = non_shuffled['decay_tau'].to_numpy() / shuffled['decay_tau'].to_numpy()

shuffled_full = []
shuffled_adjusted = []
nonshuffled_full = []
nonshuffled_adjusted = []

for k in files.keys():
    if 'non-shuffled' in k:
        if 'adjusted' in k:
            nonshuffled_adjusted.append(files[k][1])
        else:
            nonshuffled_full.append(files[k][1])
    else:
        if 'adjusted' in k:
            shuffled_adjusted.append(files[k][1])
        else:
            shuffled_full.append(files[k][1])
shuffled_full = np.array(shuffled_full)
shuffled_adjusted = np.array(shuffled_adjusted)
nonshuffled_full = np.array(nonshuffled_full)
nonshuffled_adjusted = np.array(nonshuffled_adjusted)

"""Plotting"""
fig = plt.figure(figsize=(8, 11), constrained_layout=True)
gs = fig.add_gridspec(6,6)

tt_ex_path = db_path = os.path.join(
    dirname, 'data', 'tempotron_membrane_example.npy')
tempotron_example = np.load(tt_ex_path)


ax1 = fig.add_subplot(gs[0:2,0:4])
tempotron_threshold = 24.757166075118896
ax1.plot(tempotron_example[0,:], tempotron_example[1,:])
ax1.plot(tempotron_example[2,:], tempotron_example[3,:])
ax1.hlines(tempotron_threshold, 0, 2000, color='k')
ax1.legend(("Pre learning", "Post learning", "Threshold"))
ax1.set_ylabel("Tempotron Output (AU)")
ax1.set_xlabel("Time (ms)")


ax2 = fig.add_subplot(gs[0:2,4:])
x = np.arange(200)
legend_names = ["Full", "NoFB Adjusted"]
colors = ['#716969', '#a09573']
for idx, curr_data in enumerate([nonshuffled_full, nonshuffled_adjusted]):
    print(curr_data.shape[0])
    y = (curr_data / curr_data[:,0].mean()).mean(axis=0)
    y_err = (curr_data / curr_data[:,0].mean()).std(axis=0) / np.sqrt(curr_data.shape[0])
    p = ax2.plot(x, y, label=legend_names[idx], color=colors[idx])
    f = ax2.fill_between(x,
                     y-y_err,
                     y+y_err,
                     alpha = 0.5,
                     color=colors[idx])
ax2.set_ylabel("Average normalized loss")
ax2.set_xlabel("Epoch")
ax2.legend()


granule_pal = {'non-shuffled': '#09316c', 'shuffled': '#0a9396'}
ax3 = fig.add_subplot(gs[2:4,0:3])
sns.boxplot(data=df, x='network', y='decay_tau', hue='shuffling', ax=ax3, palette=granule_pal)
plt.ylabel("Decay tau (epochs)")

sns.boxplot(x='network', y='decay_tau', hue='shuffling', ax=ax3, zorder=1,
            data=df, linewidth=0.5, fliersize=1, palette=granule_pal)
loc = 0.4
tunings = ['full', 'no-feedback-adjusted']
grid_seeds = list(np.arange(1,31,1))
for grid in grid_seeds:
    tune_idx = 0
    for tuning in tunings:
        ns_data = df.loc[(df['shuffling']=='non-shuffled')
                                      &
                                      (df['network']==tuning)
                                      &
                                      (df['grid_seed']==grid)]
        s_data = df.loc[(df['shuffling']=='shuffled')
                                      &
                                      (df['network']==tuning)
                                      &
                                      (df['grid_seed']==grid)]
        if (np.array(ns_data['decay_tau']).size > 0 and
            np.array(s_data['decay_tau']).size > 0):
            ns_info = np.array(ns_data['decay_tau'])[0]
            s_info = np.array(s_data['decay_tau'])[0]   
            sns.lineplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax3,
                         y = [ns_info, s_info], color='k', linewidth = 0.2)
            sns.scatterplot(x= [-loc/2+tune_idx, loc/2+tune_idx], ax=ax3,
                          y = [ns_info, s_info], color='k', s = 10)
        tune_idx+=1

ax3.set_xticklabels(['full', 'no fb'], rotation=60)
handles, labels = ax3.get_legend_handles_labels()
ax3.legend(handles[:2], labels[:2], loc='upper left', title=None, prop={'size': 8})
ax3.set_ylabel('1 / decay tau (epochs)')

ax4 = fig.add_subplot(gs[2:4,3:])
grid_pal = {'full': '#716969', 'no-feedback-adjusted': '#a09573'}

sns.boxplot(data=non_shuffled, x='network', y='decay_tau_norm', ax=ax4, palette=grid_pal)

sns.boxplot(x='network', y='decay_tau_norm', ax=ax4, zorder=1,
            data=non_shuffled, linewidth=0.5, fliersize=1, palette=grid_pal)
loc = 0.4
tunings = ['full', 'no fb']
grid_seeds = list(np.arange(1,31,1))
for grid in grid_seeds:
    tune_idx = 0

    ns_data = non_shuffled.loc[(non_shuffled['network']=='full')
                                  &
                                  (df['grid_seed']==grid)]
    s_data = non_shuffled.loc[(df['network']=='no-feedback-adjusted')
                                  &
                                  (df['grid_seed']==grid)]
    if (np.array(ns_data['decay_tau_norm']).size > 0 and
        np.array(s_data['decay_tau_norm']).size > 0):
        ns_info = np.array(ns_data['decay_tau_norm'])[0]
        s_info = np.array(s_data['decay_tau_norm'])[0]   
        sns.lineplot(x= ['full', 'no-feedback-adjusted'], ax=ax4,
                     y = [ns_info, s_info], color='k', linewidth = 0.2)
        sns.scatterplot(x= ['full', 'no-feedback-adjusted'], ax=ax4,
                      y = [ns_info, s_info], color='k', s = 10)

ax4.set_xticklabels(['full', 'no fb'], rotation=60)
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles[:2], labels[:2], loc='upper left', title=None, prop={'size': 8})
ax4.set_ylabel('norm decay tau')

# ax4 = fig.add_subplot(gs[1,1])
# sns.boxplot(data=df, x='network', y='mean_rate', hue='shuffling',ax=ax3)
# plt.ylabel("Average rate (Hz)")