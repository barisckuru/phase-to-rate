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

dirname = os.path.dirname(__file__)
db_path = os.path.join(
    dirname, 'data', 'tempotron_thresholds_mean.db')
con = sqlite3.connect(db_path)
cur = con.cursor()
# rows = cur.execute(f"""SELECT * FROM tempotron_run WHERE cell_type='granule_spikes' AND learning_rate=0.001 AND epochs=100""")
rows = cur.execute(f"""SELECT * FROM tempotron_run WHERE n_cells=2000 AND (tempotron_seed=91 OR tempotron_seed=95)""")
data = []
for x in rows:
    data.append(x)
    print(x)

def exp_decay(x, tau, init):
    return init*np.e**(-x/tau)

df = pd.DataFrame(data=data, columns=['tempotron_seed', 'epochs', 'time', 'Vrest', 'tau',
             'tau_s', 'threshold', 'learning_rate', 'n_cells',
             'trajectory_one', 'trajectory_two', 'pre_accuracy',
             'trained_accuracy', 'pre_loss', 
             'trained_loss', 'delta_loss', 'distance', 'grid_seed', 'duration', 
             'shuffling', 'network', 'cell_type', 'file_id'])

df['delta_learning'] = df['trained_accuracy'] - df['pre_accuracy']
# df['learning_index'] = (df['pre_accuracy']/(df['trained_accuracy'] - df['pre_accuracy']))**-1
df['learning_index'] = (df['trained_accuracy'] - df['pre_accuracy'])/df['pre_accuracy']

df = df.drop_duplicates()

rate_path = os.path.join(dirname, 'data', 'mean_rates.csv')
df_mean_rates = pd.read_csv(rate_path, index_col=0)
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

plt.figure()
sns.scatterplot(data=df, x='pre_accuracy', y='trained_accuracy', hue='shuffling')

a = df[(df['shuffling'] == 'non-shuffled')].sort_values('grid_seed')["delta_learning"]
b = df[(df['shuffling'] == 'shuffled')].sort_values('grid_seed')["delta_learning"]
print(ttest_rel(a, b, alternative='greater'))

a = df[(df['shuffling'] == 'non-shuffled')].sort_values('grid_seed')["learning_index"]
b = df[(df['shuffling'] == 'shuffled')].sort_values('grid_seed')["learning_index"]
print(ttest_rel(a, b, alternative='greater'))


a = df[(df['shuffling'] == 'non-shuffled')].sort_values('grid_seed')["pre_accuracy"]
b = df[(df['shuffling'] == 'shuffled')].sort_values('grid_seed')["pre_accuracy"]
print(ttest_rel(a, b, alternative='two-sided'))

print(df.groupby("shuffling").describe()['delta_learning'])

print(df.groupby("shuffling").describe()['learning_index'])

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
df['decay_tau'] = taus

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

x = np.arange(200)
fig, ax = plt.subplots()
plot_pairs = []
legend_names = ["shuffled_full", "shuffled_adjusted", "nonshuffled_full", "nonshuffled_adjusted"]
for idx, curr_data in enumerate([shuffled_full, shuffled_adjusted, nonshuffled_full, nonshuffled_adjusted]):
    print(curr_data.shape[0])
    y = (curr_data / curr_data[:,0].mean()).mean(axis=0)
    y_err = (curr_data / curr_data[:,0].mean()).std(axis=0) / np.sqrt(curr_data.shape[0])
    p = ax.plot(x, y, label=legend_names[idx])
    f = ax.fill_between(x,
                     y-y_err,
                     y+y_err,
                     alpha = 0.5)
    plot_pairs.append((p, f))
    
ax.legend()
ax.set_xlabel("Epoch")
ax.set_ylabel("Normalized loss")

a = df[(df['shuffling'] == 'non-shuffled')].sort_values('grid_seed')["decay_tau"]
b = df[(df['shuffling'] == 'shuffled')].sort_values('grid_seed')["decay_tau"]
print(ttest_rel(a, b, alternative='less'))

plt.figure()
sns.boxplot(data=df, x='network', y='decay_tau', hue='shuffling')
plt.ylabel("Decay tau (epochs)")

formula = 'decay_tau ~ C(network) + C(shuffling) + C(network):C(shuffling)'
model = ols(formula, df).fit()
aov_table = anova_lm(model, typ=2)

aovrm = AnovaRM(df, 'decay_tau', 'grid_seed', within=['shuffling', 'network'])
res = aovrm.fit()

print(res)

shuffled = df[df['shuffling'] == 'shuffled'].sort_values(['grid_seed', 'network'])
non_shuffled = df[df['shuffling'] == 'non-shuffled'].sort_values(['grid_seed', 'network'])

non_shuffled['decay_tau_norm'] = shuffled['decay_tau'].to_numpy() / non_shuffled['decay_tau'].to_numpy()

plt.figure()
sns.boxplot(data=non_shuffled, x='network', y='decay_tau_norm')
plt.ylabel("Decay tau shuffled/non-shuffled")

a = df[(df['network'] == 'full')].sort_values('grid_seed')["decay_tau"]
b = df[(df['network'] == 'no-feedback-adjusted')].sort_values('grid_seed')["decay_tau"]
print(ttest_rel(a, b, alternative='less'))

plt.figure()
sns.boxplot(data=df, x='network', y='mean_rate', hue='shuffling')
plt.ylabel("mean_rate (Hz)")

plt.figure()
sns.scatterplot(data=df, x='decay_tau', y='mean_rate', hue='network')

plt.figure()
sns.scatterplot(data=df, x='mean_rate', y='decay_tau', hue='network')

plt.figure()
sns.scatterplot(data=non_shuffled, x='decay_tau_norm', y='mean_rate', hue='network')
plt.ylabel("mean_rate (Hz)")


