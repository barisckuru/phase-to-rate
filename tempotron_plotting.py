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
to_exclude = ((df_mean_rates['mean_rate'] < 0.2) | (df_mean_rates['mean_rate'] > 0.3))
grid_seeds_excluded = df_mean_rates[to_exclude]['grid_seed'].unique()
df.drop(df[df['grid_seed'].isin(grid_seeds_excluded)].index, inplace=True)

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
    key = f"{fid}_{df.loc[idx]['shuffling']}"
    file_path = db_path = os.path.join(
        dirname, 'data', 'arrays', fid+'.npy')
    curr_data = np.load(file_path)
    popt, pcov = scipy.optimize.curve_fit(exp_decay, np.arange(200), curr_data[1,:], p0=[100, 300])
    fit_tau, fit_init = popt
    files[key] = curr_data
    taus.append(fit_tau)
df['decay_tau'] = taus

mean_shuffled = np.zeros(200)
mean_nonshuffled = np.zeros(200)
for k in files.keys():
    if 'non-shuffled' in k:
        mean_nonshuffled += files[k][1]
    else:
        mean_shuffled += files[k][1]
mean_shuffled = mean_shuffled / 30
mean_nonshuffled = mean_nonshuffled / 30

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