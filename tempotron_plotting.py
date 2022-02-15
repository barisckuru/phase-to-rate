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

dirname = os.path.dirname(__file__)
db_path = os.path.join(
    dirname, 'data', 'tempotron.db')
con = sqlite3.connect(db_path)
cur = con.cursor()
# rows = cur.execute(f"""SELECT * FROM tempotron_run WHERE cell_type='granule_spikes' AND learning_rate=0.001 AND epochs=100""")
rows = cur.execute(f"""SELECT * FROM tempotron_run WHERE n_cells=400""")
data = []
for x in rows:
    data.append(x)
    print(x)

df = pd.DataFrame(data=data, columns=['tempotron_seed', 'epochs', 'time', 'Vrest', 'tau',
             'tau_s', 'threshold', 'learning_rate', 'n_cells',
             'trajectory_one', 'trajectory_two', 'pre_accuracy',
             'trained_accuracy', 'pre_loss', 
             'trained_loss', 'delta_loss', 'distance' 'grid_seed', 'duration', 
             'shuffling', 'network', 'cell_type'])

df['delta_learning'] = df['trained_accuracy'] - df['pre_accuracy']
# df['learning_index'] = (df['pre_accuracy']/(df['trained_accuracy'] - df['pre_accuracy']))**-1
df['learning_index'] = (df['trained_accuracy'] - df['pre_accuracy'])/df['pre_accuracy']

df = df.drop_duplicates()

"""
plt.figure()
sns.lineplot(
    data=df, x="distance", y="learning_index", hue='shuffling', err_style="bars", ci='sd', alpha=0.5)
plt.xlabel("distance")
plt.ylabel("Learning Index")

plt.figure()
sns.lineplot(
    data=df, x="distance", y="delta_learning", hue='shuffling', err_style="bars", ci='sd', alpha=0.5)
plt.xlabel("distance")
plt.ylabel("Delta Learning (% points)")
"""

plt.figure()
sns.boxplot(data=df, x='shuffling', y='delta_learning')
plt.ylabel("Delta Learning (% points)")

plt.figure()
sns.boxplot(data=df, x='shuffling', y='learning_index')
plt.ylabel("Learning Index")

plt.figure()
sns.boxplot(data=df, x='shuffling', y='pre_accuracy')
plt.ylabel("Pre Accuracy")

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

