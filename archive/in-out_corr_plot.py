#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:45:03 2020

@author: bariskuru
"""

import seaborn as sns, pandas as pd, numpy as np
import os
import matplotlib.pyplot as plt

# seed_order = (100,200
#               101,200
#               102,200
#               100,201
#               101,201
#               102,201
#               100,202
#               101,202
#               102,202)

result_list = []
granule_out = []
input_grid = []
seeds = []
input_corrs = []
output_corrs = []

bin_start = 2000
bin_end = 2500

dir_fold = '/Users/bariskuru/output_arrays/'

for file in sorted(os.listdir(dir_fold)):
    if file.endswith(".npz"):
        print(file)
        load = np.load(os.path.join(dir_fold, file), allow_pickle=True)
        result_list.append(load)
        granule_out.append(load['granule_output'])
        input_grid.append(load['input_grid_out'])
        seeds.append(load['seeds'])
        
        
def ct_a_bin(arr, bin_start, bin_end):
    counts = np.empty((arr.shape[0], arr.shape[1]))
    for index, value in np.ndenumerate(arr):
        # print(index)
        counts[index] = ((value > bin_start) & (value< bin_end)).sum()
    return counts

def pearson_r_bin(x, y, bin_start, bin_end):
    x_ct_bin = ct_a_bin(x, bin_start, bin_end)
    y_ct_bin = ct_a_bin(y, bin_start, bin_end)
    #corr mat is doubled in each axis since it is 2d*2d
    corr_mat = np.corrcoef(x_ct_bin, y_ct_bin, rowvar=False) 
    #slice out the 1 of 4 identical mat
    corr_mat = corr_mat[int(corr_mat.shape[0]/2):, :int(corr_mat.shape[0]/2)] 
    # indices in upper triangle
    iu =np.triu_indices(int(corr_mat.shape[0]), k=0)
    # corr arr is the values vectorized 
    diag_low = corr_mat[iu]
    diag = corr_mat.diagonal()
    return diag, diag_low        

for in_grid, out_gra in zip(input_grid, granule_out):
    input_corrs.append(pearson_r_bin(in_grid,in_grid, bin_start, bin_end)[1])
    output_corrs.append(pearson_r_bin(out_gra,out_gra, bin_start, bin_end)[1])
    
    
a = np.linspace(-0.2, 1, 100)
b = np.linspace(-0.2, 1, 100)

sns.set(context='paper',style='whitegrid',palette='colorblind',font='Arial',font_scale=1.5,color_codes=True)

# df = pd.DataFrame({'Rin':input_corrs, 'Rout':output_corrs})

# sns.lmplot(x="Rin", y="Rout", data=df,
#            markers=["o", "x"], palette="Set1");


# plt.plot(input_corrs[0],output_corrs[0], 'o') 
# plt.plot(input_corrs[1],output_corrs[1], 'o')   
# plt.plot(input_corrs[2],output_corrs[2], 'o')


# plt.plot(input_corrs[3],output_corrs[3], 'o') 
# plt.plot(input_corrs[4],output_corrs[4], 'o')   
# plt.plot(input_corrs[5],output_corrs[5], 'o')
# plt.plot(input_corrs[6],output_corrs[6], 'o') 
# plt.plot(input_corrs[7],output_corrs[7], 'o')   
# plt.plot(input_corrs[8],output_corrs[8], 'o')  

# plt.legend()
# plt.title('Rin vs Rout for one seed set')
# plt.xlabel('Rin')
# plt.ylabel('Rout')
# plt.plot(a,b)

same_input = pearson_r_bin(input_grid[3], input_grid[9], bin_start, bin_end)[1]
same_input_out = pearson_r_bin(granule_out[3], granule_out[9], bin_start, bin_end)[1]
plt.figure()
plt.plot(same_input,same_input_out, 'ro', label='seed1=100 vs seed1=101') 
plt.legend()
plt.title('Rin vs Rout\nonly seed 1 is different')
plt.xlabel('Rin')
plt.ylabel('Rout')
plt.plot(a,b)

# import seaborn as sns
# sns.set()

# # Load the iris dataset
# iris = sns.load_dataset("iris")


