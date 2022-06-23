# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 13:00:34 2022

@author: Daniel
"""

import os

dire = r'C:\Users\Daniel\repos\phase_coding\data\tempotron\batch5'

file_names = os.listdir(dire)

for f in file_names:
    extension = f.split('.')[-1]
    f_split = f.split('_')
    if len(f_split) == 8:
        continue
    new_f = f_split[:4] + f_split[5:-1]
    new_f = '_'.join(new_f) + '.' + extension
    os.rename(os.path.join(dire, f), os.path.join(dire,new_f))