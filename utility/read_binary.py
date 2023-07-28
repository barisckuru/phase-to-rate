# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:59:42 2022

@author: Daniel
"""

import numpy as np
import os

def read_binary(file_path, input_type=np.int16):
    return np.fromfile(file_path, dtype=input_type)
    
if __name__ == '__main__':
    example_file = r'D:\mizuseki\extracted\ec016.665.tar\ec016.665\ec016.41\ec016.665\ec016.665.eeg'
    test = read_binary(example_file)