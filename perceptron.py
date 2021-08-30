#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:01:35 2021

@author: baris
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim




# labels maker
# output for training the network, 5 for each trajectory

def label(n_poiss):
    a = np.tile([1, 0], (n_poiss,1))
    b = np.tile([0, 1], (n_poiss,1))
    labels = np.vstack((a,b))
    labels = torch.FloatTensor(labels) 
    out_len = labels.shape[1]
    return labels, out_len


#BUILD THE NETWORK

class Net(nn.Module):
    def __init__(self, n_inp, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)
    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        return y

#TRAIN THE NETWORK

def train_net(net, train_data, labels, n_iter=1000, lr=1e-4):
    optimizer = optim.SGD(net.parameters(), lr=lr)
    track_loss = []
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    for i in range(n_iter):
        out = net(train_data)
        loss = torch.sqrt(loss_fn(out, labels))
        
        # Compute gradients
        optimizer.zero_grad()
        loss.backward()
    
        # Update weights
        optimizer.step()
    
        # Store current value of loss
        # .item() needed to transform the tensor output of loss_fn to a scalar
        track_loss.append(loss.item())
        # Track progress
        if (i + 1) % (n_iter // 5) == 0:
          print(f'iteration {i + 1}/{n_iter} | loss: {loss.item():.3f}')

    return track_loss, out

