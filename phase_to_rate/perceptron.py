#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:01:35 2021

@author: baris
"""

import torch
import torch.nn as nn
from torch import optim
import numpy as np


# labels maker
# output for training the network, 5 for each trajectory


def _label(n_poiss):
    a = np.tile([1, 0], (n_poiss, 1))
    b = np.tile([0, 1], (n_poiss, 1))
    labels = np.vstack((a, b))
    labels = torch.FloatTensor(labels)
    out_len = labels.shape[1]
    return labels, out_len


# BUILD THE NETWORK


class _Net(nn.Module):
    def __init__(self, n_inp, n_out):
        super(_Net, self).__init__()
        self.fc1 = nn.Linear(n_inp, n_out)

    def forward(self, x):
        y = torch.sigmoid(self.fc1(x))
        return y


# TRAIN THE NETWORK


def _train_net(net, train_data, labels, n_iter=1000, lr=1e-4):
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
            print(f"iteration {i + 1}/{n_iter} | loss: {loss.item():.3f}")

    return track_loss, out


def run_perceptron(neural_code, grid_seed, learning_rate=1e-4,
                   n_iter=10000, threshold=0.2):
    """

    Generate and run the perceptron network.

    Parameters
    ----------
    neural_code : numpy array
        Neural code generated from a cell population.
    grid_seed : TYPE
        Grid cell population generation seed,
        modified and used to seed perceptron network as well.
    learning_rate : float
        Learning rate in the perceptron network. The default is 1e-4.
    n_iter : int
        Number of epochs for perceptron learning. The default is 10000.
    threshold : float
        Threshold considered sufficient for learning,
        which the loss function reaches. The default is 0.2.

    Returns
    -------
    Threshold crossing points and loss value in each epoch.
    """
    neural_code = np.transpose(neural_code, (1, 0))
    n_sample, inp_len = neural_code.shape
    n_poiss = int(n_sample / 2)
    perc_seed = grid_seed + 100
    labels, out_len = _label(n_poiss)
    # Convert into tensor
    neural_code = torch.FloatTensor(neural_code)
    torch.manual_seed(perc_seed)
    net_neural = _Net(inp_len, out_len)
    train_loss, _ = _train_net(net_neural, neural_code,
                               labels, n_iter=n_iter, lr=learning_rate)
    # threshold crossing points
    th_cross = np.argmax(np.array(train_loss) < threshold)
    return th_cross, train_loss
