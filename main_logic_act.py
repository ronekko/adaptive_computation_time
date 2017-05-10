# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:59:41 2017

@author: sakurai
"""


import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F

from models.act_rnn import ACTRNN
from datasets.logic import LogicDataset


if __name__ == '__main__':
    use_gpu = False
    seq_len = 3
    max_ops = 3
    min_ops = 1
    n_gates = 10
    dim_vector = 100
    state_size = 128
    batch_size = 16
    learning_rate = 1e-4
    time_penalty = 0.01  # hyperparameter "tau"

    dataset = LogicDataset(batch_size, seq_len)
    dim_vector = dataset.dim_vector

    model = ACTRNN(dim_vector, state_size, 1)
    if use_gpu:
        model.to_gpu()
    optimizer = chainer.optimizers.Adam(learning_rate)
    optimizer.setup(model)
    optimizer.use_cleargrads(True)

    loss_log = []
    for i in range(1000000):
        print('{}:'.format(i), end=' ')
        x, t = next(dataset)

        y, ponder_cost = model(x)
        loss = F.sigmoid_cross_entropy(y, t) + time_penalty * ponder_cost
        model.cleargrads()
        loss.backward()
        loss_log.append(chainer.cuda.to_cpu(loss.data))
        optimizer.update()

        accuracy = F.binary_accuracy(y, t)

        print('acc:', accuracy.data)

        if i % 50 == 0:
            plt.plot(loss_log, '.', markersize=1)
            plt.grid()
            plt.show()
