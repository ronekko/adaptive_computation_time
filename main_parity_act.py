# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:26:20 2017

@author: ryuhei
"""


import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F

from models import ACTNet
from datasets.parity import generate_parity_data


if __name__ == '__main__':
    use_gpu = False
    max_bit_len = 64
    state_size = 128
    batch_size = 128
    learning_rate = 1e-4
    time_penalty = 0.0001  # hyperparameter "tau"

    model = ACTNet(max_bit_len, state_size, 1)
    if use_gpu:
        model.to_gpu()
    optimizer = chainer.optimizers.Adam(learning_rate)
    optimizer.setup(model)
    optimizer.use_cleargrads(True)

    loss_log = []
    for i in range(1000000):
        print('{}:'.format(i), end=' ')
        x, t = generate_parity_data(
            batch_size=batch_size, max_bits=max_bit_len, use_gpu=use_gpu)

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
