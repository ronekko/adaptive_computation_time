# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:26:20 2017

@author: ryuhei

As written in the paper, this does not work well.
"""


import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L

from datasets.parity import generate_parity_data


class RNN(chainer.Chain):
    def __init__(self, in_size, s_size, out_size=1):
        super(RNN, self).__init__(
            l_xs=L.Linear(in_size, s_size),
            l_ss=L.Linear(s_size, s_size),
            l_sy=L.Linear(s_size, out_size))
        self.in_size = in_size
        self.s_size = s_size

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x)
        batch_size, seq_len, dim_features = x.shape
        s = xp.zeros((batch_size, self.s_size), dtype=x.dtype)

        for t in range(seq_len):
            x_t = x[:, t]
            s = F.tanh(self.l_xs(x_t) + self.l_ss(s))

        y = self.l_sy(s)
        return y


if __name__ == '__main__':
    max_bit_len = 64
    state_size = 128
    batch_size = 128
    learning_rate = 1e-4

    model = RNN(max_bit_len, state_size)
    optimizer = chainer.optimizers.Adam(learning_rate)
    optimizer.setup(model)
    optimizer.use_cleargrads(True)

    for i in range(10000):
        x, t = generate_parity_data(batch_size=batch_size,
                                    max_bits=max_bit_len)
        y = model(x)
        loss = F.sigmoid_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        accuracy = F.binary_accuracy(y, t)
        print(i, accuracy.data)
