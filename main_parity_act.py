# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:26:20 2017

@author: ryuhei
"""


import numpy as np
import matplotlib.pyplot as plt

import chainer
import chainer.functions as F
import chainer.links as L

from datasets.parity import generate_parity_data


class ACTRNN(chainer.Chain):
    def __init__(self, in_size, s_size, epsilon=0.01, max_ponder_steps=100):
        super(ACTRNN, self).__init__(
            l_xs=L.Linear(in_size + 1, s_size),
            l_ss=L.Linear(s_size, s_size),
            l_sh=L.Linear(s_size, 1, bias=5),
            l_sy=L.Linear(s_size, 1))
        self.in_size = in_size
        self.s_size = s_size
        self.epsilon = epsilon
        self.max_ponder_steps = max_ponder_steps

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x)
        batch_size, seq_len, dim_features = x.shape

        ponder_losses = []
        y = []
        s_tm1 = xp.zeros((batch_size, self.s_size), dtype=x.dtype)  # s_{t-1}

        for t in range(seq_len):
            x_t = x[:, t]
            n = 1
            x_t_1 = xp.hstack((x_t, xp.ones((batch_size, 1), x.dtype)))
            s_t_1 = F.tanh(self.l_xs(x_t_1) + self.l_ss(s_tm1))
            h_t_1 = F.sigmoid(self.l_sh(s_t_1))
            y_t_1 = self.l_sy(s_t_1)

            nt = xp.ones((batch_size, 1))
            p_t_1 = F.where(h_t_1.data > 1.0 - self.epsilon,
                            h_t_1 + (1.0 - h_t_1),  # i.e. R(t) = 1.0
                            h_t_1)
            c_t = p_t_1  # cumlative sum of p_t_n
            not_halted = c_t.data < 1.0

            s_t_ns = [s_t_1]
            p_t_ns = [p_t_1]
            y_t_ns = [y_t_1]

            x_t_n = xp.hstack((x_t, xp.zeros((batch_size, 1), x.dtype)))
            x_t_n = self.l_xs(x_t_n)  # precompute
            s_t_n = s_t_1
            while xp.any(not_halted) and n < self.max_ponder_steps:
                nt += not_halted.astype(xp.int)
                s_t_n = F.tanh(x_t_n + self.l_ss(s_t_n))
                h_t_n = F.sigmoid(self.l_sh(s_t_n))
                if n <= self.max_ponder_steps:
                    halt = c_t.data + h_t_n.data > 1 - self.epsilon
                else:
                    halt = xp.ones(h_t_n.data.shape, np.bool)
                rt = h_t_n - ((h_t_n + c_t) - 1.0)
                p_t_n = F.where(not_halted,
                                F.where(halt,
                                        rt,
                                        h_t_n),
                                xp.zeros((batch_size, 1), xp.float32))
                y_t_n = self.l_sy(s_t_n)

                s_t_ns.append(s_t_n)
                p_t_ns.append(p_t_n)
                y_t_ns.append(y_t_n)
                c_t = c_t + p_t_n
                not_halted = c_t.data < 1.0
                n += 1

            print(n)
            p_t_ns = F.concat(p_t_ns)
            s_t_ns = F.dstack(s_t_ns)
            y_t_ns = F.dstack(y_t_ns)
            s_tm1 = F.batch_matmul(s_t_ns, p_t_ns).reshape(
                batch_size, self.s_size)
            y_t = F.batch_matmul(y_t_ns, p_t_ns).reshape(
                batch_size, 1)
            y.append(y_t)

            ponder_losses.append(ponder_loss(p_t_ns))

        y = F.concat(y)
        loss = sum(ponder_losses)
        return y, loss


class PonderLoss(chainer.Function):
    def forward(self, inputs):
        p_t_ns = inputs[0]
        xp = chainer.cuda.get_array_module(p_t_ns)
        batch_size, max_time_steps = p_t_ns.shape
        nt = p_t_ns.argmin(1)
        n_nonzero = xp.count_nonzero(p_t_ns, axis=1)
        nt[n_nonzero == 0] = max_time_steps

        rt = p_t_ns[xp.arange(batch_size), nt - 1]
        loss = xp.sum(nt + rt)
        self._nt = nt
        return xp.array(loss, dtype=np.float32),

    def backward(self, inputs, gy):
        p_t_ns = inputs[0]
        xp = chainer.cuda.get_array_module(p_t_ns)
        gp_t_ns = xp.zeros_like(p_t_ns)
        gp_t_ns[xp.arange(batch_size), self._nt - 1] = gy
        return gp_t_ns,


def ponder_loss(p_t_ns):
    return PonderLoss()(p_t_ns)


if __name__ == '__main__':
    max_bit_len = 10
    state_size = 128
    batch_size = 128
    learning_rate = 1e-4
    time_penalty = 0.0001  # hyperparameter "tau"

    model = ACTRNN(max_bit_len, state_size)
    optimizer = chainer.optimizers.Adam(learning_rate)
    optimizer.setup(model)
    optimizer.use_cleargrads(True)

    loss_log = []
    for i in range(100000):
        print('{}:'.format(i), end=' ')
        x, t = generate_parity_data(batch_size=batch_size,
                                    max_bits=max_bit_len)
        y, ponder_loss_value = model(x)
        loss = F.sigmoid_cross_entropy(y, t) + time_penalty * ponder_loss_value
        model.cleargrads()
        loss.backward()
        loss_log.append(loss.data)
        optimizer.update()

        accuracy = F.binary_accuracy(y, t)

        print('acc:', accuracy.data)
        print()

        if i % 10 == 0:
            plt.plot(loss_log)
            plt.show()
