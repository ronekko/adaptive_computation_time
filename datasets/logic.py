# -*- coding: utf-8 -*-
"""
Created on Fri May  5 22:36:16 2017

@author: ryuhei
"""

import numpy as np
import matplotlib.pyplot as plt


class LogicDataset(object):
    def __init__(self, batch_size=16, seq_len=3, max_ops=3, min_ops=1,
                 n_gate_types=10, n_slots_of_ops=10):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.max_ops = max_ops
        self.min_ops = min_ops
        self.n_gate_types = n_gate_types
        self.dim_vector = n_slots_of_ops * n_gate_types + 2
        self.table = self._create_table()

    def _create_table(self):
        table = np.zeros((self.n_gate_types, 2, 2), np.int32)
        table[0] = np.array([[1, 0], [0, 0]])
        table[1] = np.array([[0, 1], [0, 0]])
        table[2] = np.array([[0, 0], [1, 0]])
        table[3] = np.array([[0, 1], [1, 0]])
        table[4] = np.array([[1, 1], [1, 0]])
        table[5] = np.array([[0, 0], [0, 1]])
        table[6] = np.array([[1, 0], [0, 1]])
        table[7] = np.array([[1, 1], [0, 1]])
        table[8] = np.array([[1, 0], [1, 1]])
        table[9] = np.array([[0, 1], [1, 1]])
        return table

    def __next__(self):
        return self.next()

    def next(self):
        batch_size = self.batch_size
        # generate random number of operations
        x = np.zeros((batch_size, self.seq_len, self.dim_vector - 2),
                     np.float32)
        n_ops = np.random.randint(
            self.min_ops, self.max_ops, (batch_size, self.seq_len))
        flat_ops = np.random.randint(
            0, self.n_gate_types, n_ops.sum()).tolist()
        ops = []
        i = 0
        for x_i, n_ops_i in zip(x, n_ops):
            ops_i = []
            for x_it, n_ops_it in zip(x_i, n_ops_i):
                ops_it = flat_ops[i:i+n_ops_it]
                ops_i.append(ops_it)
                for x_ito in x_it.reshape((self.n_gate_types, -1))[:n_ops_it]:
                    x_ito[flat_ops[i]] = 1
                    i += 1
            ops.append(ops_i)

        # generate inputs and outputs
        b0s = []
        b1s = []
        ts = []
        for ops_i in ops:
            b0 = np.random.choice(2)
            for ops_it in ops_i:
                b1 = np.random.choice(2)
                b0s.append(b0)
                b1s.append(b1)
                for ops_ito in ops_it:
                    b2 = self.table[ops_ito, b0, b1]
                    b0 = b1
                    b1 = b2
                ts.append(b2)
                b0 = b2
        t = np.reshape(ts, (batch_size, self.seq_len, 1))
        b0 = np.reshape(b0s, (batch_size, self.seq_len, 1)).astype(np.float32)
        b1 = np.reshape(b1s, (batch_size, self.seq_len, 1)).astype(np.float32)
        x = np.dstack((b0, b1, x))
        return x, t

    def __iter__(self):
        return self.iter()

    def iter(self):
        while True:
            yield self.next()


if __name__ == '__main__':
    batch_size = 16
    n_gates = 10
    seq_len = 3

    # min and max number of operations (i.e. `B` in the paper)
    min_ops = 1
    max_ops = 3

    dataset = LogicDataset()
