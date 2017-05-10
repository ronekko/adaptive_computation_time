# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:23:42 2017

@author: ryuhei
"""

import numpy as np
import chainer


def generate_parity_data(
        batch_size=128, max_bits=64, min_bits=1, use_gpu=False):
    x = generate_sequences(batch_size, max_bits, min_bits)
    y = 1 - (np.count_nonzero(x, 2).astype(np.int32) % 2)
    if use_gpu:
        x = chainer.cuda.to_gpu(x)
        y = chainer.cuda.to_gpu(y)
    return x, y


def generate_sequences(batch_size=128, max_bits=64, min_bits=1):
    assert batch_size >= 1
    assert min_bits >= 1
    assert max_bits >= min_bits

    x = np.random.choice((-1, 1), (batch_size, 1, max_bits)).astype(np.float32)
    bits_lengths = np.random.randint(min_bits, max_bits + 1, batch_size)
    for i, bits_length in enumerate(bits_lengths):
        x[i, 0, bits_length:] = 0
    return x


if __name__ == '__main__':
    pass