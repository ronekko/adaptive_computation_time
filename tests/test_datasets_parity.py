# -*- coding: utf-8 -*-
"""
Created on Fri May  5 13:25:14 2017

@author: ryuhei
"""

import unittest

import numpy as np

from ..datasets.parity import generate_parity_data, generate_sequences


class TestGenerateParityData(unittest.TestCase):
    def test_generate_sequences(self):
        self.check_generate_sequences(1000, 64, 1)
        self.check_generate_sequences(1100, 30, 5)

    def check_generate_sequences(self, batch_size, max_bits, min_bits):
        xs = generate_sequences(batch_size, max_bits, min_bits)
        self.assertEqual(xs.shape, (batch_size, 1, max_bits))
        bits_lengths = np.sum(xs != 0, axis=2)
        self.assertEqual(bits_lengths.min(), min_bits)
        self.assertEqual(bits_lengths.max(), max_bits)
        np.testing.assert_array_equal(
            np.unique(bits_lengths), np.arange(min_bits, max_bits + 1))


if __name__ == '__main__':
    unittest.main(__name__)
