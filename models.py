# -*- coding: utf-8 -*-
"""
Created on Thu May 11 02:04:07 2017

@author: ryuhei
"""

import chainer
import chainer.functions as F
from links.stateless_simple_act import StatelessSimpleACT


class ACTNet(chainer.Chain):
    def __init__(self, in_size, s_size, out_size,
                 epsilon=0.01, max_ponder_steps=100):
        super(ACTNet, self).__init__(
            act=StatelessSimpleACT(in_size, s_size, out_size,
                                   epsilon, max_ponder_steps)
        )

    def __call__(self, x_seq):
        batch_size, seq_len, in_size = x_seq.shape
        y_seq = []
        ponder_cost_seq = []

        s_t = self.xp.zeros((batch_size, self.act.s_size), dtype=x_seq.dtype)
        for x_t in F.separate(x_seq, axis=1):
            y_t, s_t, ponder_cost_t = self.act(x_t, s_t)
            y_seq.append(y_t)
            ponder_cost_seq.append(ponder_cost_t)
        y_seq = F.stack(y_seq, axis=1)
        ponder_cost = sum(ponder_cost_seq)
        return y_seq, ponder_cost
