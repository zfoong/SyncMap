# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 23:54:12 2021

@author: zfoong
"""

import numpy as np
import random

random.seed(111)
np.random.seed(111)

class MC_gen():
    def __init__(self):
        self.N = 10 # Possible state
        self.possible_trans_rate = 0.3 # Rate of element in TP being nonzero        
        self.data_count = 10000 # Total of sequence generated
        self.maxlen = 20 # Maximum length of each sequence
        self.init_state = [0, 3] # Initial state
        self.terminal_state = [1] # End state
        self.data = []
    
    def TP_matrix_gen(self, N, rate):
        tp = np.random.binomial(1, rate, [N, N]) # Transition Probability Matrix
        tp = tp.astype('float64')
        for i, j in enumerate(tp):
            ones = j.nonzero()[0]
            rand_to_1 = np.random.dirichlet(np.ones(len(ones)),size=1)
            for k, l in enumerate(ones):
                tp[i][l] = rand_to_1[0][k]
        return tp
    
    def seq_gen(self, init_state, terminal_state, maxlen):
        current_state = np.random.choice(init_state, replace=True)
        d = [current_state]
        for _ in range(maxlen):
            if np.sum(self.TP[current_state]) == 0 or current_state in terminal_state:
                break
            next_state = np.random.choice(self.N, replace=True, p=self.TP[current_state])
            d.append(next_state)
            current_state = next_state
        return d
        
    def data_gen(self):
        self.TP = self.TP_matrix_gen(self.N, self.possible_trans_rate)
        self.data.append([self.seq_gen(self.init_state, 
                                       self.terminal_state, 
                                       self.maxlen) 
                          for _ in range(self.data_count)])
        return self.data


