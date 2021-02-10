# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 21:28:21 2021

@author: zfoong
"""

import cv2     
import math  
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  
import pandas as pd
import numpy as np   
from tqdm import tqdm
import os
import random
from utils import MC_gen
from sklearn.cluster import DBSCAN

random.seed(111)

def input_encoding(S_t, X_t, decay_rate):
    return X_t * np.exp(-decay_rate) + S_t
    
def weight_update(x, w):
    PS_NS = x > 0.1
    PS_NS.astype('float64')
    PS = np.count_nonzero(PS_NS)
    NS = len(PS_NS) - PS
    if PS > 1 and NS > 1:
        PS_NS = PS_NS.reshape((-1,1))
        cp = np.sum(PS_NS * w) / PS
        cn = np.sum((1 - PS_NS) * w) / NS
        return w + a * ( PS_NS * cp + (1 - PS_NS) * cn) / np.linalg.norm(w-cp)
    return w

def mat_norm(a):
    a = a - a.mean(axis=0)
    a = a / np.abs(a).max(axis=0)
    return a

def organize(w):	
	return DBSCAN(eps=1, min_samples=2).fit_predict(w)

# Param Init
a = 0.01
decay_rate = 1

# Variable Init
k = 3 # Dimension variable
N = 10 # Input size
w = 2.0 * np.random.random((N, k)) - 1.0 # weight (-1, 1)

data = MC_gen().data_gen()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ims = []

for i, j in tqdm(enumerate(data[0])):
    x = np.zeros(N)
    for k, l in enumerate(j):
        s_t = np.zeros(N)
        s_t[l] = 1
        x = input_encoding(s_t, x, decay_rate)
        prev_w = w
        w = weight_update(x, w)
        w = mat_norm(w)
        # if not np.array_equal(prev_w, w):
        #     img = ax.scatter(w[:,0], w[:,1], w[:,2], color="blue")
        #     ims.append([img])
            
labels = organize(w)
cdict = {0: 'red', 1: 'blue', 2: 'green'}
for l in np.unique(labels):
    i = np.where(labels == l)
    img = ax.scatter(w[i,0], w[i,1], w[i,2], c = cdict[l])
plt.show()

print("Generating animation...")
# ani = animation.ArtistAnimation(fig, ims, interval=25, blit=True,
#                                 repeat_delay=100)
print("Generating animation completed!")
print("Saving animation as MP4")
# ani.save('movie.mp4')
print("Saving animation as MP4 completed!")
# plt.show()


