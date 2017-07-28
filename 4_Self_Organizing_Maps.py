#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 09:33:26 2017

@author: dhingratul
"""

import numpy as np
import math
from PIL import Image


def visualize(weights):
    im = Image.fromarray(weights.astype('uint8'), mode='RGB')
    im.format = 'JPG'
    im.show()


def dmatrix(weights, vector):
    return (np.sum(weights - vector) ** 2, 2)


def bmu(weights, vector):
    dist = dmatrix(weights, vector)
    return np.unravel_index(dist.argmin(), dist.shape)


def bmu_dist(weights, vector):
    x, y, rgb = weights.shape
    xi = np.arange(x).reshape(x, 1).repeat(y, 1)
    yi = np.arange(y).reshape(1, y).repeat(x, 0)
    return np.sum(
            (np.dstack((xi, yi)) - np.array(bmu(weights, vector))) ** 2, 2)


def hood_radius(map_radius, t_const, iteration):
        return map_radius * math.exp(-iteration/t_const)


def teach_row(weights, vector, i, dis_cut, dist):
    hood_radius_2 = hood_radius(map_radius, t_const, i) ** 2
    bmu_distance = bmu_dist(weights, vector).astype('float64')
    if dist is None:
        temp = hood_radius_2 - bmu_distance
    else:
        temp = dist ** 2 - bmu_distance
    influence = np.exp(-bmu_distance / (2 * hood_radius_2))
    if dis_cut:
        influence *= ((np.sign(temp) + 1) / 2)
    return np.expand_dims(influence, 2) * (vector - weights)


def teach(weights, t_set, t_iter, distance_cutoff=False, distance=None):
    for i in range(t_iter):
        for x in t_set:
            weights += teach_row(weights, x, i, distance_cutoff, distance)
    visualize(weights)

# Driver Program
x_size = 200
y_size = 200
trait_num = 3
t_iter = 100
t_step = 0.1
weights = np.random.randint(256,
                            size=(x_size, y_size, trait_num)).astype('float64')
map_radius = max(weights.shape)/2
t_const = t_iter/math.log(map_radius)
t_set = np.random.randint(256, size=(15, 3))
# t_set = np.array([[200, 0, 0], [0, 200, 0], [0, 0, 200], [120, 0, 100]])
teach(weights, t_set, t_iter)
