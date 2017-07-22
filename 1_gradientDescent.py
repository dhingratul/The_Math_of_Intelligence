#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 13:42:03 2017

@author: dhingratul

Gradient Descent Algorithm, trying to fit y = mx + b, optimize
for m, b independently
"""
import numpy as np


def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, LR):
    b_g = 0
    m_g = 0
    n = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_g += -(2 / n) * (y - ((m_current * x) + b_current))
        m_g += -(2 / n) * x * (y - ((m_current * x) + b_current))
    b_upd = b_current - (LR * b_g)
    m_upd = m_current - (LR * m_g)
    return [b_upd, m_upd]


def gradient_descent(points, b_start, m_start, LR, iters):
    b = b_start
    m = m_start
    for i in range(iters):
        b, m = step_gradient(b, m, np.array(points), LR)
    return [b, m]


# Driver Program
points = np.genfromtxt('1_data.csv', delimiter=',')
LR = 0.0001
m_init = 0
b_init = 0
iters = 10000
print("Starting gradient descent at b = {0}, m = {1},error = {2}"
      .format(b_init, m_init,
              compute_error_for_line_given_points(b_init, m_init, points)))
[b, m] = gradient_descent(points, b_init, m_init, LR, iters)
print("After {0} iterations b = {1}, m = {2}, error = {3}"
      .format(iters, b, m, compute_error_for_line_given_points(b, m, points)))
