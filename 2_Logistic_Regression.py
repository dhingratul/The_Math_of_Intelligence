# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:26:00 2017

@author: dhingratul
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings


def sigmoid(x):
    """
    Squashes the function b/w 0-1 -- probability
    """
    return (1 / 1 + np.exp(-x))


np.random.seed(0)
tol = 1e-8  # Convergence tolerance
lam = None  # L-2 regularizer
max_iter = 20
# Create data
cov = 0.95  # Covariance b/w height and weight(x and z) v = b.p
n = 1000  # Size of data
sigma = 1
# Model Settings -- log(p_dis) = log(odds(p_dis)) = beta_0 + beta_1 * x_1 + ...
beta_x, beta_z, beta_v = -4, .9, 1  # True beta coeff
var_x, var_z, var_v = 1, 1, 4 # variances of inputs
## the model specification you want to fit
formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'
