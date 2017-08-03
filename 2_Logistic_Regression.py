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
    return 1 / (1 + np.exp(-x))


np.random.seed(0)
tol = 1e-8  # Convergence tolerance
lam = None  # L-2 regularizer
max_iter = 20
# Create data
r = 0.95  # Covariance b/w height and weight(x and z) v = b.p
n = 1000  # Size of data
sigma = 1
# Model Settings -- log(p_dis) = log(odds(p_dis)) = beta_0 + beta_1 * x_1 + ...
beta_x, beta_z, beta_v = -4, .9, 1  # True beta coeff
var_x, var_z, var_v = 1, 1, 4  # variances of inputs
# the model specification you want to fit
formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'
x, z = np.random.multivariate_normal([0, 0], [[var_x, r], [r, var_z]], n).T
v = np.random.normal(0, var_v, n) ** 3  # B.p
A = pd.DataFrame({'x': x, 'z': z, 'v': v})
A['log_odds'] = sigmoid(
        A[['x', 'z', 'v']].dot([beta_x, beta_z, beta_v]) +
        sigma * np.random.normal(0, 1, n)
        )
# Probability sample using Binomial distribution
A['y'] = [np.random.binomial(1, p) for p in A.log_odds]
y, X = dmatrices(formula, A, return_type='dataframe')
# Catch singular matrix errors like dividing by zero


def catch_singularity(f):
    '''Silences LinAlg Errors and throws a warning instead.'''

    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated - singular Hessian!')
            return args[0]
    return silencer


@catch_singularity
def newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    W = np.diag((p * (1 - p))[:, 0])
    # derive the hessian
    hessian = X.T.dot(W).dot(X)
    # derive the gradient
    grad = X.T.dot(y - p)

    # regularization step (avoiding overfitting)
    if lam:
        # Return the least-squares solution to a linear matrix equation
        step, *_ = np.linalg.lstsq(hessian + lam*np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    # update our
    beta = curr + step

    return beta


@catch_singularity
def alt_newton_step(curr, X, lam=None):
    '''One naive step of Newton's Method'''

    # compute necessary objects
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    W = np.diag((p * (1 - p))[:, 0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y-p)

    # regularization
    if lam:
        # Compute the inverse of a matrix.
        step = np.dot(np.linalg.inv(hessian + lam*np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)

    # update our weights
    beta = curr + step

    return beta


def check_coefs_convergence(beta_old, beta_new, tol, iters):
    '''Checks whether the coefficients have converged in the l-infinity norm.
    Returns True if they have converged, False otherwise.'''
    # calculate the change in the coefficients
    coef_change = np.abs(beta_old - beta_new)

    # if change hasn't reached the threshold and we have more iterations to go,
    # keep training
    return not (np.any(coef_change > tol) & (iters < max_iter))


# Driver Program
# initial coefficients (weight values), 2 copies, we'll update one
beta_old, beta = np.ones((len(X.columns), 1)), np.zeros((len(X.columns), 1))

# num iterations we've done so far
iter_count = 0
# have we reached convergence?
coefs_converged = False

# if we haven't reached convergence... (training step)
while not coefs_converged:
    # set the old coefficients to our current
    beta_old = beta
    # perform a single step of newton's optimization on our data, set our
    # updated beta values
    beta = newton_step(beta, X, lam=lam)
    # increment the number of iterations
    iter_count += 1

    # check for convergence between our old and new beta values
    coefs_converged = check_coefs_convergence(beta_old, beta, tol, iter_count)

print('Iterations : {}'.format(iter_count))
print('Beta : {}'.format(beta))
