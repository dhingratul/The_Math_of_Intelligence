#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:08:08 2017

@author: dhingratul
"""

import numpy as np
from matplotlib import pyplot as plt


def svm_sgd(X, y, viz=False):
    w = np.zeros(len(X[0]))
    lr = 0.01
    epochs = 100000
    errors = []

    # Training
    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            # mis-classification
            if (y[i] * np.dot(X[i], w)) < 1:
                # Weight update -- w = w * LR(y_i * x_i- 2 * lambda * w)
                w = w + lr * ((X[i] * y[i]) + (-2 * (1/epoch) * w))
                error = 1
            else:
                # Correct classification -- w = w * LR(-2 * lambda * w)
                w = w + lr * (-2 * (1/epoch) * w)
        errors.append(error)

    if viz is True:
        plt.plot(errors, '|')
        plt.ylim(0.5, 1.5)
        plt.axes().set_yticklabels([])
        plt.xlabel('Epoch')
        plt.ylabel('Misclassified')
        plt.show()

    return w


# Driver program
# Data
X = np.array([[-2, 4, -1], [4, 1, -1], [1, 6, -1], [2, 4, -1], [6, 2, -1], ])
y = np.array([-1, -1, 1, 1, 1])
# Viz Data Points
for d, sample in enumerate(X):
    # Plot the negative samples
    if d < 2:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
plt.plot([-2, 6], [6, 0.5])

# Add our test samples
plt.scatter(2, 2, s=120, marker='_', linewidths=2, color='yellow')
plt.scatter(4, 3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by svm_sgd()
w = svm_sgd(X, y, True)
x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]
x2x3 = np.array([x2, x3])
X, Y, U, V = zip(*x2x3)
ax = plt.gca()
ax.quiver(X, Y, U, V, scale=1, color='blue')
