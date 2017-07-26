#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 09:51:20 2017

@author: dhingratul
"""
import copy
import numpy as np

# Input Data - binary numbers for integers corrsp from 0-256
int_to_binary = {}
binary_dim = 8
max_val = (2 ** binary_dim)  # 2 ^ 8 =256
binary_val = np.unpackbits(np.array([range(max_val)], dtype=np.uint8).T,
                           axis=1)
for i in range(max_val):  # map Integer values to Binary values
    int_to_binary[i] = binary_val[i]

# sigmoid function


def sigmoid(x, deriv=False):
    if deriv is True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
# hyperparameters
inputLayerSize = 2
hiddenLayerSize = 16
outputLayerSize = 1
# 3 weight values
W1 = 2 * np.random.random((inputLayerSize, hiddenLayerSize)) - 1
W2 = 2 * np.random.random((hiddenLayerSize, outputLayerSize)) - 1
W_h = 2 * np.random.random((hiddenLayerSize, hiddenLayerSize)) - 1
# Initialize Updated Weights Values
W1_update = np.zeros_like(W1)
W2_update = np.zeros_like(W2)
W_h_update = np.zeros_like(W_h)

# Compute the the Sum of two integers
num_iters = 100000
for j in range(num_iters):

    # a + b = c (random values)
    a_int = np.random.randint(max_val/2)
    b_int = np.random.randint(max_val/2)
    c_int = a_int + b_int

    # get binary values for a,b, and c
    a = int_to_binary[a_int]
    b = int_to_binary[b_int]
    c = int_to_binary[c_int]

    # Save predicted binary outputs
    d = np.zeros_like(c)

    # Initialize Error
    overallError = 0

    # Store output gradients & hidden layer values
    output_layer_gradients = list()
    hidden_layer_values = list()
    hidden_layer_values.append(np.zeros(hiddenLayerSize))  # init as 0

    # Forward propagation to compute the sum of two 8 digit long binary ints
    for position in range(binary_dim):

        # input - binary values of a & b
        X = np.array([[a[binary_dim - position - 1],
                       b[binary_dim - position - 1]]])
        # output - the sum c
        y = np.array([[c[binary_dim - position - 1]]]).T

        # Calculate the error
        layer_1 = sigmoid(np.dot(X, W1) + np.dot(hidden_layer_values[-1], W_h))
        layer_2 = sigmoid(np.dot(layer_1, W2))
        output_error = y - layer_2

        # Save the error gradients at each step as it will be propagated back
        output_layer_gradients.append((output_error) * sigmoid(layer_2,
                                      deriv=True))

        # Save the sum of error at each binary position
        overallError += np.abs(output_error[0])

        # Round off the values to nearest "0" or "1" and save it to a list
        d[binary_dim - position - 1] = np.round(layer_2[0][0])

        # Save the hidden layer to be used later
        hidden_layer_values.append(copy.deepcopy(layer_1))

    future_layer_1_gradient = np.zeros(hiddenLayerSize)

    # backpropagate the error to the previous timesteps!
    for position in range(binary_dim):
        # a[0], b[0] -> a[1]b[1] ....
        X = np.array([[a[position], b[position]]])
        # The last step Hidden Layer where we are currently a[0],b[0]
        layer_1 = hidden_layer_values[-position - 1]
        # The hidden layer before the current layer, a[1],b[1]
        prev_hidden_layer = hidden_layer_values[-position-2]
        # Errors at Output Layer, a[1],b[1]
        output_layer_gradient = output_layer_gradients[-position-1]
        layer_1_gradients = (future_layer_1_gradient.dot(W_h.T) +
                             output_layer_gradient.dot(
                                     W2.T)) * sigmoid(layer_1, deriv=True)

        # Update all the weights and try again
        W2_update += np.atleast_2d(layer_1).T.dot(output_layer_gradient)
        W_h_update += np.atleast_2d(prev_hidden_layer).T.dot(layer_1_gradients)
        W1_update += X.T.dot(layer_1_gradients)

        future_layer_1_gradient = layer_1_gradients

    # Update the weights with the values
    W1 += W1_update
    W2 += W2_update
    W_h += W_h_update

    # Clear the updated weights values
    W1_update *= 0
    W2_update *= 0
    W_h_update *= 0

    # Print out the Progress of the RNN
    if (j % 1000 == 0):
        print("Error:" + str(overallError))
        print("Pred:" + str(d))
        print("True:" + str(c))
        out = 0
        for index, x in enumerate(reversed(d)):
            out += x * pow(2, index)
        print(str(a_int) + " + " + str(b_int) + " = " + str(out))
        print("------------")
