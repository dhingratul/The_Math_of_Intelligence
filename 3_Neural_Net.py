import numpy as np


# Sigmoid non-linearity
def sigmoid(x, deriv=False):
    if deriv is True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# Driver Program
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0, 0, 1, 1]]).T
#  Makes it deterministic
np.random.seed(1)
#  Initial weights
w0 = 2 * np.random.random((3, 1)) - 1
for iter in range(10000):

    # forward pass
    train = X
    pred = sigmoid(np.dot(train, w0))

    # how much did we miss?
    error = y - pred

    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    delta = error * sigmoid(pred, True)

    # update weights
    w0 += np.dot(train.T, delta)

print("Output After Training:")
print(pred)
