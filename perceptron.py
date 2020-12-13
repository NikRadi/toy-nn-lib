import numpy as np


class Perceptron:
    def __init__(self, size, learning_rate):
        self.learning_rate = learning_rate
        self.weights = np.random.uniform(-1, 1, (size, 1))
        self.bias = np.random.uniform(-1, 1)

    def fit(self, inputs, target):
        lr = self.learning_rate / len(inputs)
        error = (self.forward(inputs) - target) * lr
        self.bias -= sum(error)
        w_errors = [e * i for e, i in zip(error, inputs)]
        weights = self.weights.transpose()
        for w_error in w_errors:
            weights -= w_error

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return np.where(z < 0.5, 0, 1)
