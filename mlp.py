import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return np.greater(x, 0).astype(int)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class MLP:
    def __init__(self, sizes, learning_rate):
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(i, j) for i, j in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(1, i) for i in sizes[1:]]
        self.afunc = sigmoid
        self.afunc_prime = sigmoid_prime

    def fit(self, inputs, labels):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in zip(inputs, labels):
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        self.weights = [w - (self.learning_rate / len(inputs)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.learning_rate / len(inputs)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        x = np.array([x])
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        activation = x
        activations = [x]
        zs = []
        for weights, biases in zip(self.weights, self.biases):
            z = np.dot(activation, weights) + biases
            zs.append(z)
            activation = self.afunc(z)
            activations.append(activation)

        delta = (activations[-1] - y) * relu_prime(zs[-1])
        nabla_w[-1] = activations[-2].transpose() * delta
        nabla_b[-1] = delta
        for i in range(len(self.weights) - 1, 0, -1):
            z = zs[i - 1]
            w = self.weights[i]
            delta = np.dot(w, delta.transpose()).transpose() * self.afunc_prime(z)
            nabla_w[i - 1] = activations[i - 1].transpose() * delta
            nabla_b[i - 1] = delta

        return nabla_w, nabla_b

    def forward(self, inputs):
        for weights, biases in zip(self.weights, self.biases):
            inputs = self.afunc(np.dot(inputs, weights) + biases)

        return inputs
