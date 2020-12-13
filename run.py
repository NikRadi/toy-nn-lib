import matplotlib.pyplot as plt
import numpy as np
import random as rand
from mlp import MLP
from perceptron import Perceptron


def generate_data(amount, function):
    data = np.empty((amount, 2))
    labels = np.empty((amount, 1))
    for i in range(amount):
        x = round(np.random.rand() * 10, 2)
        y = round(np.random.rand() * 10, 2)
        data[i] = [x, y]
        labels[i] = [int(function(x) < y)]

    return data, labels


def plot_nn(nn, function, num_data):
    data, labels = generate_data(num_data, function)
    facecolors = ['r' if x == 0 else 'b' for x in labels]

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    function_xs = np.linspace(0, 10, 11)
    function_ys = [function(x) for x in function_xs]
    ax.plot(function_xs, function_ys)

    scatter_xs = [d[0] for d in data]
    scatter_ys = [d[1] for d in data]
    scatter = ax.scatter(scatter_xs, scatter_ys, c=facecolors)

    i = 0
    while True:
        predictions = [int(p > 0.5) for p in nn.forward(data)]
        corrects = np.array([p == l for p, l in zip(predictions, labels)])
        num_correct = sum(corrects.astype(int))
        percent_correct = int(num_correct / len(labels) * 100)
        ax.set_title(f"{percent_correct}% correct, iteration {i}")

        edgecolors = ['r' if p == 0 else 'b' for p in predictions]
        scatter.set_edgecolors(edgecolors)
        if percent_correct == 100:
            break

        nn.fit(data, labels)

        fig.canvas.draw()
        fig.canvas.flush_events()
        i += 1

    plt.pause(120)


if __name__ == "__main__":
    def function(x):
        return x

    nn = Perceptron(2, 0.001)
    plot_nn(nn, function, 10)
