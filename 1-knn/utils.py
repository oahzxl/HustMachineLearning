import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_acc(loss):
    plt.plot(range(1, len(loss) + 1), loss)
    plt.xticks(np.arange(1, len(loss) + 1, 1))
    plt.xlabel('K')
    plt.ylabel('Acc')
    plt.show()


def plot_neighbour(x):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=math.ceil(float(len(x)) / 2))
    ax = ax.flatten()
    for i in range(len(x)):
        img = x[i].reshape(28, 28)
        img = (img * 255).cpu().numpy()
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


class Counter(object):
    def __init__(self):
        self.count = torch.zeros(10)

    def update(self, n):
        self.count[n] += 1

    def reset(self):
        self.count = torch.zeros(10)

    def max(self):
        return self.count.argmax()
