import random

import torch
import tqdm

from utils import Counter, plot_neighbour


class KNN(object):
    def __init__(self, args, k, train_data):
        self.k = k
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.data = torch.tensor(train_data[0]).to(self.device)
        self.label = torch.tensor(train_data[1]).to(self.device)
        self.counter = Counter()

    def _get_neighbour(self, inputs):
        self.counter.reset()
        inputs = (self.data - inputs) ** 2
        inputs = torch.sum(inputs, dim=1)
        for i in range(self.k):
            label = torch.argmin(inputs)
            inputs[label] = 999
            self.counter.update(self.label[label])
        return self.counter.max()

    def plot(self, images):
        neighbour = []
        r = random.randint(0, (images.shape[0] - 1))
        image = images[r, :]
        inputs = (self.data - image) ** 2
        inputs = torch.sum(inputs, dim=1)
        for i in range(self.k):
            label = torch.argmin(inputs)
            inputs[label] = 999
            neighbour.append(self.data[label, :])
        plot_neighbour(neighbour)

    def predict(self, images):
        outputs = torch.zeros((images.shape[0]))
        for i in tqdm.tqdm(range(int(images.shape[0]))):
            x = self._get_neighbour(images[i, :])
            outputs[i] = x
        return outputs
