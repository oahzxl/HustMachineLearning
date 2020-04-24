import logging

import torch

from get_dataset import get_dataset
from knn import KNN
from utils import plot_acc


class Runner:
    def __init__(self, args):
        self.num_updates = 0
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._build_loader()
        self._build_model(args.k)

    def _build_loader(self):
        logging.info("Loading data...")
        self.train_data, self.test_data = get_dataset(self.args.path, self.args.scale, train=True)
        if self.args.evaluate:
            self.eval_data = get_dataset(self.args.path, self.args.scale, train=False)

    def _build_model(self, k):
        self.model = KNN(self.args, k, self.train_data)

    def predict(self, train=True):
        if train:
            images, labels = self.test_data
        else:
            images, labels = self.eval_data
        images = torch.tensor(images[:100]).to(self.device)
        labels = torch.tensor(labels[:100]).to(self.device)

        outputs = self.model.predict(images).to(self.device)
        acc = torch.sum(labels == outputs).float() / labels.shape[0]
        logging.info('K = %d, Acc = %.2f%%' % (
            self.model.k, acc * 100
        ))
        return acc

    def acc(self):
        loss = []
        for k in range(1, 6):
            self._build_model(k)
            loss.append(float(self.predict(train=False)))
        plot_acc(loss)

    def plot(self):
        images, labels = self.eval_data
        images = torch.tensor(images).to(self.device)
        self.model.plot(images)
