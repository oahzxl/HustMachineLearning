import logging

import torch

from getdataset import get_dataset
from knn import KNN
from utils import AccMeter, TimeMeter


class Runner:
    def __init__(self, args):
        self.num_updates = 0
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.acc_meter = AccMeter()
        self.time_meter = TimeMeter()
        self._build_loader()
        self._build_model()

    def _build_loader(self):
        logging.info("Loading data...")
        self.train_data, self.test_data = get_dataset(self.args.path, self.args.scale, train=True)
        if self.args.evaluate:
            self.eval_data = get_dataset(self.args.path, self.args.scale, train=False)

    def _build_model(self):
        self.model = KNN(self.args, self.args.k, self.train_data)

    def predict(self, train=True):
        if train:
            logging.info("Start predicting test data...")
            images, labels = self.test_data
        else:
            logging.info("Start predicting eval data...")
            images, labels = self.eval_data
        images = torch.tensor(images).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        self.acc_meter.reset()
        self.time_meter.reset()

        for i in range(int(len(labels) / self.args.batch_size)):
            if i != int(len(labels / self.args.batch_size)):
                image = images[i * self.args.batch_size:(i + 1) * self.args.batch_size]
                label = labels[i * self.args.batch_size:(i + 1) * self.args.batch_size]
            else:
                image = images[i * self.args.batch_size:]
                label = labels[i * self.args.batch_size:]

            outputs = self.model.predict(image).to(self.device)

            self.acc_meter.update(torch.sum(label == outputs), self.args.batch_size)
            self.time_meter.update()
            logging.info('%5d / %5d, Acc = %.2f%%, %.3f seconds/batch' % (
                self.acc_meter.count, len(labels), self.acc_meter.acc() * 100, 1.0 / self.time_meter.avg
            ))
