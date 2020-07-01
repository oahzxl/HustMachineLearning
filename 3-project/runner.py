import logging
import os
import random

import numpy as np
import torch.backends.cudnn
import torch.nn as nn
from torchtext.data import Field, LabelField
from torchtext.data.iterator import BucketIterator

from get_dataset import get_dataset
from models_ import *
from utils import *


class Runner:
    def __init__(self, args):
        self.num_updates = 0
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._setup_seed(2020)
        self._build_loader()
        self._build_model()

    def _build_loader(self):
        print("Loading data...")

        TEXT = Field(batch_first=True, fix_length=self.args.max_words)
        LABEL = LabelField(sequential=False, batch_first=True, use_vocab=False)
        field = [('text', TEXT), ('label', LABEL)]

        train = get_dataset("train", field)
        test = get_dataset("test", field)
        evl = get_dataset("eval", field)
        TEXT.build_vocab(train, test, evl, min_freq=3)

        self.vocab = TEXT
        self.train_iter, self.test_iter, self.evl_iter = BucketIterator.splits(
            (train, test, evl),
            batch_sizes=(self.args.batch_size, self.args.batch_size, self.args.batch_size),
            device=self.device,
            shuffle=True,
            sort=False,
            repeat=False,
        )

    @staticmethod
    def _setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def _build_model(self):
        # self.model = FC(self.args, self.vocab)
        self.model = RNN(self.args, self.vocab)
        # self.model = TextCNN(self.args, self.vocab)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=1)
        self.criteria = nn.CrossEntropyLoss()

    def train(self):
        if not os.path.exists(self.args.model_saved_path):
            os.makedirs(self.args.model_saved_path)
        for epoch in range(1, self.args.max_num_epochs + 1):
            # logging.info('Start Epoch {}'.format(epoch))
            self._train_one_epoch(epoch)
            # path = os.path.join(self.args.model_saved_path, 'model-%d' % epoch)
            # torch.save(self.model.state_dict(), path)
            # logging.info('model saved to %s' % path)
            # self.eval()
        logging.info('Done.')

    def _train_one_epoch(self, epoch):
        self.model.train()
        loss_meter1 = AverageMeter()
        time_meter = TimeMeter()
        for b, (text, label) in enumerate(self.train_iter, 1):

            text = text.to(self.device)
            label = label.to(self.device)
            outputs = self.model(text)

            loss = self.criteria(outputs, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch - 1 + b / len(self.train_iter))

            loss_meter1.update(loss.item())
            time_meter.update()

        self.model.eval()
        with torch.no_grad():
            acc_meter = AverageMeter()
            loss_meter2 = AverageMeter()
            f1_meter = F1Meter()

            for b, (text, label) in enumerate(self.evl_iter, 1):
                text = text.to(self.device)
                label = label.to(self.device)
                outputs = self.model(text).squeeze(-1)
                loss = self.criteria(outputs, label)
                outputs = torch.argmax(outputs, dim=-1)
                acc = torch.sum(outputs == label)
                acc_meter.update(acc, self.args.batch_size)
                f1_meter.update(outputs, label)
                loss_meter2.update(loss.item())

        p, p0, p1, p2, r, r0, r1, r2, f1 = f1_meter.get()
        logging.info(
            'Epoch %2d, train loss = %.3f, evl loss = %.3f, '
            'acc = %.3f, p = %.3f, p0 = %.3f, p1 = %.3f, p2 = %.3f, '
            'r = %.3f, r0 = %.3f, r1 = %.3f, r2 = %.3f, f1 = %.3f' % (
                epoch, loss_meter1.avg, loss_meter2.avg, acc_meter.avg,
                p, p0, p1, p2, r, r0, r1, r2, f1
            ))

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            acc_meter = AverageMeter()
            loss_meter = AverageMeter()

            for b, (text, label) in enumerate(self.test_iter, 1):

                text = text.to(self.device)
                label = label.to(self.device)
                outputs = self.model(text).squeeze(-1)
                loss = self.criteria(outputs, label)
                outputs = torch.argmax(outputs, dim=-1)
                acc = torch.sum(outputs == label)
                acc_meter.update(acc, self.args.batch_size)
                loss_meter.update(loss.item())

            print('test loss = %.4f, test acc = %.4f' % (
                loss_meter.avg, acc_meter.avg
            ))
