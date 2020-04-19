import time

import torch


class AccMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def acc(self):
        return float(self.sum) / self.count


class TimeMeter(object):
    def __init__(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class Counter(object):
    def __init__(self):
        self.count = torch.zeros(10)

    def update(self, n):
        self.count[n] += 1

    def reset(self):
        self.count = torch.zeros(10)

    def max(self):
        return self.count.argmax()
