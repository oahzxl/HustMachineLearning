import time

import torch
import torch.backends.cudnn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = float(self.sum) / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

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


def read_txt(path):
    with open(path, "r") as f:
        data = f.readlines()
    new = []
    for i in range(len(data)):
        d = data[i].split(' ')
        d = [d[:-1], int(d[-1][:-1])]
        new.append(d)
    return new
