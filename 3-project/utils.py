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


class F1Meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.tp = [0, 0, 0]
        self.tn = [0, 0, 0]
        self.fp = [0, 0, 0]
        self.fn = [0, 0, 0]

    def reset(self):
        self.tp = [0, 0, 0]
        self.tn = [0, 0, 0]
        self.fp = [0, 0, 0]
        self.fn = [0, 0, 0]

    def update(self, predict, target):
        p = torch.zeros((target.size(0), 3))
        t = torch.zeros((target.size(0), 3))
        for i in range(target.size(0)):
            p[i, predict[i]] = 1
            t[i, target[i]] = 1
        for i in range(3):
            pi = p[:, i]
            ti = t[:, i]
            self.tp[i] += ((pi == 1) & (ti == 1)).cpu().sum()
            self.tn[i] += ((pi == 0) & (ti == 0)).cpu().sum()
            self.fp[i] += ((pi == 0) & (ti == 1)).cpu().sum()
            self.fn[i] += ((pi == 1) & (ti == 0)).cpu().sum()

    def get(self):
        p0 = float(self.tp[0]) / float(self.tp[0] + self.fp[0])
        p1 = float(self.tp[1]) / float(self.tp[1] + self.fp[1])
        p2 = float(self.tp[2]) / float(self.tp[2] + self.fp[2])
        
        r0 = float(self.tp[0]) / float(self.tp[0] + self.fn[0])
        r1 = float(self.tp[1]) / float(self.tp[1] + self.fn[1])
        r2 = float(self.tp[2]) / float(self.tp[2] + self.fn[2])

        p = (p0 + p1 + p2) / 3
        r = (r0 + r1 + r2) / 3
        f1 = 2 * r * p / (r + p)
        return p, p0, p1, p2, r, r0, r1, r2, f1
