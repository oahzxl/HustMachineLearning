import csv
import random
import re

import jieba
import numpy as np
import tqdm
from torchtext.data import Dataset
from torchtext.data import Example

from utils import *


def get_dataset(mode, field):
    if mode == "train":
        data = read_txt(r'./data/processed/train.txt')
    elif mode == "test":
        data = read_txt(r'./data/processed/test.txt')
    elif mode == "eval":
        data = read_txt(r'./data/processed/eval.txt')
    else:
        raise KeyError("Mode must be train, test or eval!")

    examples = []
    for d in data:
        d[1] = d[1] + 1
        examples.append(Example.fromlist(d, field))

    return Dataset(examples, field)


def process_data():

    # set random seed
    np.random.seed(2020)
    random.seed(2020)

    # read data
    csv_reader = csv.reader(open(r'./data/train.csv', encoding='utf-8'))
    data = list(csv_reader)[1:]
    # csv_reader = csv.reader(open(r'./data/test_labled.csv', encoding='utf-8'))
    # data = data + list(csv_reader)[1:]

    # split
    for i in tqdm.tqdm(range(len(data))):
        sub_str = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", " ", data[i][0])
        new_str = []
        for j in sub_str.split(' '):
            new_str += list(jieba.cut(j))
        data[i][0] = new_str.copy()

    # save
    random.shuffle(data)
    save_data("./data/processed/train.txt", data[:int(45000 * 0.6)])
    save_data("./data/processed/test.txt", data[int(45000 * 0.6):int(45000 * 0.8)])
    save_data("./data/processed/eval.txt", data[int(45000 * 0.8):])


def save_data(file_name, data):
    with open(file_name, 'w+') as f:
        for d in data:
            for w in d[0]:
                f.write(w)
                f.write(' ')
            f.write(d[1])
            f.write('\n')
    print("保存文件成功，处理结束")


if __name__ == '__main__':
    process_data()
