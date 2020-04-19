import os
import struct

import numpy as np


def get_dataset(path, scale=0, train=True):
    if train:
        images_path = os.path.join(path, 'train-images-idx3-ubyte')
        label_path = os.path.join(path, 'train-labels-idx1-ubyte')
    else:
        images_path = os.path.join(path, 't10k-images-idx3-ubyte')
        label_path = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    with open(images_path, 'rb') as f:
        buffers = f.read()
        magic, num, rows, cols = struct.unpack_from('>IIII', buffers, 0)
        bits = num * rows * cols
        images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    images = np.reshape(images, [num, rows * cols]) / 255
    
    with open(label_path, 'rb') as f:
        buffers = f.read()
        magic, num = struct.unpack_from('>II', buffers, 0)
        labels = struct.unpack_from('>' + str(num) + "B", buffers, struct.calcsize('>II'))
    labels = np.reshape(labels, [num])
    
    if train:
        return (images[:int(scale * len(images))], labels[:int(scale * len(labels))]), \
               (images[int(scale * len(images)):], labels[int(scale * len(labels)):])
    else:
        return images, labels
