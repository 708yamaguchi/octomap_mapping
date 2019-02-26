#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os


def load_data(fruit, data_length=5):
    ret = np.zeros((data_length, 3), dtype=np.float32)
    for i in range(1, 6):
        for l in open('/'.join(os.getcwd().split('/')[0:-1])
                      + '/euslisp/' + fruit + '/' + str(i) + '.txt').readlines():
            data = l[:-1].split(',')
            for j in range(3):  # x, y, z
                ret[i-1][j] = float(data[j])
    return ret


def classify():
    apple = load_data('apple', data_length=5)
    banana = load_data('banana', data_length=5)
    mango = load_data('mango', data_length=5)

    # plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("Eigen Vector 1")
    ax.set_ylabel("Eigen Vector 2")
    ax.set_zlabel("Eigen Vector 3")
    ax.plot(apple[:, 0], apple[:, 1], apple[:, 2], "o", color="#00ff00", ms=4, mew=0.5, label='apple')
    ax.plot(banana[:, 0], banana[:, 1], banana[:, 2], "o", color="#0000ff", ms=4, mew=0.5, label='banana')
    ax.plot(mango[:, 0], mango[:, 1], mango[:, 2], "o", color="#ff0000", ms=4, mew=0.5, label='mango')
    lg = plt.legend(loc='upper right', fontsize=10)
    lg.get_title().set_fontsize(10)
    plt.title("Classification by Groping", fontsize=20)
    plt.show()


if __name__ == '__main__':
    classify()
