#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os


class ClassifyFruit(object):

    def __init__(self):
        self.apple = self.load_data('apple', data_length=15)
        self.banana = self.load_data('banana', data_length=15)
        self.mango = self.load_data('mango', data_length=15)

    def load_data(self, fruit, data_length=5):
        ret = np.zeros((data_length, 3), dtype=np.float32)
        for i in range(1, 1+data_length):
            for l in open('/'.join(os.getcwd().split('/')[0:-1])
                          + '/euslisp/' + fruit + '/'
                          + str(i) + '.txt').readlines():
                data = l[:-1].split(',')
                for j in range(3):  # x, y, z
                    ret[i-1][j] = float(data[j])
        return ret

    def visualize(self):
        # plot
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("Eigen Vector 1")
        ax.set_ylabel("Eigen Vector 2")
        ax.set_zlabel("Eigen Vector 3")
        ax.plot(self.apple[:, 0], self.apple[:, 1], self.apple[:, 2],
                "o", color="#00ff00", ms=4, mew=0.5, label='apple')
        ax.plot(self.banana[:, 0], self.banana[:, 1], self.banana[:, 2],
                "o", color="#0000ff", ms=4, mew=0.5, label='banana')
        ax.plot(self.mango[:, 0], self.mango[:, 1], self.mango[:, 2],
                "o", color="#ff0000", ms=4, mew=0.5, label='mango')
        lg = plt.legend(loc='upper right', fontsize=10)
        lg.get_title().set_fontsize(10)
        plt.title("Classification by Groping", fontsize=20)
        plt.show()


if __name__ == '__main__':
    cf = ClassifyFruit()
    cf.visualize()
