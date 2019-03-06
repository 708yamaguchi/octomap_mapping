#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import os

import rospkg

from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier


class ClassifyFruit(object):

    def __init__(self):
        self.apple = self.load_data('apple', data_length=15)
        self.banana = self.load_data('banana', data_length=15)
        self.mango = self.load_data('mango', data_length=15)

    def load_data(self, fruit, data_length=5):
        ret = np.zeros((data_length, 3), dtype=np.float32)
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('octomap_server')
        for i in range(1, 1+data_length):
            for l in open(package_path + '/euslisp/' + fruit + '/'
                          + str(i) + '.txt').readlines():
                data = l[:-1].split(',')
                for j in range(3):  # x, y, z
                    ret[i-1][j] = float(data[j])
        return ret

    def classify_test(self, k=1):
        # see https://blog.amedama.jp/entry/2017/03/18/140238
        # prepare for classification (e.g. create train and test data)
        features = np.concatenate((self.apple, self.banana, self.mango), axis=0)
        target_apple = np.zeros(len(self.apple), dtype=np.int)
        target_banana = np.zeros(len(self.apple), dtype=np.int) + 1
        target_mango = np.zeros(len(self.apple), dtype=np.int) + 2
        targets = np.append(np.append(target_apple, target_banana), target_mango)
        predicted_labels = []
        loo = LeaveOneOut()
        # split train and test data
        for train, test in loo.split(features):
            # print("leave one: {}".format(test[0]))
            train_data = features[train]
            target_data = targets[train]
            # load model of k nearest neighbor
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(train_data, target_data)
            # predict
            predicted_label = model.predict(features[test])
            predicted_labels.append(predicted_label)
        # calc score
        score = accuracy_score(targets, predicted_labels)
        return score

    def classify(self, data, k=7):  # 7 is desirable for this dataset
        # see https://blog.amedama.jp/entry/2017/03/18/140238
        # prepare for classification (e.g. create train and test data)
        features = np.concatenate((self.apple, self.banana, self.mango), axis=0)
        target_apple = np.zeros(len(self.apple), dtype=np.int)
        target_banana = np.zeros(len(self.apple), dtype=np.int) + 1
        target_mango = np.zeros(len(self.apple), dtype=np.int) + 2
        targets = np.append(np.append(target_apple, target_banana), target_mango)
        # train k nearest neighbor
        train = np.zeros(features.shape[0], dtype=np.int)
        for (i, element) in enumerate(train):
            train[i] = i
        train_data = features[train]
        target_data = targets[train]
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_data, target_data)
        # predict
        input_data = np.array(data, dtype=np.float32)
        predicted_label = model.predict(input_data[None])
        # label: 0->apple, 1->banana, 2->mango
        return predicted_label

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
        print(self.mango[:, 0])
        lg = plt.legend(loc='upper right', fontsize=10)
        lg.get_title().set_fontsize(10)
        plt.title("Classification by Groping", fontsize=20)
        plt.show()

    def visualize_target(self, eigen1, eigen2, eigen3):
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
        ax.plot([eigen1], [eigen2], [eigen3],
                "o", color="#000000", ms=4, mew=0.5, label='target')
        lg = plt.legend(loc='upper right', fontsize=10)
        lg.get_title().set_fontsize(10)
        plt.title("Classification by Groping", fontsize=20)
        plt.show()


if __name__ == '__main__':
    cf = ClassifyFruit()
    # cf.classify([411.728, 163.605, 40.9302], k=1) # banana/1.txt
    # cf.classify([716.922, 476.102, 80.1956], k=1) # apple/1.txt
    # cf.visualize_target(291.401, 113.970, 51.8338) # classify target pca
    cf.visualize()
    # for i in range(20):
    #     print("k-nearest-neighbor: {}, score: {}".format(
    #         i+1, cf.classify_test(k=(i+1))))
