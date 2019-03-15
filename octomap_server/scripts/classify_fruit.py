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

    def classify_test_old(self, k=1):
        # see https://blog.amedama.jp/entry/2017/03/18/140238
        # prepare for classification (e.g. create train and test data)
        features = np.concatenate((self.apple, self.banana, self.mango), axis=0)
        target_apple = np.zeros(len(self.apple), dtype=np.int)
        target_banana = np.zeros(len(self.apple), dtype=np.int) + 1
        target_mango = np.zeros(len(self.apple), dtype=np.int) + 2
        targets = np.append(np.append(target_apple, target_banana), target_mango) # label
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

    def classify_cross_validation(self, k=7, divide=5):
        # features = np.concatenate((self.apple, self.banana, self.mango), axis=0)
        if len(self.apple) != len(self.banana) or \
           len(self.banana) != len(self.mango) or \
           len(self.mango) != len(self.apple) or \
           len(self.apple) % divide != 0:
            print('invalid input data')
            return
        data_length = len(self.apple)
        test_length = len(self.apple) / divide
        train_label_apple = np.zeros(data_length - test_length, dtype=np.int)
        train_label_banana = np.zeros(data_length - test_length, dtype=np.int) + 1
        train_label_mango = np.zeros(data_length - test_length, dtype=np.int) + 2
        train_labels = np.append(np.append(train_label_apple, train_label_banana), train_label_mango) # label for train data
        test_label_apple = np.zeros(test_length, dtype=np.int)
        test_label_banana = np.zeros(test_length, dtype=np.int) + 1
        test_label_mango = np.zeros(test_length, dtype=np.int) + 2
        test_labels = np.append(np.append(test_label_apple, test_label_banana), test_label_mango) # label for train data
        scores = np.array([], dtype=np.float32)
        scores_apple = np.array([], dtype=np.float32)
        scores_banana = np.array([], dtype=np.float32)
        scores_mango = np.array([], dtype=np.float32)
        mean_score = 0
        mean_score_apple = 0
        mean_score_banana = 0
        mean_score_mango = 0
        # divide dataset for cross validation
        for i in range(divide):
            train_data = np.concatenate(
                (np.concatenate((self.apple[0:i*test_length],
                                 self.apple[(i+1)*test_length:data_length]), axis=0),
                 np.concatenate((self.banana[0:i*test_length],
                                 self.banana[(i+1)*test_length:data_length]), axis=0),
                 np.concatenate((self.mango[0:i*test_length],
                                 self.mango[(i+1)*test_length:data_length]), axis=0)),
                axis=0)
            test_data = np.concatenate(
                (self.apple[i*test_length:(i+1)*test_length],
                 self.banana[i*test_length:(i+1)*test_length],
                 self.mango[i*test_length:(i+1)*test_length]),
                axis=0)
            # load model of k nearest neighbor
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(train_data, train_labels)
            # predict
            predicted_label = model.predict(test_data)
            predicted_label_apple = model.predict(test_data[0:test_length])
            predicted_label_banana = model.predict(test_data[test_length:test_length*2])
            predicted_label_mango = model.predict(test_data[test_length*2:test_length*3])
            # calc score
            score = accuracy_score(test_labels, predicted_label)
            score_apple = accuracy_score(test_label_apple, predicted_label_apple)
            score_banana = accuracy_score(test_label_banana, predicted_label_banana)
            score_mango = accuracy_score(test_label_mango, predicted_label_mango)
            scores = np.append(scores, score)
            scores_apple = np.append(scores_apple, score_apple)
            scores_banana = np.append(scores_banana, score_banana)
            scores_mango = np.append(scores_mango, score_mango)
            # print("accuracy {} at {} cycle".format(score, i))
        mean_score = scores.mean()
        mean_score_apple = scores_apple.mean()
        mean_score_banana = scores_banana.mean()
        mean_score_mango = scores_mango.mean()
        # print("mean accuracy: {}".format(mean_score))
        print("apple: {}, banana: {}, mango: {}".format(mean_score_apple, mean_score_banana, mean_score_mango))
        # print("banana: {}, (apple+mango)/2: {}".format(mean_score_banana, (mean_score_apple+mean_score_mango)/2.0))
        return mean_score

    def classify(self, data, k=7):  # 7 is desirable for this dataset
        # see https://blog.amedama.jp/entry/2017/03/18/140238
        # prepare for classification (e.g. create train and test data)
        features = np.concatenate((self.apple, self.banana, self.mango), axis=0)
        train_label_apple = np.zeros(len(self.apple), dtype=np.int)
        train_label_banana = np.zeros(len(self.apple), dtype=np.int) + 1
        train_label_mango = np.zeros(len(self.apple), dtype=np.int) + 2
        train_labels = np.append(np.append(train_label_apple, train_label_banana), train_label_mango)
        # train k nearest neighbor
        train = np.zeros(features.shape[0], dtype=np.int)
        for (i, element) in enumerate(train):
            train[i] = i
        train_data = features[train]
        target_data = train_labels[train]
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_data, target_data)
        # predict
        input_data = np.array(data, dtype=np.float32)
        predicted_label = model.predict(input_data[None])
        # label: 0->apple, 1->banana, 2->mango
        return predicted_label

    def visualize_dataset(self):
        # plot
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("Eigen Vector 1")
        ax.set_ylabel("Eigen Vector 2")
        ax.set_zlabel("Eigen Vector 3")
        # ax.plot(self.apple[:, 0], self.apple[:, 1], self.apple[:, 2],
        #         "o", color="#00ff00", ms=4, mew=0.5, label='apple')
        ax.plot(self.apple[:, 0], self.apple[:, 1], self.apple[:, 2],
                "o", color="#008800", ms=8, mew=0.5, label='apple')
        # ax.plot(self.banana[:, 0], self.banana[:, 1], self.banana[:, 2],
        #         "o", color="#0000ff", ms=4, mew=0.5, label='banana')
        ax.plot(self.banana[:, 0], self.banana[:, 1], self.banana[:, 2],
                "o", color="#000088", ms=8, mew=0.5, label='banana')
        # ax.plot(self.mango[:, 0], self.mango[:, 1], self.mango[:, 2],
        #         "o", color="#ff0000", ms=4, mew=0.5, label='mango')
        ax.plot(self.mango[:, 0], self.mango[:, 1], self.mango[:, 2],
                "o", color="#880000", ms=8, mew=0.5, label='mango')

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
                "o", color="#008800", ms=8, mew=0.5, label='apple')
        ax.plot(self.banana[:, 0], self.banana[:, 1], self.banana[:, 2],
                "o", color="#000088", ms=8, mew=0.5, label='banana')
        ax.plot(self.mango[:, 0], self.mango[:, 1], self.mango[:, 2],
                "o", color="#880000", ms=8, mew=0.5, label='mango')
        ax.plot([eigen1], [eigen2], [eigen3],
                "o", color="#ff0000", ms=8, mew=0.5, label='target')
        lg = plt.legend(loc='upper right', fontsize=14)
        lg.get_title().set_fontsize(10)
        # plt.title("Classification by Groping", fontsize=20)
        plt.show()


if __name__ == '__main__':
    cf = ClassifyFruit()
    # cf.classify([411.728, 163.605, 40.9302], k=1) # banana/1.txt
    # cf.classify([716.922, 476.102, 80.1956], k=1) # apple/1.txt
    # cf.visualize_dataset()
    for i in range(20):
        print("k-nearest-neighbor' k: {}, mean score: {}".format(
            i+1, cf.classify_cross_validation(k=(i+1))))
    # visualize target pca. this is sample of groping-in-bag experiment
    # cf.visualize_target(291.401, 113.970, 51.8338)
