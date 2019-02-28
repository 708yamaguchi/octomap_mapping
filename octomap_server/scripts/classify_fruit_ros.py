#!/usr/bin/env python
# -*- coding: utf-8 -*-

from classify_fruit import ClassifyFruit
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String


class ClassifyFruitRos:
    def __init__(self):
        self.sub = rospy.Subscriber('/pca_fruit',
                                    Float32MultiArray,
                                    self.cb)

        self.pub = rospy.Publisher('/fruit_class',
                                   String, queue_size=1)

    def cb(self, msg):
        # classify fruit
        eigen_values = msg.data
        cf = ClassifyFruit()
        label_num = cf.classify(eigen_values, k=7)
        if label_num == 0:
            label_string = 'apple'
        elif label_num == 1:
            label_string = 'banana'
        elif label_num == 2:
            label_string = 'mango'
        else:
            print('invalid label is returned.!!')
        fruit_class = String()
        fruit_class.data = label_string
        # publish
        print("target fruit is {}".format(label_string))
        self.pub.publish(fruit_class)


if __name__ == '__main__':
    rospy.init_node('classify_fruit_ros')
    ClassifyFruitRos()
    rospy.spin()
