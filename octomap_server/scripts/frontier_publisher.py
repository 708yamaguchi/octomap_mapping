#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from jsk_topic_tools import ConnectionBasedTransport


class FrontierPublisher(ConnectionBasedTransport):
    def __init__(self):
        super(FrontierPublisher, self).__init__()
        # rospy.wait_for_message(topic, topic_type, timeout)
        self.occupied_pub = self.advertise('occupied', PointCloud2, queue_size=1)
        self.unknown_pub = self.advertise('unknown', PointCloud2, queue_size=1)

    def subscribe(self):
        self.sub_occupied = rospy.Subscriber("octomap_point_cloud_centers", PointCloud2, self.cb_occupied)
        self.sub_unknown = rospy.Subscriber("octomap_unknown_point_cloud_centers", PointCloud2, self.cb_unknown)

    def unsubscribe(self):
        self.sub_occupied.unregister()
        self.sub_unknown.unregister()

    def cb_occupied(self, msg):
        rospy.loginfo("hoge")
        self.occupied_cloud = msg

    def cb_unknown(self, msg):
        self.unknown_cloud = msg
        self.occupied_pub.publish(f.occupied_cloud)
        self.unknown_pub.publish(f.unknown_cloud)
        rospy.sleep(100)

if __name__ == '__main__':
    rospy.init_node('frontier_publisher')
    f = FrontierPublisher()
    rospy.spin()
