#!/usr/bin/env python

from __future__ import division
# from __future__ import print_function

import copy
# import time

import chainer
import chainer.functions as F
import numpy as np

from geometry_msgs.msg import Point
from jsk_topic_tools import ConnectionBasedTransport
import rospy
# from sensor_msgs.msg import PointCloud2
# from std_srvs.srv import Empty
import threading
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


# When using this class,
# do not forget to subscribe topic which this class advertise.
class FrontierPublisher(ConnectionBasedTransport):
    def __init__(self):
        super(FrontierPublisher, self).__init__()
        # rospy.wait_for_message(topic, topic_type, timeout)
        self.frontier_pub = self.advertise('frontier_cells_vis_array',
                                           MarkerArray, queue_size=1)
        # Set occupancy region
        self.occupancy_region_min_x = rospy.get_param(
            '/octomap_server_contact/occupancy_min_x', -0.3)
        self.occupancy_region_max_x = rospy.get_param(
            '/octomap_server_contact/occupancy_max_x', 0.2)
        self.occupancy_region_min_y = rospy.get_param(
            '/octomap_server_contact/occupancy_min_y', -0.6)
        self.occupancy_region_max_y = rospy.get_param(
            '/octomap_server_contact/occupancy_max_y', -0.1)
        self.occupancy_region_min_z = rospy.get_param(
            '/octomap_server_contact/occupancy_min_z', 0.2)
        self.occupancy_region_max_z = rospy.get_param(
            '/octomap_server_contact/occupancy_max_z', 0.8)
        self.resolution = rospy.get_param(
            '/octomap_server_contact/resolution', 0.005)

        # Bool array for occupied grid
        x_num = int(
            (self.occupancy_region_max_x - self.occupancy_region_min_x) /
            self.resolution)
        y_num = int(
            (self.occupancy_region_max_y - self.occupancy_region_min_y) /
            self.resolution)
        z_num = int(
            (self.occupancy_region_max_z - self.occupancy_region_min_z) /
            self.resolution)
        self.free = np.full((x_num, y_num, z_num), False, dtype=np.bool)
        # self.free_past = np.full((x_num, y_num, z_num), False, dtype=np.bool)
        self.unknown = np.full((x_num, y_num, z_num), False, dtype=np.bool)
        self.frontier = np.full((x_num, y_num, z_num), False, dtype=np.bool)
        self.lock = threading.Lock()

    def subscribe(self):
        sub_free = rospy.Subscriber("free_cells_vis_array",
                                    MarkerArray, self.cb_free)
        sub_unknown = rospy.Subscriber("unknown_cells_vis_array",
                                       MarkerArray, self.cb_unknown)
        self.subs = [sub_free, sub_unknown]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def update_grid(self, marker_array, occupancy_type):
        assert occupancy_type in ['free', 'unknown']
        x_min = self.occupancy_region_min_x
        y_min = self.occupancy_region_min_y
        z_min = self.occupancy_region_min_z
        resolution = self.resolution

        if occupancy_type == 'free':
            self.free[:] = False
            # Update each grid.
            # Be careful that size of grids in octomap may differ
            # from each other.
            for marker in marker_array.markers:
                x_size = marker.scale.x
                y_size = marker.scale.y
                z_size = marker.scale.z

                # NOTE +1 in below codes are very important to avoid
                #      making gap between unknown grids and frontier grids.
                x_num_plus1 = int(x_size / resolution) + 1
                y_num_plus1 = int(y_size / resolution) + 1
                z_num_plus1 = int(z_size / resolution) + 1
                for point in marker.points:
                    xyz_min = np.round(np.array([
                        point.x - (x_size / 2.0) - x_min,
                        point.y - (y_size / 2.0) - y_min,
                        point.z - (z_size / 2.0) - z_min
                    ]) / resolution).astype(np.int)
                    self.free[xyz_min[0]:xyz_min[0] + x_num_plus1,
                              xyz_min[1]:xyz_min[1] + y_num_plus1,
                              xyz_min[2]:xyz_min[2] + z_num_plus1] = True
            # copy latest free to past free
            # self.free_past = copy.deepcopy(self.free)

        elif occupancy_type == 'unknown':
            self.unknown[:] = False
            # Be careful that size of all grids are resolution
            for marker in marker_array.markers:  # markers' length is always 1
                for point in marker.points:
                    xyz_min = np.round(np.array([
                        point.x - x_min,
                        point.y - y_min,
                        point.z - z_min]) / resolution - 1).astype(np.int)
                    self.unknown[xyz_min[0]][xyz_min[1]][xyz_min[2]] = True

    # Only when free grid topic comes, publish frontier grid
    def cb_free(self, msg):
        # rospy.loginfo('[cb_free]')
        self.lock.acquire()
        self.update_grid(msg, 'free')
        self.lock.release()

    def cb_unknown(self, msg):
        # rospy.loginfo('[cb_unknown] publish frontier grids')
        self.frame_id = msg.markers[0].header.frame_id
        self.ns = msg.markers[0].ns
        self.update_grid(msg, 'unknown')
        self.publish_frontier()

    def publish_frontier(self):
        self.frontier[:] = False
        # Use max_pooling for detecting free grids adjacent to unknown grids
        unknown_grid = chainer.Variable(
            np.array([[self.unknown]], dtype=np.float32))
        max_grid = F.max_pooling_nd(
            unknown_grid, ksize=3, stride=1, pad=1).data[0][0].astype(np.bool)
        self.lock.acquire()
        self.frontier = np.logical_and(max_grid, self.free)
        self.lock.release()

        # For debug, visualize unknown grid as frontier grid
        # self.frontier = (self.unknown == 1).astype(np.int)
        # self.frontier = (self.free == 1).astype(np.int)
        # self.frontier = (max_grid == 1).astype(np.int)
        print(11111111111111)
        print(np.sum(self.frontier == 1))
        print(np.sum(self.free))
        # print(np.sum(max_grid == 1))
        print(np.sum(self.unknown))
        print(22222222222222)

        frontier_marker = Marker()
        point_list = [None] * np.sum(self.frontier)
        count = 0
        for i in range(self.frontier.shape[0]):
            for j in range(self.frontier.shape[1]):
                for k in range(self.frontier.shape[2]):
                    if self.frontier[i][j][k]:
                        point = Point()
                        point.x = (i + 1) * self.resolution + \
                            self.occupancy_region_min_x
                        point.y = (j + 1) * self.resolution + \
                            self.occupancy_region_min_y
                        point.z = (k + 1) * self.resolution + \
                            self.occupancy_region_min_z
                        point_list[count] = point
                        count = count + 1

        frontier_marker.points = point_list
        frontier_marker.header.stamp = rospy.Time.now()
        frontier_marker.header.frame_id = self.frame_id
        frontier_marker.ns = self.ns
        frontier_marker.type = 6
        frontier_marker.action = 0
        frontier_marker.scale.x = self.resolution
        frontier_marker.scale.y = self.resolution
        frontier_marker.scale.z = self.resolution
        frontier_marker.color.r = 1.0
        frontier_marker.color.g = 0.0
        frontier_marker.color.b = 0.0
        frontier_marker.color.a = 1.0
        pub_marker = MarkerArray()
        pub_marker.markers = [frontier_marker]

        self.frontier_pub.publish(pub_marker)


if __name__ == '__main__':
    rospy.init_node('get_frontier')
    rospy.loginfo("start publishing frontier grids")
    fp = FrontierPublisher()
    # s = rospy.Service('publish_frontier', Empty, fp.publish_frontier)
    rospy.spin()
