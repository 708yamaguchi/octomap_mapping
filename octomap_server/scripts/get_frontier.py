#!/usr/bin/env python

import time
import chainer
import chainer.functions as F
import copy
from geometry_msgs.msg import Point
import rospy
# from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from jsk_topic_tools import ConnectionBasedTransport
import numpy as np
# from std_srvs.srv import *


# when using this class, do not forget to subscribe topic which this class advertise
class FrontierPublisher(ConnectionBasedTransport):
    def __init__(self):
        super(FrontierPublisher, self).__init__()
        # rospy.wait_for_message(topic, topic_type, timeout)
        self.frontier_pub = self.advertise('frontier_cells_vis_array',
                                           MarkerArray, queue_size=1)
        # set occupancy region
        self.occupancy_region = {}
        for i, rosparam in enumerate(['occupancy_min_x', 'occupancy_max_x',
                                      'occupancy_min_y', 'occupancy_max_y',
                                      'occupancy_min_z', 'occupancy_max_z']):
            if rospy.has_param(rosparam):
                self.occupancy_region[rosparam] = rospy.get_param(rosparam)
            else:
                self.occupancy_region[rosparam] = [-0.3, 0.2, -0.7, 0.0, 0.3, 0.8][i]
        if rospy.has_param('resolution'):
            self.resolution = rospy.get_param('resolution')
        else:
            self.resolution = 0.005

        # bool array for occupied grid
        x_num = (self.occupancy_region['occupancy_max_x']
                 - self.occupancy_region['occupancy_min_x']) / self.resolution
        y_num = (self.occupancy_region['occupancy_max_y']
                 - self.occupancy_region['occupancy_min_y']) / self.resolution
        z_num = (self.occupancy_region['occupancy_max_z']
                 - self.occupancy_region['occupancy_min_z']) / self.resolution
        # grid is 0: free, 1: occupied, 2: unknown # NOQA
        self.grid = np.full((int(x_num), int(y_num), int(z_num)),
                            2,  # initialize array by 2
                            dtype=np.int)
        self.frontier = np.full((int(x_num), int(y_num), int(z_num)),
                                0,  # initialize array by 0
                                dtype=np.int)

    def subscribe(self):
        self.sub_free = rospy.Subscriber("free_cells_vis_array",
                                         MarkerArray, self.cb_free)
        self.sub_occupied = rospy.Subscriber("occupied_cells_vis_array",
                                             MarkerArray, self.cb_occupied)
        self.sub_unknown = rospy.Subscriber("unknown_cells_vis_array",
                                            MarkerArray, self.cb_unknown)

    def unsubscribe(self):
        self.sub_free.unregister()
        self.sub_occupied.unregister()
        self.sub_unknown.unregister()

    def update_grid(self, marker_array, occupancy_type=True):
        if occupancy_type is True:
            rospy.logerr('occupancy type is not set.')
        elif occupancy_type == 'free':
            oc_type = 0
        elif occupancy_type == 'occupied':
            oc_type = 1
        elif occupancy_type == 'unknown':
            oc_type = 2
        # set grid_type to each grid
        # be careful that size of grids in octomap may differ from each other
        for marker in marker_array.markers:
            x_size = marker.scale.x
            y_size = marker.scale.y
            z_size = marker.scale.z
            for point in marker.points:
                x_min_index = int(np.round((point.x - (x_size / 2.0) - self.occupancy_region['occupancy_min_x']) / self.resolution))
                x_max_index = int(np.round((point.x + (x_size / 2.0) - self.occupancy_region['occupancy_min_x']) / self.resolution))
                y_min_index = int(np.round((point.y - (y_size / 2.0) - self.occupancy_region['occupancy_min_y']) / self.resolution))
                y_max_index = int(np.round((point.y + (y_size / 2.0) - self.occupancy_region['occupancy_min_y']) / self.resolution))
                z_min_index = int(np.round((point.z - (z_size / 2.0) - self.occupancy_region['occupancy_min_z']) / self.resolution))
                z_max_index = int(np.round((point.z + (z_size / 2.0) - self.occupancy_region['occupancy_min_z']) / self.resolution))
                for x in range(x_min_index, x_max_index):
                    for y in range(y_min_index, y_max_index):
                        for z in range(z_min_index, z_max_index):
                            self.grid[x][y][z] = oc_type

    # only when free grid topic comes, publish frontier grid
    def cb_free(self, msg):
        self.update_grid(msg, 'free')
        self.frame_id = msg.markers[0].header.frame_id
        self.ns = msg.markers[0].ns
        t_start = time.time()
        self.publish_frontier()
        t_end = time.time()

        rospy.loginfo('frontier grids computation time: {}'.format(t_end - t_start))

    def cb_occupied(self, msg):
        self.update_grid(msg, 'occupied')

    def cb_unknown(self, msg):
        self.update_grid(msg, 'unknown')

    def publish_frontier(self):
        self.frontier[:] = 0
        # use conv for detecting free grids adjacent to unknown grids
        grid = copy.copy(self.grid)
        grid = chainer.Variable(grid.astype(np.float32))
        max_grid = F.max_pooling_nd(grid, ksize=3, stride=1, pad=1)
        self.frontier = np.logical_and(
            max_grid.data == 2,
            self.grid == 0).astype(np.int)

        pub_marker = MarkerArray()
        frontier_marker = Marker()
        point_list = []
        for i in range(self.frontier.shape[0]):
            for j in range(self.frontier.shape[1]):
                for k in range(self.frontier.shape[2]):
                    if self.frontier[i][j][k]:
                        point = Point()
                        point.x = (i+1) * self.resolution + self.occupancy_region['occupancy_min_x']
                        point.y = (j+1) * self.resolution + self.occupancy_region['occupancy_min_y']
                        point.z = (k+1) * self.resolution + self.occupancy_region['occupancy_min_z']
                        point_list.append(point)
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
        pub_marker.markers = [frontier_marker]

        self.frontier_pub.publish(pub_marker)


if __name__ == '__main__':
    rospy.init_node('frontier_publisher')
    rospy.loginfo("start publishing frontier grids")
    fp = FrontierPublisher()
    # s = rospy.Service('publish_frontier', Empty, fp.publish_frontier)
    rospy.spin()
