#!/usr/bin/env python

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

    def update_grid(self, points, occupancy_type):
        for point in points:
            x_index = int(np.round(((point.x - self.occupancy_region['occupancy_min_x']) / self.resolution) - 1))
            y_index = int(np.round(((point.y - self.occupancy_region['occupancy_min_y']) / self.resolution) - 1))
            z_index = int(np.round(((point.z - self.occupancy_region['occupancy_min_z']) / self.resolution) - 1))
            if occupancy_type == 'free':
                self.grid[x_index][y_index][z_index] = 0
            elif occupancy_type == 'occupied':
                self.grid[x_index][y_index][z_index] = 1
            elif occupancy_type == 'unknown':
                self.grid[x_index][y_index][z_index] = 2

    def cb_free(self, msg):
        free_points = msg.markers[0].points
        self.update_grid(free_points, 'free')
        self.frame_id = msg.markers[0].header.frame_id
        self.ns = msg.markers[0].ns
        self.get_frontier()

    def cb_occupied(self, msg):
        occupied_points = msg.markers[0].points
        self.update_grid(occupied_points, 'occupied')

    def cb_unknown(self, msg):
        unknown_points = msg.markers[0].points
        self.update_grid(unknown_points, 'unknown')

    def get_frontier(self):
        # return EmptyResponse()
        # rospy.loginfo("Calculate frontier.")
        t = rospy.Time.now()
        rospy.loginfo('start get_frontier: {}.{}'.format(t.secs, t.nsecs))
        self.frontier[:] = 0
        for i in range(1, self.frontier.shape[0] - 1):
            for j in range(1, self.frontier.shape[1] - 1):
                for k in range(1, self.frontier.shape[2] - 1):
                    if self.grid[i][j][k] == 0:  # if this grid is free
                        flag = False
                        for l in [-1, 0, 1]:
                            if flag is False:
                                for m in [-1, 0, 1]:
                                    if flag is False:
                                        for n in [-1, 0, 1]:
                                            if flag is False:
                                                if self.grid[i+l][j+m][k+n] == 2:
                                                    flag = True
                                                    self.frontier[i][j][k] = 1
                                            else:
                                                break
                                    else:
                                        break
                            else:
                                break

        pub_marker = MarkerArray()
        frontier_marker = Marker()
        frontier_marker.header.stamp = rospy.Time.now()
        frontier_marker.header.frame_id = self.frame_id
        frontier_marker.ns = self.ns
        frontier_marker.type = 6
        frontier_marker.action = 2
        frontier_marker.color.r = 1.0
        frontier_marker.color.g = 0.0
        frontier_marker.color.b = 0.0
        frontier_marker.color.r = 1.0
        pub_marker.markers = [frontier_marker]

        t = rospy.Time.now()
        rospy.loginfo('end get_frontier: {}.{}'.format(t.secs, t.nsecs))

        return


if __name__ == '__main__':
    rospy.init_node('frontier_publisher')
    rospy.loginfo("fugo")
    fp = FrontierPublisher()
    # s = rospy.Service('get_frontier', Empty, fp.get_frontier)
    rospy.spin()
