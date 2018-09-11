#!/usr/bin/env python

import rospy
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import ContactSensor, ContactSensorArray

if __name__ == '__main__':
    rospy.init_node('publish_fetch_contact_sensor')
    contact_sensor_array_pub = rospy.Publisher('contact_sensors_in', ContactSensorArray, latch=True)
    r = rospy.Rate(10)
    while not rospy.is_shutdown():
        msg = ContactSensorArray()
        msg.header = Header(frame_id="/octomap_world", stamp=rospy.Time.now())

        gripper_link_sensor = ContactSensor(header=Header(
            frame_id="/octomap_world", stamp=rospy.Time.now()), contact=False, link_name='gripper_link')
        l_gripper_finger_link_sensor = ContactSensor(header=Header(
            frame_id="/octomap_world", stamp=rospy.Time.now()), contact=False, link_name='l_gripper_finger_link')
        r_gripper_finger_link_sensor = ContactSensor(header=Header(
            frame_id="/octomap_world", stamp=rospy.Time.now()), contact=False, link_name='r_gripper_finger_link')
        msg.datas = [gripper_link_sensor, l_gripper_finger_link_sensor, r_gripper_finger_link_sensor]
        contact_sensor_array_pub.publish(msg)
        r.sleep()
