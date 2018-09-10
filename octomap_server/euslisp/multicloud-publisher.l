#!/usr/bin/env roseus

(ros::load-ros-manifest "sernsor_msgs")
(ros::roseus "multicloud_publisher")

(defun main ()
  (let ((pc1 nil) (pc2 nil) (count 0))
    (unless (boundp '*tfb*)
      (setq *tfb* (instance ros::transform-broadcaster :init)))
    (setq pc1 (instance pointcloud :init))
    (send pc1 :points (matrix (list 1000 1000 1000)))
    (setq pc2 (instance pointcloud :init))
    (send pc2 :points (matrix (list 1000 1000 -1000)))
    (ros::advertise "/multicloud" sensor_msgs::Pointcloud2)
    (ros::rate 50)
    (do-until-key
     (setq count (+ count 1))
     (send *tfb* :send-transform (make-coords :rpy (float-vector (* count 0.01) 0 0)) "world" "octomap_world")
     (ros::publish "/multicloud" (make-ros-msg-from-eus-pointcloud pc1 :frame "/camera_1"))
     (ros::publish "/multicloud" (make-ros-msg-from-eus-pointcloud pc2 :frame "/camera_2"))
     (ros::sleep))))
