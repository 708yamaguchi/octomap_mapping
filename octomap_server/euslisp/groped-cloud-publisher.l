#!/usr/bin/env roseus

(ros::roseus "groped_cloud_publisher")

(load "package://octomap_server/euslisp/av-and-groped-pc1.l")
(setq *eus-pc* (cadr av-and-groped-pc1))
;; (load "package://octomap_server/euslisp/av-and-groped-pc2.l")
;; (setq *eus-pc* (cadr av-and-groped-pc2))

(send *eus-pc* :colors
      (apply #'matrix (make-list (array-dimension (send *eus-pc* :points) 0)
                                 :initial-element #f(1 1 1))))
(setq *ros-pc*
      (make-ros-msg-from-eus-pointcloud *eus-pc*
                                        :frame "/map"
                                        :with-color :rgb ;;create 'rgb' field to PointCloud2 (not r, g, and b)
                                        ))

(defun main ()
  (ros::advertise "/groped_pointcloud" sensor_msgs::PointCloud2 1)
  (ros::rate 10)
  (do-until-key
   (ros::publish "/groped_pointcloud" *ros-pc*)
   (ros::sleep)
   )
  )
