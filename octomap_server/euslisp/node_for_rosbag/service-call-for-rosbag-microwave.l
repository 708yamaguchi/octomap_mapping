#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::roseus "service_call_for_rosbag")

(setq *camera-request-service* "/camera_passthrough/request")
(setq *camera-stop-service* "/camera_passthrough/stop")
(setq *octomap-reset-service* "/octomap_server_contact/reset")
(setq *proximitycloud-request-service* "/proximitycloud_passthrough/request")

;; start camera and proximity sensor
(ros::subscribe "/camera_request" std_msgs::Empty #'camera-request)
(ros::subscribe "/camera_stop" std_msgs::Empty #'camera-stop)
(ros::subscribe "/octomap_reset" std_msgs::Empty #'octomap-reset)

(defun camera-request (msg)
   (ros::wait-for-service *camera-request-service*)
   (ros::service-call *camera-request-service* (instance std_srvs::EmptyRequest :init))
   ;; request proximity (double check)
   (ros::wait-for-service *proximitycloud-request-service*)
   (ros::service-call *proximitycloud-request-service* (instance std_srvs::EmptyRequest :init))
   )

(defun camera-stop (msg)
  (ros::wait-for-service *camera-stop-service*)
  (ros::service-call *camera-stop-service* (instance std_srvs::EmptyRequest :init))
  )

(defun octomap-reset (msg)
  (ros::wait-for-service *octomap-reset-service*)
  (ros::service-call *octomap-reset-service* (instance std_srvs::EmptyRequest :init))
  )
