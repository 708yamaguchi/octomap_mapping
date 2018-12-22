#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::roseus "service_call_for_rosbag")

(setq *camera-request-service* "/camera_passthrough/request")
(setq *camera-stop-service* "/camera_passthrough/stop")
(setq *proximitycloud-request-service* "/proximitycloud_passthrough/request")

;; start camera and proximity sensor
(do-until-key
 (when (eq (send (ros::time-now) :sec) 1545499614) ;; right after starting rosbag
   ;; start camera
   (ros::wait-for-service *camera-request-service*)
   (ros::service-call *camera-request-service* (instance std_srvs::EmptyRequest :init))
   ;; start proximity sensor
   (ros::wait-for-service *proximitycloud-request-service*)
   (ros::service-call *proximitycloud-request-service* (instance std_srvs::EmptyRequest :init))

(do-until-key
 ;; stop camera at proper timing
 (when (eq (send (ros::time-now) :sec) 1545499638) ;; after looking around
   (ros::wait-for-service *camera-stop-service*)
   (ros::service-call *camera-stop-service* (instance std_srvs::EmptyRequest :init))
   (return))
 (unix:sleep 1))

(exit)
