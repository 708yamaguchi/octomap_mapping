#!/usr/bin/env roseus

(ros::load-ros-manifest "visualization_msgs")
(ros::roseus "octomap_setup")

(defun main()
  ;; start camera_passthrough and proximity_passthrough
  (ros::wait-for-service "/proximitycloud_passthrough/request")
  (ros::service-call "/proximitycloud_passthrough/request" (instance std_srvs::EmptyRequest :init))
  (ros::wait-for-service "/camera_passthrough/request")
  (ros::service-call "/camera_passthrough/request" (instance std_srvs::EmptyRequest :init))

  ;; subscribe octomap topics
  (ros::subscribe "/free_cells_vis_array" visualization_msgs::MarkerArray #'cb "free")
  (ros::subscribe "/occupied_cells_vis_array" visualization_msgs::MarkerArray #'cb "occupied")
  (ros::subscribe "/unknown_cells_vis_array" visualization_msgs::MarkerArray #'cb "unknown")
  (ros::subscribe "/frontier_cells_vis_array" visualization_msgs::MarkerArray #'cb "frontier")
  (ros::subscribe "/potential_cells_vis_array" visualization_msgs::MarkerArray #'cb "potential")

  ;; subscribe camera_in topic once
  (one-shot-subscribe "/camera_in" sensor_msgs::Pointcloud2)

  (ros::rate 10)
  (do-until-key
   (ros::sleep)
   (ros::spin-once)
   (unless (boundp '*tfb*)
     (setq *tfb* (instance ros::transform-broadcaster :init)))
   (send *tfb* :send-transform (make-coords) "base_link" "octomap_world")
   )
  )


(defun cb (type msg)
  (ros::ros-info "subscribing topic... [~A]" type)
  )


(main)
