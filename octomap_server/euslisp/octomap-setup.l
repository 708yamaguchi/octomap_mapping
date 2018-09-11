#!/usr/bin/env roseus

(defun main()
  ;; make octomap_world tf
  (ros::roseus "octomap_setup")
  (ros::rate 10)
  (ros::wait-for-service "/proximitycloud_passthrough/request")
  (ros::service-call "/proximitycloud_passthrough/request" (instance std_srvs::EmptyRequest :init))
  (ros::wait-for-service "/camera_passthrough/request")
  (ros::service-call "/camera_passthrough/request" (instance std_srvs::EmptyRequest :init))
  (do-until-key
   (ros::sleep)
   (unless (boundp '*tfb*)
     (setq *tfb* (instance ros::transform-broadcaster :init)))
   (send *tfb* :send-transform (make-coords) "base_link" "octomap_world")
   )
  )

(main)
