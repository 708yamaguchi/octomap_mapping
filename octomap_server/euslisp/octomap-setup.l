#!/usr/bin/env roseus

(defun main()
  ;; make octomap_world tf
  (ros::roseus "octomap_setup")
  (ros::rate 10)
  (do-until-key
   (ros::sleep)
   (unless (boundp '*tfb*)
     (setq *tfb* (instance ros::transform-broadcaster :init)))
   (send *tfb* :send-transform (make-coords) "base_link" "octomap_world")
   (ros::service-call "/proximitycloud_passthrough/request" (instance std_srvs::EmptyRequest :init))
   )
  )

(main)
