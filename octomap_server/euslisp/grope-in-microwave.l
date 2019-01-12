#!/usr/bin/env roseus

(load "package://bag_segmentation/euslisp/bag-manipulation-interface.l")
(load "package://bag_segmentation/euslisp/utils.l")
(load "package://euslib/euslisp/arrow-utils.l")

(setq *region* (list (* 0.5 1000) (* 0.8 1000) (* -0.15 1000) (* 0.15 1000) (* 0.3 1000) (* 0.6 1000)))
(setq *resolution* 10)
(let ((param-list (list "occupancy_min_x" "occupancy_max_x" "occupancy_min_y" "occupancy_max_y" "occupancy_min_z" "occupancy_max_z"))
      (count 0))
  (dolist (param param-list)
    (setq param (format nil "/~A/~A" "octomap_server_contact" param))
    (if (ros::has-param param)
      (setf (nth count *region*) (* 1000 (ros::get-param param))))
    (setq count (+ count 1)))
  (setq *min-x* (nth 0 *region*))
  (setq *max-x* (nth 1 *region*))
  (setq *min-y* (nth 2 *region*))
  (setq *max-y* (nth 3 *region*))
  (setq *min-z* (nth 4 *region*))
  (setq *max-z* (nth 5 *region*)))
(if (ros::has-param "resolution")
  (setq *resolution* (* (ros::get-param "resolution") 1000)))
(if (not (boundp '*target-region*))
  (setq *target-region* nil))

;; torso length
(setq *torso* 20)
;; direction accumulation
(setq *direction-accum* 0)

;; service name for stop/request of pointcloud from camera
(setq *camera-request-service* "/camera_passthrough/request")
(setq *camera-stop-service* "/camera_passthrough/stop")
;; service name for reset octomap
(setq *octomap-reset-service* "/octomap_server_contact/reset")

(unless (boundp '*ri*)
  (set-fetch))
(unless (boundp '*irtviewer*)
  (fetch)
  (objects (list *fetch*)))

(warning-message 3 (format nil "~%fetch's pre-pose:~%  (progn (send *fetch* :reset-pose) (send *fetch* :rarm :shoulder-y :joint-angle -80) (send *fetch* :inverse-kinematics
            (make-coords :pos (float-vector 500 0 700))
            :use-torso t
            :debug-view nil
            :min-loop 50
            :revert-if-fail nil
            :additional-nspace-list (mapcar #'(lambda (link refangle)
                                                (eval `(list link #'(lambda () (* 0.01 (- ,refangle (send ,link :joint :joint-angle)))))))
                                            (send-all (list (nth 1 (send *fetch* :rarm :joint-list))) :child-link)
                                            (list -70))))~%"))
(warning-message 3 (format nil "~%send av to *ri*:~%  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)~%"))
