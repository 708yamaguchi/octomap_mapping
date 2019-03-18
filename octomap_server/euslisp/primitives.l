#!/usr/bin/env roseus

(load "package://octomap_server/euslisp/grope-in-bag.l")

(defun center-motion ()
  (ros::spin-once)
  (let ((l-dist (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side0")) :data) cons))
        (r-dist (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side0")) :data) cons)))
    ;; centering object in gripper
    (send *fetch* :rarm :move-end-pos
          (float-vector 0 (/ (- (apply #'min l-dist) (apply #'min r-dist)) 2.0) 0) :local
          :use-torso nil)
    ;; sensing object at the tip of side0
    (ros::spin-once)
    (unix:usleep (* 100 1000))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    ))


(defun align ()
  (ros::spin-once)
  (let ((angle (/ (+ (plane-incline-side0 "/left")
                     (plane-incline-side0 "/right")) 2.0)))
    (setq angle (plane-incline-side0 "/left"))
    (ros::spin-once)
    (if (and angle (< (abs angle) 45))
        (rotate-wrist-r angle))
    (send *ri* :wait-interpolation)
  ))
