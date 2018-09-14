#!/usr/bin/env roseus

(load "package://bag_segmentation/euslisp/bag-manipulation-interface.l")
(load "package://bag_segmentation/euslisp/utils.l")

(unless (boundp '*ri*)
  (set-fetch))
(unless (boundp '*irtviewer*)
  (objects (list *fetch*)))

(defun main ()
  (send *fetch* :reset-pose)
  (send *fetch* :inverse-kinematics (make-coords :pos #f(600 0 900) :rpy (float-vector 0 pi/2 0)))
  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
  (send *ri* :wait-interpolation)

  (dotimes (i 10)
    (grope))

  )

(defun grope ()
  (let* ((ec nil) (range 150.0)
         (rand-vec-x (- (random range) (/ range 2))) (rand-vec-y (- (random range) (/ range 2.0))) (rand-vec-rot (- (random 1.0) 0.5))
         )
    (send *ri* :stop-grasp :wait t)
    (send *fetch* :inverse-kinematics (make-coords
                                       :pos (v+ (float-vector rand-vec-x rand-vec-y 0) (float-vector 600 0 900))
                                       :rpy (float-vector 0 pi/2 rand-vec-rot)))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (send *fetch* :rarm :move-end-pos #f(0 0 -200) :world :use-torso nil)
    (setq ec (send *fetch* :rarm :end-coords :copy-worldcoords))
    (detect ec (list "/left" "/right") "/top" 30)
    (unix:sleep 4)
    ))
