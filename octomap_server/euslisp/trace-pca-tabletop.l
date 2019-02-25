#!/usr/bin/env roseus

(load "package://octomap_server/euslisp/grope-in-bag.l")


(defun pose-for-tabletop ()
  (let ((av1 nil) (av2 nil))
    (send *fetch* :reset-pose)
    (send *fetch* :torso :waist-z :joint-angle 350)
    (setq av1 (send *fetch* :angle-vector))
    ;; (send *fetch* :inverse-kinematics (make-coords :pos #f(500 0 770) :rpy (float-vector 0 pi/2 0)) :use-torso nil) ;; for banana
    (send *fetch* :inverse-kinematics (make-coords :pos #f(500 0 790) :rpy (float-vector 0 pi/2 0)) :use-torso nil)
    (setq av2 (send *fetch* :angle-vector))
    (send *ri* :angle-vector-sequence (list av1 av2) (list 3000 7000))
    (unix:sleep 10)))

(defun trace-pca-tabletop ()
  ;; trace object
  (camera-stop)
  (octomap-reset)
  (trace :direction nil :count 2)
  (print 111111)
  (trace :direction t :count 4)
  (print 222222)
  (trace :direction nil :count 2)
  (print 333333)
  ;; classify object
  (let ((region-all (make-cube 70 100 120)))
    (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
    (send region-all :translate #f(0 0 0) :local)
    (send region-all :move-coords (send *fetch* :rarm :end-coords :copy-worldcoords) (make-coords))
    (if (eq (classify-object (send *fetch* :angle-vector) (get-octo-pointcloud :type "occupied") :region region-all) :ball)
        (print :ball) ;; TODO put apple on the kitchen
        (print :cuboid)) ;; TODO put banana on the kitchen
    )
  )
