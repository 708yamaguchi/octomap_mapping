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
  (send *fetch* :angle-vector)(send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
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
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (if (eq (classify-object (send *fetch* :angle-vector) (setq in-hand-cloud (get-octo-pointcloud :type "in-hand")) :region nil) :ball)
      (print :ball) ;; TODO put apple on the kitchen
      (print :cuboid)) ;; TODO put banana on the kitchen
  )


;; e.g. (extract-pca-feature "apple/1.l" "apple/1.txt")
(defun extract-pca-feature (in_path out_path)
  (let ((eigen-value nil))
    (with-open-file
        (in in_path :direction :input)
      (let ((read-in (read in)) (cloud nil))
        (setq cloud (eval read-in))
        (setq eigen-value (car (classify-object (send *fetch* :angle-vector)
                                                cloud :region nil)))))
    (with-open-file
        (out out_path
             :direction :output
             :if-exists :append
             :if-does-not-exist :create)
      (princ (format nil "~A, ~A, ~A"
                     (aref eigen-value 0) (aref eigen-value 1) (aref eigen-value 2))
             out))
    ))
