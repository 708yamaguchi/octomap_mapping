#!/usr/bin/env roseus

(load "package://octomap_server/euslisp/grope-with-octomap.l")
(load "package://octomap_server/euslisp/box-env2.l")

(defun main ()
  (let* ((pc occupied-pointcloud) (target-region target-region)
         (x-offset 40) (y-offset 40) (z-offset 20)
         (hand-pos nil) (x nil) (y nil) (z nil) (fetch-copy nil) (points 0) (max-points 0)
         (min-x (+ (reduce #'min (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))) x-offset))
         (max-x (- (reduce #'max (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))) x-offset))
         (min-y (+ (reduce #'min (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))) y-offset))
         (max-y (- (reduce #'max (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))) y-offset))
         (min-z (+ (reduce #'min (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices))) z-offset))
         (max-z (- (reduce #'max (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices))) z-offset)))
    (search-graspable-point pc :min-x min-x :max-x max-x :min-y min-y :max-y max-y :min-z min-z :max-z max-z :iter-count 300 :visualize-searching t)))

(objects (list *fetch*))
(format t "Press Enter key to start~%")

(do-until-key
 (x::window-main-one)
 (unix:sleep 1))

(main)
