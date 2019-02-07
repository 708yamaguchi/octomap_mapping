#!/usr/bin/env roseus

(load "package://octomap_server/euslisp/grope-in-box.l")
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
    ;; setup for visualization
    (let ((cube (make-cube (- max-x min-x) (- max-y min-y) (- max-z min-z))))
      (send cube :locate (float-vector (/ (+ max-x min-x) 2.0)
                                       (/ (+ max-y min-y) 2.0)
                                       (/ (+ max-z min-z) 2.0)))
      (send cube :worldcoords)
      (objects (append (list *fetch*) (send cube :edges)))
      (send *irtviewer* :change-background #f(0 0 0))

      (send *fetch* :reset-pose)
      (send *irtviewer* :viewer :viewing :look #f(1528.4 1471.28 1304.32) #f(604.334 -129.251 631.657) #f(0 0 1))
      (send *irtviewer* :draw-objects :flush t)
      (send *irtviewer* :resize 500 500)
      (format t "Press Enter key to start~%")
      (do-until-key
       (x::window-main-one)
       (unix:sleep 1))

    (search-graspable-point pc :min-x min-x :max-x max-x :min-y min-y :max-y max-y :min-z min-z :max-z max-z :iter-count 500 :visualize-searching t))))


(main)
