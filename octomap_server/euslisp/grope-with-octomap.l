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

  (dotimes (i 30)
    (random-grope))
  )


(load "av-and-groped-pc1.l")
(setq *pc* (cadr av-and-groped-pc1))
(defun search-graspable-point (pc)
  (let (min-x max-x min-y max-y min-z max-z
        stick-l stick-r
        random-x random-y random-rot
        best-x best-y best-rot
        (gr-x 40.0) (gr-y 30.0) (gr-z 60.0) (gr-open 100.0)
        (gripper-l (send (make-cube gr-x gr-y gr-z) :locate #f(0 (/ gr-open -2) 0)))
        (gripper-r (send (make-cube gr-x gr-y gr-z) :locate #f(0 (/ gr-open 2) 0)))
        (gripper-inside (make-cube gr-x (- gr-open gr-y) gr-z))
        (point-list (send pc :point-list))
        (point-list-sorted (copy-object point-list))
        highest-z
        (inside-point 0) (max-inside-point 0)
        )
    ;; get working space
    ;; (setq min-x (ros::get-param "occupancy_min_x"))
    ;; (setq max-x (ros::get-param "occupancy_max_x"))
    ;; (setq min-y (ros::get-param "occupancy_min_y"))
    ;; (setq max-y (ros::get-param "occupancy_max_y"))
    ;; (setq min-z (ros::get-param "occupancy_min_z"))
    ;; (setq max-z (ros::get-param "occupancy_max_z"))
    (setq min-x 0.4)
    (setq max-x 0.8)
    (setq min-y -0.25)
    (setq max-y 0.25)
    (setq min-z 0.3)
    (setq max-z 0.8)

    ;; assoc the virtual gripper model
    (send gripper-l :set-color #f(1 0 0))
    (send gripper-r :set-color #f(0 1 0))
    (send gripper-inside :set-color #f(0 0 1))
    (setq stick-l (send (make-cube (make-cube gr-x gr-y (- max-z min-z))) :locate #f(0 (/ gr-open -2) (/ (+ max-z min-z) 2.0))))
    (send stick-l :set-color #f(1 0 0 0.5))
    (setq stick-r (send (make-cube (make-cube gr-x gr-y (- max-z min-z))) :locate #f(0 (/ gr-open 2) (/ (+ max-z min-z) 2.0))))
    (send stick-r :set-color #f(0 1 0 0.5))
    (send gripper-inside :assoc gripper-l)
    (send gripper-inside :assoc gripper-r)
    (send gripper-inside :assoc stick-l)
    (send gripper-inside :assoc stick-r)

    ;; sort points in the order of z-height
    ;; (sort) is destructively sort
    (sort point-list-sorted #'(lambda (x y) (> (aref x 2) (aref y 2))))

    ;; search for most graspable point
    (dotimes (i 10)
      ;; move gripper randomly and get the highest point under the gripper
      (setq random-x (+ (random (- max-x min-x)) min-x))
      (setq random-y (+ (random (- max-y min-y)) min-y))
      (setq random-rot (- (random pi) pi/2))
      (send gripper-inside :locate (float-vector random-x random-y 0) :world) ;; change position absolutely
      (send gripper-inside :orient :z random-rot) ;; change rotation absolutely
      (dolist (point point-list-sorted)
        (if (or (equal (send stick-l :insidep point) :inside)
                (equal (send stick-r :insidep point) :inside))
          (setq highest-z (aref point 0))))
      (if (null highest-z)
        (return))
      ;; move the hand down until collision with the highest point, and count the points inside the gripper.
      (send gripper-inside :dissoc stick-l)
      (send gripper-inside :dissoc stick-r)
      (send gripper-inside :locate (float-vector 0 0 (+ highest-z (/ gr-z 2.0)) ))
      ;; count the point inside of gripper-inside
      (dolist (point point-list)
        (if (equal (send gripper-inside :insidep point) :inside)
          (setq inside-point (+ inside-point 1))))
      (when (> inside-point max-inside-point)
        (setq max-inside-point inside-point)
        (setq best-x random-x)
        (setq best-y random-y)
        (setq best-rot random-rot))
      ;; redo the movement of gripper-l, gripper-r and gripper-inside
      (send gripper-inside :locate (float-vector 0 0 0))
      (send gripper-inside :assoc stick-l)
      (send gripper-inside :assoc stick-r)
      )
    ))

(defun random-grope ()
  (let* ((ec nil) (range 150.0)
         (rand-vec-x (- (random range) (/ range 2))) (rand-vec-y (- (random range) (/ range 2.0))) (rand-vec-rot (- (random 1.6) 0.8))
         )
    (send *ri* :start-grasp :wait t)
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
