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

(warning-message 3 (format nil "~%fetch's pre-pose:~%  (progn (send *fetch* :reset-pose) (send *fetch* :torso :waist-z :joint-angle 200) (send *fetch* :rarm :shoulder-y :joint-angle -80) (send *fetch* :inverse-kinematics
            (make-coords :pos (float-vector 500 0 700))
            :use-torso nil
            :debug-view nil
            :min-loop 50
            :revert-if-fail nil
            :additional-nspace-list (mapcar #'(lambda (link refangle)
                                                (eval `(list link #'(lambda () (* 0.01 (- ,refangle (send ,link :joint :joint-angle)))))))
                                            (send-all (list (nth 1 (send *fetch* :rarm :joint-list))) :child-link)
                                            (list -70))))~%"))
(warning-message 3 (format nil "~%send av to *ri*:~%  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)~%"))

;; NOTE type is :setting, :main or :all
;; :setting does setting of this function, :main does main part of this function and :all does both of them.
(defun main-with-microwave
  (if (not (or (equal type :setting) (equal type :main) (equal type :all)))
    (progn (format t "argument :type is not correctly set~%") (return-from main nil)))
  (setq *direction-accum* 0)
  (let ((boxes nil) (target-region nil) x-min x-max y-min y-max z-min z-max)
    (when (or (equal type :all) (equal type :setting))
      ;; settings for experiment
      ;; below is IK program for stretching wrist_flex_joint
      (send *fetch* :reset-pose)
      (send *fetch* :torso :waist-z :joint-angle 200)
      (send *fetch* :rarm :shoulder-y :joint-angle -80)
      (send *fetch* :inverse-kinematics
            (make-coords :pos (float-vector 500 (/ (+ *min-y* *max-y*) 2.0) (/ (+ *min-z* *max-z*) 2.0)))
            :use-torso nil
            :debug-view :no-message
            :min-loop 50
            :revert-if-fail nil
            :additional-nspace-list (mapcar #'(lambda (link refangle)
                                                (eval `(list link #'(lambda () (* 0.01 (- ,refangle (send ,link :joint :joint-angle)))))))
                                            (send-all (list (nth 5 (send *fetch* :rarm :joint-list))) :child-link)
                                            (list -70))
            )
      (send *fetch* :inverse-kinematics
            (make-coords :pos (float-vector 500 (/ (+ *min-y* *max-y*) 2.0) (/ (+ *min-z* *max-z*) 2.0))) :use-torso nil) ;; 950 is tuned parameter
      ;; above is IK program for stretching wrist_flex_joint
      (send *fetch* :head :look-at (float-vector (/ (+ *max-x* *min-x*) 2.0) (/ (+ *max-y* *min-y*) 2.0) (/ (+ *max-z* *min-z*) 2.0)))
      (send *fetch* :rarm :move-end-pos #f(0 -400 0) :world :use-torso nil) ;; move hand sideway to avoid occlusion
      (send *ri* :angle-vector (send *fetch* :angle-vector) 4000)
      (send *ri* :wait-interpolation)
      (send *ri* :go-grasp :pos 0.03 :wait t)
      (unix:usleep (* 500 1000))
      (send *ri* :set-init-proximities)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 4000)
      ;; set octomap
      (camera-stop)
      (octomap-reset)
      (send *ri* :wait-interpolation)
      (camera-request)
      (unix:usleep (* 500 1000))
      ;; use camera and decide target region
      (while t ;; wait for /HSI_color_filter/boxes comes once
        (setq boxes (one-shot-subscribe "/HSI_color_filter/boxes" jsk_recognition_msgs::BoundingBoxArray))
        (when (send boxes :boxes)
          (return))
        (unix:usleep (* 100 1000)))
      (setq target-region (box-recog boxes :offset 30)) ;; offset is to ignore box edge
      ;; look at top of target region
      (send *fetch* :head :neck-p :joint-angle 20)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
      (send *ri* :wait-interpolation)
      (send *fetch* :head :look-at (float-vector (/ (+ *max-x* *min-x*) 2.0) (/ (+ *max-y* *min-y*) 2.0) (/ (+ *max-z* *min-z* 200) 2.0)))
      (send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
      (send *ri* :wait-interpolation)

      (camera-stop)
      ;; set hand to initial experiment pose
      (send *fetch* :rarm :move-end-pos #f(0 400 0) :world :use-torso nil)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 4000)
      (send *ri* :wait-interpolation))

    (if (equal type :setting)
      (return-from main t))

        ;; grasp object with octomap and grope. TODO simultaneous execution of (create-map) and (decide-grope-point) may be needed.
    ;; flowchart of this (do-until-key) is loop of following steps:
    ;; 1. first create octomap, and then try to grasp the target object with the same octomap up to 5 times.
    ;; 2. if success, lift the object and finish. else, retry from the first step.
    (while t ;; do not return from this loop until the robot succeeds to grasp the target object
     (let ((find-grasp-pointp nil) (detectp nil) (graspp nil))
       ;; create octomap by tracing environment, using proximity sensors on finger.
       (dotimes (i 30)
         (create-map target-region)
         (let ((av (send *fetch* :angle-vector)))
           (when (and (< 15 (car (decide-grope-point (get-octo-pointcloud :type "occupied") target-region :x-offset 20 :y-offset 40 :z-offset 40))) ;; TODO decide proper threshold : maybe around 30?
                      (> i 1)) ;; at least do (create-map) 2 times
             (setq find-grasp-pointp t)
             (format t "find graspable point, so start grasping~%")
             (return))
           (send *fetch* :angle-vector av)))

       ;; calculate the most graspable point. In the function below, *fetch* is set.
       (dotimes (i 5) ;; try to grasp the target object with the same octomap, up to 5 times.
         (unless find-grasp-pointp
           (decide-grope-point (get-octo-pointcloud :type "occupied") target-region :x-offset 60 :y-offset 80 :z-offset 80) ;; update *fetch* at this time
           (setq find-grasp-pointp nil))

         ;; extend hand to target object with pre-grasp motion by proximity sensors
         ;; first, move hand up
         (let ((ec-x (aref (send (send *fetch* :rarm :end-coords :copy-worldcoords) :worldpos) 0)))
           (format t "try to solve IK~%")
           ;; (send *fetch* :rarm :move-end-pos (float-vector 0 0 (- *max-z* ec-z)) :world)
           (dotimes (i 3)
             (unless (send *fetch* :rarm :move-end-pos (float-vector 100 0 0) :world)
               (return)))
           (format t "IK solved ??~%")
           (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
           (send *ri* :wait-interpolation))
         (sensor-calib) ;; calibrate sensors for grasping
         ;; second, extend hand to target object with pre-grasp motion
         (while t
           (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                              *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
             (if (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                         :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :close-thre 20)
               (progn ;; if something is in front of the gripper, return from this while loop and try to pre-grasp
                 (setq detectp t)
                 (return)))
             (progn ;; if nothing is found, retry (decide-grope-point)
               (format t "cannot find object~%")
               (setq detectp nil)
               (return))))
         ;; TODO PLEASE WRITE (pre-grasp) function
         (if detectp ;; if something is in front of gripper, do pre-grasp
           (setq graspp (pre-grasp)))
         (if graspp
           (return)))
       (if graspp ;; if pre-grasp is succeeded and the robot can grasp the target object, return from do-until-key loop
         (return))
       ))

    ;; finally, lift the target object up
    (dotimes (i 3)
      (unless (send *fetch* :rarm :move-end-pos (float-vector -100 0 0) :world)
        (return)))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 7000)
    (format t "grope succeeded~%")
    ))
