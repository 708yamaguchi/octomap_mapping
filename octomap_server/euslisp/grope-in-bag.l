#!/usr/bin/env roseus

;; This program aims to grope in microwave

(load "package://bag_segmentation/euslisp/bag-manipulation-interface.l")
(load "package://bag_segmentation/euslisp/utils.l")
(load "package://euslib/euslisp/arrow-utils.l")
(load "package://octomap_server/euslisp/utils.l")

;; set working region
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
(setq *torso* 350)
;; direction accumulation
(setq *direction-accum* 0)

;; collision objects for moveit
(unless (boundp '*co*)
  (setq *co* (instance collision-object-publisher :init)))
(unless (boundp '*collision-objects*)
  (setq *collision-objects* nil))

(unless (boundp '*ri*)
  (set-fetch))
(unless (boundp '*irtviewer*)
  (fetch)
  (objects (list *fetch*)))

;; TODO asita, KOKOKARA
;; memo for pre-pose

(format t "~%fetch's pre-pose:
If you want to set fetch's prepose from reset-pose,~%")
(warning-message 3 "(prepose-for-main :type :all)~%")
(format t "If you want to set fetch's prepose from the pose groping in the bag,~%")
(warning-message 3 "(prepose-for-main :type :part)~%")
(format t "If you want to set fetch's reset-pose from the pose groping in the bag,~%")
(warning-message 3 "(prepose-for-main :type :reverse)~%")
(format t "Look at inside of the bag~%")
(warning-message 3 "(progn (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t)) (send *fetch* :head :neck-p :joint-angle 80) (send *fetch* :head :neck-y :joint-angle -90))~%")


(defun main-former ()
  ;; TODO 1. receive bag from human

  ;; TODO 2. go to kitchen

  ;; TODO 3. open the bag by using bag handle
  )


;; NOTE type is :setting, :main or :all
;; :setting does setting of this function, do (prepose-for-main :all) before doing this
;; :main does main part of this function and :all does both of them.
;; NOTE do main-former beforehand (namely, open bag and move hand above the bag)
(defun main-latter (&key (type :all))
  ;; (if (not (or (equal type :setting) (equal type :main) (equal type :all)))
  ;;   (progn (format t "argument :type is not correctly set~%") (return-from main nil)))
  ;; (unless (boundp '*co*)
  ;;   (setq *co* (instance collision-object-publisher :init)))
  ;; (send *co* :wipe-all)
  ;; (setq *direction-accum* 0)
  ;; (let ((boxes nil) (target-region nil) x-min x-max y-min y-max z-min z-max)
  ;;   (when (or (equal type :all) (equal type :setting))
  ;;     ;; settings for experiment
  ;;     ;; NOTE make sure that fetch is the pose for manipulating the bag. (use prepose-for-main)
  ;;     ;; look at bag
  ;;     ;; (prepose-for-main :type :all) or (prepose-for-main :type :part) at first
  ;;     (send *fetch* :rarm :move-end-pos #f(300 0 0) :world :use-torso nil)
  ;;     (send *fetch* :head :neck-p :joint-angle 80)
  ;;     (send *fetch* :head :neck-y :joint-angle -90)
  ;;     (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
  ;;     (send *ri* :wait-interpolation)
  ;;     ;; set octomap
  ;;     (camera-stop)
  ;;     (octomap-reset)
  ;;     (send *ri* :wait-interpolation)
  ;;     (camera-request)
  ;;     (unix:usleep (* 500 1000))
  ;;     ;; use camera and decide target region
  ;;     (while t ;; wait for /HSI_color_filter/boxes comes once
  ;;       (setq boxes (one-shot-subscribe "/attention_clipper/output/box_array" jsk_recognition_msgs::BoundingBoxArray))
  ;;       (when (send boxes :boxes)
  ;;         (return))
  ;;       (unix:usleep (* 100 1000)))
  ;;     (setq target-region (box-recog boxes :offset 0)) ;; offset is to ignore box edge
  ;;     (send target-region :translate #f(0 0 0) :world) ;; offset is to ignore green tape width on microwave
  ;;     (setq *target-region* (copy-object target-region))
  ;;     (camera-stop)

  ;;     ;; set hand to initial experiment pose ;; TODO check
  ;;     (prepose-for-main :type :part)
  ;;     (send *ri* :go-grasp :pos 0.03 :wait t)
  ;;     (unix:usleep (* 500 1000))
  ;;     (send *ri* :set-init-proximities)

  ;;     ;; set environment as collision object for moveit
  ;;     ;; (setq *collision-objects* (collision-object-for-microwave target-region))
  ;;     ;; (send *co* :wipe-all)
  ;;     ;; visualization of target region and collision objects
  ;;     ;; (objects (append (list *fetch*) *collision-objects* (send *target-region* :edges)))
  ;;     )

  ;;   (if (equal type :setting)
  ;;     (return-from main-in-bag t))

  ;; NOTE START
  (prepose-for-main :type :part)
  (sensor-calib)
  (send *ri* :start-grasp :wait t)
  (send *ri* :angle-vector
        (send *fetch* :rarm :move-end-pos #f(20 50 -150) :world :use-torso nil) 5000)
  (send *ri* :wait-interpolation)
  (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(50 0 0))
          :close-frame-list (list (list "/left" "/top") (list "/right" "/top"))
          :close-thre 20 :use-torso nil)

  ;; TODO maybed following steps need to be repeated, because grasp point is not always found at the first trial.

  ;; TODO search for objects to grasp by creating map
  (octomap-reset)
  (dotimes (i 3)
    (send *ri* :go-grasp :pos (* i 0.05) :wait t)
    (rotate-wrist-r (* 179 (* (- (mod i 2) 0.5) 2)) 10000)
    (send *ri* :wait-interpolation))

  ;; TODO search grasp pose
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (let ((roi nil) (prev-av (send *fetch* :angle-vector)) (post-av nil) (diff-av nil))
    (setq roi (make-cube 200 100 100))
    (send roi :locate (send (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(80 0 0) :local) :pos) :world)
    (send roi :worldcoords)
    (decide-grope-point (get-octo-pointcloud :type "occupied") roi :x-offset 0 :y-offset 0 :z-offset 0 :visualize-searching nil) ;; TODO pose is not suitable for manipulating bag. extract pos of grasp pose
    ;; (setq post-av (send *fetch* :angle-vector))

    ;; TODO grasp with pre-grasp
    ;; (send *ri* :start-grasp :wait t)
    ;; (setq diff-av
    ;;       (v- (progn
    ;;             (send *fetch* :angle-vector post-av)
    ;;             (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos))
    ;;           (progn
    ;;             (send *fetch* :angle-vector prev-av)
    ;;             (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos))))
    ;; (send *fetch* :rarm :move-end-pos diff-av :world :use-torso nil)
    ;; (send *fetch* :rarm :move-end-pos #f(-50 0 0) :local :use-torso nil)
    ;; (send *ri* :angle-vector (send *fetch* :angle-vector) 4000)
    ;; (send *ri* :wait-interpolation)
    ;; (pre-grasp) ;; TODO do not grasp in pre-grasp
    )

  ;; TODO classify shapes of objects. before classification, reset octomap, and create octomap by trace motion. then, make bounding box and classify the shape of the bouding box (use euclidean clustering?)
  ;; Is this really needed?
  ;; (send *fetch* :rarm :move-end-pos #f(0 0 20) :local :use-torso nil)
  ;; (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
  ;; (send *ri* :wait-interpolation)
  ;; (send *fetch* :rarm :move-end-pos #f(0 0 -40) :local :use-torso nil)
  ;; (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
  ;; (send *ri* :wait-interpolation)
  ;; (send *fetch* :rarm :move-end-pos #f(0 0 20) :local :use-torso nil)
  ;; (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
  ;; (send *ri* :wait-interpolation)

  ;; now, fetch's angle vector is output of search of grasp point
  ;; classify shapes of object by using occupied cloud in the gripper
  ;; then, grasp object and put it on the kitchen
  (let ((shape nil))
    (setq shape (classify-object (send *fetch* :angle-vector) (get-octo-pointcloud :type "occupied")))

    (send *fetch* :rarm :move-end-pos #f(-100 0 0) :local :use-torso nil)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 6000)
    (send *ri* :wait-interpolation)
    ;; (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
    ;;         :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :close-thre 20)

    (pre-grasp)

    (if (eq shape :ball)
        ;; put object into box of ball
        ;; put object into box of cuboid
        )
    )
  )

;; example
;; (load "package://octomap_server/euslisp/pointcloud-in-bag-apple.l")
;; (load "package://octomap_server/euslisp/pointcloud-in-bag-banana.l")
;; prev-av -> (search-graspable-point) -> post-av
;; (classify-object post-av occupied-pointcloud)
(defun classify-object (av occupied-pointcloud)
  (let ((region-upper (make-cube 80 80 25)) (point-upper 0)
        (region-center (make-cube 80 80 25)) (point-center 0)
        (region-lower (make-cube 80 80 25)) (point-lower 0)
        (region-all (make-cube 50 80 120)) (point-all 0)
        (occupied-point-list (send occupied-pointcloud :point-list))
        (point-in-region nil)
        (fetch-copy (copy-object *fetch*))
        )
    (send fetch-copy :angle-vector av)
    ;; (send region-upper :translate (float-vector 10 0 25) :local)
    ;; (send region-center :translate (float-vector 10 0 0) :local)
    ;; (send region-lower :translate (float-vector 10 0 -25) :local)
    (send region-all :translate (float-vector 0 0 0) :local)
    ;; (send region-upper :move-coords (send *fetch* :rarm :end-coords :copy-worldcoords) (make-coords))
    ;; (send region-center :move-coords (send *fetch* :rarm :end-coords :copy-worldcoords) (make-coords))
    ;; (send region-lower :move-coords (send *fetch* :rarm :end-coords :copy-worldcoords) (make-coords))
    (send region-all :move-coords (send *fetch* :rarm :end-coords :copy-worldcoords) (make-coords))
    ;; (send region-upper :set-color #f(1 0 0 0.5))
    ;; (send region-center :set-color #f(0 1 0 0.5))
    ;; (send region-lower :set-color #f(0 0 1 0.5))
    (send region-all :set-color #f(1 1 1 0.5))

    ;; count the number of points in each region
    ;; (dolist (point occupied-point-list)
    ;;   (let ((flag nil))
    ;;     (when (and (eq flag nil) (eq (send region-upper :insidep point) :inside))
    ;;       (setq point-upper (+ 1 point-upper))
    ;;       (setq flag t))
    ;;     (when (and (eq flag nil) (eq (send region-center :insidep point) :inside))
    ;;       (setq point-center (+ 1 point-center))
    ;;       (setq flag t))
    ;;     (when (and (eq flag nil) (eq (send region-lower :insidep point) :inside))
    ;;       (setq point-lower (+ 1 point-lower))
    ;;       (setq flag t))))
    ;; (format t "upper: ~A, center: ~A, lower: ~A~%" point-upper point-center point-lower)
    ;; (if (> (max point-upper point-center point-lower) (* (min point-upper point-center point-lower) 1.5))
    ;;   (progn (format t ":ball~%") :ball)
    ;;   (progn (format t ":cuboid~%") :cuboid))
    ;; (objects (list *fetch* region-upper region-center region-lower))
    (objects (list *fetch* region-all))

    ;; extract points in gripper
    (dolist (point occupied-point-list)
      (when (eq (send region-all :insidep point) :inside)
        (setq point-in-region (append point-in-region (list point)))))
    (if (eq point-in-region nil)
        (format t "cannot find target points !!~%"))
    ;; PCA
    (let ((x nil) (y nil) (z nil) (eigen-vector nil) (covariance-mat nil) (temp nil))
      (dolist (point point-in-region)
        (setq x (append x (list (aref point 0))))
        (setq y (append y (list (aref point 1))))
        (setq z (append z (list (aref point 2)))))
      (setq temp (eigen-decompose (covariance-matrix (list x y z))))
      (print temp)
      (setq eigen-vector (car temp))
      (setq covariance-mat (cadr temp))
      (format t "first variance direction: ~A~%"
              (float-vector (aref covariance-mat 0 0)
                            (aref covariance-mat 1 0)
                            (aref covariance-mat 2 0)))
      (/ (aref eigen-vector 0) (apply #'+ (coerce eigen-vector cons))))
    )
  )

;; NOTE type is :setting, :main or :all
;; :setting does setting of this function, do (prepose-for-main :all) before doing this
;; :main does main part of this function and :all does both of them.
;; NOTE do pre-pose before executing this function
(defun main-in-bag (&key (type :all))
  (if (not (or (equal type :setting) (equal type :main) (equal type :all)))
    (progn (format t "argument :type is not correctly set~%") (return-from main nil)))
  (unless (boundp '*co*)
    (setq *co* (instance collision-object-publisher :init)))
  (send *co* :wipe-all)
  (setq *direction-accum* 0)
  (let ((boxes nil) (target-region nil) x-min x-max y-min y-max z-min z-max)
    (when (or (equal type :all) (equal type :setting))
      ;; settings for experiment
      ;; NOTE make sure that fetch is the pose for manipulating the bag. (use prepose-for-main)
      ;; look at bag
      ;; (prepose-for-main :type :all) or (prepose-for-main :type :part) at first
      (send *fetch* :rarm :move-end-pos #f(300 0 0) :world :use-torso nil)
      (send *fetch* :head :neck-p :joint-angle 80)
      (send *fetch* :head :neck-y :joint-angle -90)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
      (send *ri* :wait-interpolation)
      ;; set octomap
      (camera-stop)
      (octomap-reset)
      (send *ri* :wait-interpolation)
      (camera-request)
      (unix:usleep (* 500 1000))
      ;; use camera and decide target region
      (while t ;; wait for /HSI_color_filter/boxes comes once
        (setq boxes (one-shot-subscribe "/attention_clipper/output/box_array" jsk_recognition_msgs::BoundingBoxArray))
        (when (send boxes :boxes)
          (return))
        (unix:usleep (* 100 1000)))
      (setq target-region (box-recog boxes :offset 0)) ;; offset is to ignore box edge
      (send target-region :translate #f(0 0 0) :world) ;; offset is to ignore green tape width on microwave
      (setq *target-region* (copy-object target-region))
      (camera-stop)

      ;; set hand to initial experiment pose ;; TODO check
      (prepose-for-main :type :part)
      (send *ri* :go-grasp :pos 0.03 :wait t)
      (unix:usleep (* 500 1000))
      (send *ri* :set-init-proximities)

      ;; set environment as collision object for moveit
      ;; (setq *collision-objects* (collision-object-for-microwave target-region))
      ;; (send *co* :wipe-all)
      ;; visualization of target region and collision objects
      ;; (objects (append (list *fetch*) *collision-objects* (send *target-region* :edges)))
      )

    (if (equal type :setting)
      (return-from main-in-bag t))

    ;; grasp object with octomap and grope. TODO simultaneous execution of (create-map) and (decide-grope-point) may be needed.
    ;; flowchart of this (do-until-key) is loop of following steps:
    ;; 1. first create octomap, and then try to grasp the target object with the same octomap up to 5 times.
    ;; 2. if success, lift the object and finish. else, retry from the first step.
    (while t ;; do not return from this loop until the robot succeeds to grasp the target object
     (let ((find-grasp-pointp nil) (detectp nil) (graspp nil))
       ;; create octomap by tracing environment, using proximity sensors on finger.
       (dotimes (i 30)
         (create-map target-region) ;; TODO check
         (let ((av (send *fetch* :angle-vector)))
           ;; TODO decide proper threshold
           ;; NOTE pointcloud is decimated now
           (when (and (< 15 (car (decide-grope-point (get-octo-pointcloud :type "occupied") target-region :x-offset 0 :y-offset 0 :z-offset 0)))
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
         ;; first, withdraw hand from microwave
         (let ((ec-x (aref (send (send *fetch* :rarm :end-coords :copy-worldcoords) :worldpos) 0))
               (ec-y (aref (send (send *fetch* :rarm :end-coords :copy-worldcoords) :worldpos) 1))
               (ec-z (aref (send (send *fetch* :rarm :end-coords :copy-worldcoords) :worldpos) 2))
               (av-orig (send *fetch* :angle-vector)))
           (format t "try to solve IK~%")
           ;; (send *fetch* :rarm :move-end-pos (float-vector 0 0 (- *max-z* ec-z)) :world)
           ;; (send *co* :wipe-all)
           (send *fetch* :rarm :move-end-pos #f(0 0 100) :world)
           (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
           (send *ri* :wait-interpolation)
           (send *fetch* :rarm :move-end-pos (float-vector
                                              (- (aref (send target-region :centroid) 0) ec-x)
                                              (- (aref (send target-region :centroid) 1) ec-y)
                                              50)
                 :world)
           (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
           (send *ri* :wait-interpolation)
           (send *fetch* :angle-vector av-orig)
           ;; (dolist (object *collision-objects*)
           ;;   (send *co* :add-object object))
           ;; TODO here, maybe default pose for bag should be used
           (dotimes (i 3)
             (unless (send *fetch* :rarm :move-end-pos (float-vector 0 0 100) :world)
               (return)))
           (format t "IK solved ??~%")
           (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
           (send *ri* :wait-interpolation)
           (send *co* :wipe-all))
         (sensor-calib) ;; calibrate sensors for grasping

         ;; second, extend hand to target object with pre-grasp motion
         (while t
           (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                              *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 0)
             (if (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                         :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :close-thre 20)
               (progn ;; if something is in front of the gripper, return from this while loop and try to pre-grasp
                 (setq detectp t)
                 (return)))
             (progn ;; if nothing is found, retry (decide-grope-point)
               (format t "cannot find object~%")
               (setq detectp nil)
               (return))))
         (if detectp ;; if something is in front of gripper, do pre-grasp
           (let ((angle-list nil) (max-angle nil) (min-angle nil) (now-angle nil)
                 (smaller-target-region nil) (offset 50)
                 )
             (setq now-angle (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t)))
             ;; TODO make target-region smaller and changd collision-objects temporally and locally.
             ;; TODO in order to avoid collision when rotating hand.
             ;; (setq smaller-target-region (make-cube (- (x-of-cube target-region) offset)
             ;;                                        (- (y-of-cube target-region) offset)
             ;;                                        (- (z-of-cube target-region) offset)))
             ;; (send smaller-target-region :move-coords (send target-region :copy-worldcoords) (make-coords))
             ;; (setq angle-list (pqp-with-hand (collision-object-for-microwave smaller-target-region)))
             ;; (setq max-angle (car angle-list))
             ;; (setq min-angle (cadr angle-list))
             ;; ;; (setq graspp (pre-grasp (max (- max-angle 5) now-angle) (min (+ min-angle 5) now-angle)))
             ;; (setq graspp (pre-grasp max-angle min-angle))
             (setq graspp (pre-grasp))
             ))
         (if graspp
           (return)))
       (if graspp ;; if pre-grasp is succeeded and the robot can grasp the target object, return from do-until-key loop
         (return))
       ))

    ;; finally, lift the target object up
    ;; (send *co* :wipe-all)
    ;; (dolist (object *collision-objects*)
    ;;   (send *co* :add-object object))
    (dotimes (i 3)
      (if (send *fetch* :rarm :move-end-pos (float-vector 0 0 100) :world)
        (if (send *ri* :angle-vector (send *fetch* :angle-vector) 7000)
          nil
          (return))
        (return))
      (send *ri* :wait-interpolation))
    (format t "grope succeeded~%")
    ;; (send *co* :wipe-all)

    ;; TODO put object on kitchen
    ;; make trajectory to avoid kitchen
    ;; TODO put object on kitchen
    ))

;; TODO: what method is best solution to create map?
(defun create-map (target-region)
  (camera-stop)
  (unless target-region
    (format t "target-region is not set~%")
    (break))
  (format t "*direction-accum*: ~A~%" *direction-accum*)
  (let ((target-coords (make-coords)) ;; where to move fetch's hand
        (direction (rad2deg (random pi))) (count 0) (unknown-dist nil))
    (send *ri* :go-grasp :pos 0.03 :wait t)
    ;; first, moving hand up
    (format t "set prepose for main experiment. preparing...~%")
    ;; (unless *collision-objects*
    ;;   (setq *collision-objects* (collision-object-for-microwave target-region)))
    (prepose-for-main :type :part) ;; NOTE use moveit and collision object in this function. now, *collision-objects* is nil

    ;; calculate where to grope, during moving hand up (time saving)
    (let* ((max-val -1) (max-place nil) (adjacent-max-val -1) (adjacent-max-place nil)
           (min-x (reduce #'min (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))))
           (max-x (reduce #'max (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))))
           (min-y (reduce #'min (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))))
           (max-y (reduce #'max (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))))
           (min-z (reduce #'min (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices))))
           (max-z (reduce #'max (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices))))
           (min-x-round (round min-x))
           (max-x-round (round max-x))
           (min-y-round (round min-y))
           (max-y-round (round max-y))
           (min-z-round (round min-z))
           (max-z-round (round max-z))
           (block-size 50.0)
           (diff-x (/ (- (- max-x-round min-x-round) (* (floor (/ (- max-x-round min-x-round) block-size)) block-size)) 2.0))
           (diff-y (/ (- (- max-y-round min-y-round) (* (floor (/ (- max-y-round min-y-round) block-size)) block-size)) 2.0))
           (diff-z (/ (- (- max-z-round min-z-round) (* (floor (/ (- max-z-round min-z-round) block-size)) block-size)) 2.0))
           )
      ;; calculate position of hand, x and y
      ;; correct min/max-x/y/z in order to adapt to (unknown-distribution)
      (setq min-x-round (+ min-x-round diff-x))
      (setq max-x-round (- max-x-round diff-x))
      (setq min-y-round (+ min-y-round diff-y))
      (setq max-y-round (- max-y-round diff-y))
      (setq min-z-round (+ min-z-round diff-z))
      (setq max-z-round (- max-z-round diff-z))
      (format t "target bb region:~%x: ~A ~A, y: ~A ~A, z: ~A ~A, size: (~A, ~A, ~A)~%"
              min-x max-x min-y max-y min-z max-z (- max-x min-x) (- max-y min-y) (- max-z min-z))
      (format t "clipped bb region:~%x: ~A ~A, y: ~A ~A, z: ~A ~A, size: (~A, ~A, ~A)~%"
              min-x-round max-x-round min-y-round max-y-round min-z-round max-z-round (- max-x-round min-x-round) (- max-y-round min-y-round) (- max-z-round min-z-round))
      (ros::spin-once)
      ;; TODO some offset may be needed for creating unknown-distribution
      ;; TODO block size maybe should be set smaller (e.g. 30)
      (setq unknown-dist (unknown-distribution (get-octo-pointcloud :type "occupied") (get-octo-pointcloud :type "unknown")
                                               :min-x min-x-round :max-x max-x-round
                                               :min-y min-y-round :max-y (- max-y-round 50) ;; 50 is offset to avoid door of microwave
                                               :min-z min-z-round :max-z max-z-round
                                               :block-size block-size))
      (break)

      ;; calculate most unknown area
      (dotimes (i (car (array-dimensions unknown-dist)))
        (dotimes (j (cadr (array-dimensions unknown-dist)))
          (when (< max-val (aref unknown-dist i j))
            (setq max-val (aref unknown-dist i j))
            (setq max-place (list i j)))))

      ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
      ;; Start, replace: x->y, y->z ;;
      ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
      ;; calculate angle of wrist, and record the accumulated rotation angle and reset the angle regularly
      (dotimes (i 4)
        (let ((x nil) (y nil))
          (cond ((eq i 0) (setq x 1) (setq y 0))
                ((eq i 1) (setq x 0) (setq y 1))
                ((eq i 2) (setq x -1) (setq y 0))
                ((eq i 3) (setq x 0) (setq y -1)))
          ;; when ((+ (car max-place) x), (+ (cadr max-place) y)) is not out of unknown-dist
          (when (not (or (< (+ (car max-place) x) 0) (<= (car (array-dimensions unknown-dist)) (+ (car max-place) x))
                         (< (+ (cadr max-place) y) 0) (<= (cadr (array-dimensions unknown-dist)) (+ (cadr max-place) y))))
            (when (< adjacent-max-val (aref unknown-dist (+ (car max-place) x) (+ (cadr max-place) y)))
              (setq adjacent-max-val (aref unknown-dist (+ (car max-place) x) (+ (cadr max-place) y)))
              (setq adjacent-max-place (list x y))))))
      ;; sometimes groping randomly
      (let* ((m (make-random-state t)) ;; set max-place randomly at probability of 0.5
             (random-x (random (car (array-dimensions unknown-dist)) m))
             (random-y (random (cadr (array-dimensions unknown-dist)) m))
             direction-arrow)
        (when (< (random 1.0 m) 0.5) ;; sometimes move hand randomly, else, move hand based on unknown distribution
          (format t "randomly decided grope starting position~%")
          (setq max-place (list random-x random-y))
          (if (< (random 1.0 m) 0.5) ;; if t, move hand in x-axis. if nil, move hand in y-axis
            (if (<= random-x (/ (car (array-dimensions unknown-dist)) 2))
              (setq adjacent-max-place (list 1 0))
              (setq adjacent-max-place (list -1 0)))
            (if (<= random-y (/ (cadr (array-dimensions unknown-dist)) 2))
              (setq adjacent-max-place (list 0 1))
              (setq adjacent-max-place (list 0 -1)))))
        ;; move target-coords to max-place
        (setf (aref (send target-coords :worldpos) 0) min-x)
        (setf (aref (send target-coords :worldpos) 1) (+ min-y (/ block-size 2.0) (* (car max-place) block-size)))
        (setf (aref (send target-coords :worldpos) 2) (+ min-z (/ block-size 2.0) (* (cadr max-place) block-size)))
        ;; calculate direction of trace motion
        (setq direction (+ 90 (rad2deg (atan2 (cadr adjacent-max-place) (car adjacent-max-place)))))
        (print (list max-place direction adjacent-max-place))
        ;; visualize the direction
        (setq direction-arrow
              (make-arrow
               (float-vector (+ *min-x* 100)
                             (+ (* block-size (+ (car max-place) 0.5)) min-y-round)
                             (+ (* block-size (+ (cadr max-place) 0.5)) min-z-round)) ;; pos
               (float-vector 0
                             (* block-size (car adjacent-max-place))
                             (* block-size (cadr adjacent-max-place))) ;; vec
               :color #f(1 0 0)))
        (objects direction-arrow)))
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
    ;; END, replace: x->y, y->z ;;
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

    ;; from now, move and rotate hand
    (send *ri* :wait-interpolation) ;; calculate where to grope during arm movement (time saving)
    ;; calibration proximity sensor's param every time before groping.
    (sensor-calib)
    ;; move hand above the target object
    (send *fetch* :torso :waist-z :joint-angle 290)
    (send target-coords :translate (float-vector (- 500 (aref (send target-coords :pos) 0)) 0 0) :world)
    (send *fetch* :inverse-kinematics target-coords :use-torso nil)
    ;; calculate accumulation of rotation angle, to untie USB cable
    (setq direction (round direction)) ;; nearest integer
    (when (and (<= (+ direction *direction-accum*) -270) (<= direction 0))
      (setq direction (+ 360 direction)))
    (when (and (<= 270 (+ direction *direction-accum*)) (<= 0 direction))
      (setq direction (+ -360 direction)))
    (setq *direction-accum* (+ direction *direction-accum*))
    ;; rotate arm twice, to select rotation direction
    (let ((av1 nil) (av2 nil) (av3 nil))
      (setq av1 (send *fetch* :angle-vector))
      (send *fetch* :rarm :wrist-r :joint-angle (+ (send *fetch* :rarm :wrist-r :joint-angle) (* 0.5 direction)))
      (setq av2 (send *fetch* :angle-vector))
      (send *fetch* :rarm :wrist-r :joint-angle (+ (send *fetch* :rarm :wrist-r :joint-angle) (* 0.5 direction)))
      (setq av3 (send *fetch* :angle-vector))
    (send *ri* :angle-vector-sequence (list av1 av2 av3) (list 6000 1500 1500))
    (unix:sleep 9)
    )
    ;; move hand down to target coords ;; these functions may not be needed any more
    ;; (format t "move hand to target coords~%")
    ;; (send *fetch* :torso :waist-z :joint-angle 50)
    ;; (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    ;; (send *ri* :wait-interpolation)

    (format t "start tracing~%")
    ;; move hand until detecting object. move-end-pos 10cm, 10cm, 10cm ...  gradually.
    (while (< count 3) ;; vertical hand move and horizontal hand move, alternately
      ;; move hand down
      (while t
        (format t "count: ~A~%" count)
        (format t "move hand down~%")
        (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                           *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
          (progn
            (format t "move hand until detecting object~%")
            (let ((detected-frame nil) (min-distance nil)
                  (hand-cover-thick 25)
                  ;; when :gripper :position is 0.3, gripper-handside-distance is about 25
                  (gripper-handside-distance (- 35 (* 0.5 1000 (- (send *ri* :gripper :position) 0.01)))))
              ;; check if there is object near top of the grippers.
              (if (setq detected-frame (cadr (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                                                     :close-frame-list (list (list "/left" "/top") (list "/right" "/top")
                                                                             (list "/left" "/side1") (list "/right" "/side1")
                                                                             (list "/left" "/side2") (list "/right" "/side2"))
                                                     :close-thre (list 30 30
                                                                       hand-cover-thick hand-cover-thick
                                                                       (+ 5 gripper-handside-distance) ;; +5 is safety margin
                                                                       (+ 5 gripper-handside-distance))
                                                     :tm 7000)))
                (cond
                 ((equal detected-frame (list "/left" "/top")) ;; object is near left top
                  (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/top")) :data) cons)) 40) ;; both left and right finger top is near object, start moving horizontally
                    (format t "start moving horizontally~%")
                    (return))
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (+ gripper-handside-distance 20) 0))
                                :close-frame-list (list (list "/right" "/side2")) :close-thre gripper-handside-distance :tm 3000)
                    (format t "something is right of my gripper~%"))
                  (return))
                 ((equal detected-frame (list "/right" "/top")) ;; object is near right top
                  (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/top")) :data) cons)) 40) ;; both left and right finger top is near object, start moving horizontally
                    (format t "start moving horizontally~%")
                    (return))
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (* -1 (+ gripper-handside-distance 20)) 0))
                                :close-frame-list (list (list "/left" "/side2")) :close-thre gripper-handside-distance :tm 3000)
                    (format t "something is left of my finger~%"))
                  (return))
                 ((or (equal detected-frame (list "/left" "/side1")) (equal detected-frame (list "/right" "/side1"))) ;; object is near side1
                  (ros::spin-once)
                  (setq min-distance (min (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side1")) :data) cons))
                                          (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side1")) :data) cons))))
                  (format t "detect object at side1, so move hand~%")
                  ;; distance between gripper's side1 and back of the hand is about 60mm. move hand at least 20mm.
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 0 (* -1 (max 20 (- hand-cover-thick min-distance)))))
                                :close-frame-list (list (list "/left" "/side3") (list "/right" "/side3")) :close-thre 30 :tm 3000)
                    (format t "cannot move hand forward any more~%")
                    (return)))
                 ((equal detected-frame (list "/left" "/side2"))
                  ;; (ros::spin-once)
                  (setq min-distance (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side2")) :data) cons)))
                  (format t "detect object at left side2, so move hand~%")
                  ;; distance between left gripper's side2 and side of the hand is about 35mm (at start-grasp state). move hand at least 10mm.
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (max 20 (- gripper-handside-distance min-distance)) 0))
                                :close-frame-list (list (list "/right" "/side2")) :close-thre gripper-handside-distance :tm 3000)
                    (format t "cannot move hand rightside any more~%")
                    (return)))
                 ((equal detected-frame (list "/right" "/side2"))
                  ;; (ros::spin-once)
                  (setq min-distance (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side2")) :data) cons)))
                  (format t "detect object at right side2, so move hand~%")
                  ;; distance between left gripper's side2 and side of the hand is about 35mm (at start-grasp state). move hand at least 10mm.
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (* -1 (max 20 (- gripper-handside-distance min-distance))) 0))
                                :close-frame-list (list (list "/left" "/side2")) :close-thre gripper-handside-distance :tm 3000)
                    (format t "cannot move hand leftside any more~%")
                    (return)))
                 ))))
          (progn
            (format t "robot hand is outside of target region~%")
            (return-from create-map t))))
      ;; move hand horizontally
      (while t
        (format t "count: ~A~%" count)
        (format t "move hand sideway~%")
        (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                           *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
          (progn
            (format t "move hand until missing object~%")
            (let ((result (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(0 0 -100))
                                    :close-frame-list (list (list "/left" "/side3") (list "/right" "/side3"))
                                    :away-frame-list (list (list "/left" "/top") (list "/right" "/top"))
                                    :close-thre 50 :away-thre 50)))
              (when (equal result :ik-not-solved)
                (return-from create-map t))
              (when (equal (car result) :close-frame-list)
                (format t "find something around gripper~%")
                (return))
              (when (equal (car result) :away-frame-list)
                (format t "missing object at the top of the gripper~%")
                (send *fetch* :rarm :move-end-pos #f(0 0 -40) :local)
                (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
                (send *ri* :wait-interpolation)
                (return))))
          (progn
            (format t "robot hand is outside of target region~%")
            (return-from create-map t))))
      (setq count (+ 1 count))
      )

    ;; at last, move hand down
    (format t "at last, move hand down~%")
    (while t
      (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                         *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
        (progn
          (format t "move hand until detecting object~%")
          ;; if side1 is near object, stop moving hand down
          (ros::spin-once)
          (if (or (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side1")) :data) cons)) 50)
                  (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side1")) :data) cons)) 50)
                  (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side2")) :data) cons)) 50)
                  (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side2")) :data) cons)) 50))
            (progn (format t "detecting object at the side of the gripper ~%") (return-from create-map t)))
          (if (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                      :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :close-thre 30)
            (return)))
        (progn
          (format t "robot hand is outside of target region~%")
          (return-from create-map t))))
    ))


;; before calling this function, necessary to subscribe "/unknown(frontier)_cells_vis_array"
;; (load "package://octomap_server/euslisp/occupied-and-unknown-cloud1.l")
;; (load "package://octomap_server/euslisp/occupied-and-unknown-cloud2.l")
;; only occupied and unknown pointcloud can be gotten from this function.

;; for debug, call (search-graspable-point *pc1)
(load "package://octomap_server/euslisp/av-and-groped-pc1.l")
(load "package://octomap_server/euslisp/av-and-groped-pc2.l")
(setq *pc1* (cadr av-and-groped-pc1))
(setq *pc2* (cadr av-and-groped-pc1))

(defun search-graspable-point (pc &key (min-x 300) (max-x 800) (min-y -300) (max-y 300) (min-z 400) (max-z 1000) (iter-count 1000) (visualize-searching nil))
  (print (list min-x max-x min-y max-y min-z max-z))
  (let* (stick-l stick-r stick-center
         random-x random-y random-rot
         best-x best-y best-z best-rot
         (gr-x 40.0) (gr-y 30.0) (gr-z 60.0) (gr-open 100.0) (hand-x 70.0)
         (gripper-l (send (make-cube gr-x gr-y gr-z) :locate (float-vector 0 (/ gr-open -2) 0)))
         (gripper-r (send (make-cube gr-x gr-y gr-z) :locate (float-vector 0 (/ gr-open 2) 0)))
         (gripper-inside (make-cube gr-x (- gr-open gr-y) gr-z))
         (point-list (send pc :point-list))
         (point-list-sorted nil) (point-list-inside nil) (point-list-outside nil)
         highest-z
         (inside-point 0) (max-inside-point 0) (best-pc-inside-gripper nil) (best-pc-outside-gripper nil)
         (pc-inside-gripper nil) (pc-outside-gripper nil)
         ;; (viewpoint #f(1528.4 1471.28 1304.32))
         (viewpoint #f(869.451 -2358.41 1634.45))
         ;; (viewtarget #f(604.334 -129.251 631.657))
         (viewtarget #f(34.762 -65.1178 547.884))
         )
    (send *fetch* :torso :waist-z :joint-angle *torso*)
    (send *fetch* :stop-grasp)

    (let ((cube (make-cube (- max-x min-x) (- max-y min-y) (- max-z min-z))) (region nil))
      (send cube :locate (float-vector (/ (+ max-x min-x) 2.0)
                                       (/ (+ max-y min-y) 2.0)
                                       (/ (+ max-z min-z) 2.0)) :world)
      (send cube :worldcoords)
      (setq region (send cube :edges))

      ;; assoc the virtual gripper model
      (send gripper-l :set-color #f(1 0 0))
      (send gripper-r :set-color #f(0 1 0))
      (send gripper-inside :set-color #f(0 0 1))
      ;; stick is the xy-plane field occupied by gripper
      (setq stick-l (send (make-cube gr-x gr-y (- max-z min-z)) :locate (float-vector 0 (/ gr-open -2) (/ (+ max-z min-z) 2.0))))
      (send stick-l :set-color #f(1 0 0 0.5))
      (setq stick-r (send (make-cube gr-x gr-y (- max-z min-z)) :locate (float-vector 0 (/ gr-open 2) (/ (+ max-z min-z) 2.0))))
      (send stick-r :set-color #f(0 1 0 0.5))
      (setq stick-center (send (make-cube hand-x (- gr-open gr-y) (- max-z min-z)) :locate (float-vector 0 0 (/ (+ max-z min-z) 2.0))))
      (send stick-center :set-color #f(0 0 1 0.5))
      (send gripper-inside :assoc gripper-l)
      (send gripper-inside :assoc gripper-r)
      (send gripper-inside :assoc stick-l)
      (send gripper-inside :assoc stick-r)
      (send gripper-inside :assoc stick-center)

      ;; sort points in the order of z-height
      ;; exclude points which is out of target region
      (dolist (point point-list)
        (setq point-list-sorted (append point-list-sorted (list point)))
        (if (and (<= min-x (aref point 0) max-x) (<= min-y (aref point 1) max-y) (<= min-z (aref point 2) max-z))
          (setq point-list-inside (append point-list-inside (list point)))
          (setq point-list-outside (append point-list-outside (list point)))))
      (setq point-list-sorted (copy-object point-list-sorted))
      ;; (sort) is destructively sort
      (sort point-list-sorted #'(lambda (x y) (> (aref x 2) (aref y 2))))
      ;; search for most graspable point
      (dotimes (i iter-count)
        ;; reset variables
        (setq highest-z nil)
        (setq inside-point 0)
        ;; move gripper randomly and get the highest point under the gripper
        ;; update objects related counting pointcloud
        (send stick-l :worldcoords)
        (send stick-r :worldcoords)
        (send stick-center :worldcoords)
        (send gripper-inside :worldcoords)
        ;; set random point to grope
        (setq random-x (+ (random (- max-x min-x)) min-x))
        (setq random-y (+ (random (- max-y min-y)) min-y))
        (setq random-rot (- (random pi) pi/2))
        (send gripper-inside :locate (float-vector random-x random-y 0) :world) ;; change position absolutely
        (send gripper-inside :orient random-rot :z :world) ;; change rotation absolutely
        ;; calc point at which the gripper and the hand collides with pointcloud
        (let ((l/r-flag nil) (center-flag nil))
          (dolist (point point-list-sorted)
            (when (and (not l/r-flag)
                       (or (equal (send stick-l :insidep point) :inside)
                           (equal (send stick-r :insidep point) :inside)))
              (if highest-z
                (setq highest-z (max highest-z (aref point 2)))
                (setq highest-z (aref point 2)))
              (setq l/r-flag t)
              (if (and l/r-flag center-flag)
                (return)))
            (when (and (not center-flag)
                       (equal (send stick-center :insidep point) :inside))
              (if highest-z
                (setq highest-z (max highest-z (- (aref point 2) gr-z)))
                (setq highest-z (- (aref point 2) gr-z)))
              (setq center-flag t)
              (if (and l/r-flag center-flag)
                  (return)))))

        (when highest-z
          ;; move the hand down until collision with the highest point, and count the points inside the gripper.
          (send gripper-inside :dissoc stick-l)
          (send gripper-inside :dissoc stick-r)
          (send gripper-inside :dissoc stick-center)
          (send gripper-inside :locate (float-vector 0 0 (+ highest-z (/ gr-z 2.0))) :local)
          ;; update objects related counting pointcloud
          (send gripper-inside :worldcoords)
          ;; count the point inside of gripper-inside
          (let ((point-list-inside-gripper nil)
                (point-list-outside-gripper (copy-object point-list-outside)))
            (dolist (point point-list-inside)
              (if (equal (send gripper-inside :insidep point) :inside)
                (progn
                  (setq inside-point (+ inside-point 1))
                  (setq point-list-inside-gripper (append point-list-inside-gripper (list point))))
                (setq point-list-outside-gripper (append point-list-outside-gripper (list point)))))

            (let ((pc-inside-gripper nil) (pc-outside-gripper nil))
              ;; for visualization
              (when visualize-searching
                (send *fetch* :angle-vector #f(350.0 -69.9524 -47.1977 -66.0171 104.822 32.2963 69.0654 122.165 0.0 0.0)) ;; pre-pose for IK
                (let ((coords (make-coords :pos (float-vector random-x random-y (+ highest-z (/ gr-z 2.0)))
                                           :rpy (float-vector 0 pi/2 (* -1 random-rot)))))
                  (send coords :rotate (* pi 0.1) :x :world)
                  (send *fetch* :inverse-kinematics
                        coords
                        :use-torso nil))
              ;; (send *irtviewer* :objects (append (list *fetch* pc) region))
                (if point-list-inside-gripper
                  (progn (setq pc-inside-gripper (instance pointcloud :init :points
                                                           (apply #'matrix point-list-inside-gripper)))
                         (send pc-inside-gripper :point-size 4.0)
                         (send pc-inside-gripper :colors (apply #'matrix (make-list (length point-list-inside-gripper)
                                                                                    :initial-element #f(0.3 1 0.3)))))
                  (setq pc-inside-gripper (instance pointcloud :init)))
                (if point-list-outside-gripper
                  (progn (setq pc-outside-gripper (instance pointcloud :init :points
                                                            (apply #'matrix point-list-outside-gripper)))
                         (send pc-outside-gripper :point-size 4.0)
                         (send pc-outside-gripper :colors (apply #'matrix (make-list (length point-list-outside-gripper)
                                                                                     :initial-element #f(0 0 1)))))
                  (setq pc-outside-gripper (instance pointcloud :init)))
                ;; (send *irtviewer* :redraw)
                (send *irtviewer* :objects (append (list *fetch* pc-inside-gripper pc-outside-gripper) region))
                (send *irtviewer* :viewer :viewing :look viewpoint viewtarget #f(0 0 1)) ;; fix camera position and posture
                ;; (send *irtviewer* :change-background (float-vector (/ inside-point 30.0) 0 0)) ;; 30.0 is given by human, tuned parameter.
                (send *irtviewer* :draw-objects :flush t)
                ;; (send *irtviewer* :viewer :viewsurface :string 5 20 (format nil "points: "))
                ;; (send *irtviewer* :viewer :viewsurface :string 5 50 (format nil "~A" inside-point))
                (send *irtviewer* :viewer :viewsurface :string 30 20 (format nil "Grids in"))
                (send *irtviewer* :viewer :viewsurface :string 30 50 (format nil "gripper: ~A" inside-point))
                (send *irtviewer* :viewer :viewsurface :flush))
                ;; (objects (append (list *fetch* pc stick-l stick-r stick-center gripper-inside) region))
                ;; TODO (maybe bug) this when block should be out of (when visualize-searching). See grope-in-microwave.l
                (when (> inside-point max-inside-point)
                  (setq max-inside-point inside-point)
                  (setq best-x random-x)
                  (setq best-y random-y)
                  (setq best-z highest-z)
                  (setq best-rot random-rot)
                  (setq best-pc-inside-gripper (copy-object pc-inside-gripper))
                  (setq best-pc-outside-gripper (copy-object pc-outside-gripper))
                  )))
          ;; redo the movement of gripper-l, gripper-r and gripper-inside
          (send gripper-inside :locate (float-vector 0 0 0) :world)
          (send stick-l :locate (float-vector 0 (/ gr-open -2.0) (/ (+ max-z min-z) 2.0)) :world)
          (send stick-r :locate (float-vector 0 (/ gr-open 2.0) (/ (+ max-z min-z) 2.0)) :world)
          (send stick-center :locate (float-vector 0 0 (/ (+ max-z min-z) 2.0)) :world)
          (send gripper-inside :orient 0 :z :world) ;; change rotation absolutely
          (send stick-l :orient 0 :z :world)
          (send stick-r :orient 0 :z :world)
          (send stick-center :orient 0 :z :world)
          (send gripper-inside :assoc stick-l)
          (send gripper-inside :assoc stick-r)
          (send gripper-inside :assoc stick-center)
          ))
      (format t "max-inside-point:~A~%" max-inside-point)
      (if best-z
        (progn
          (send *fetch* :angle-vector #f(350.0 -69.9524 -47.1977 -66.0171 104.822 32.2963 69.0654 122.165 0.0 0.0)) ;; pre-pose for IK
          (let ((coords (make-coords :pos (float-vector best-x best-y (+ best-z (/ gr-z 2.0)))
                                     :rpy (float-vector 0 pi/2 (* -1 best-rot)))))
            (send coords :rotate (* pi 0.1) :x :world)
            (send *fetch* :inverse-kinematics
                  coords
                  :use-torso nil))
          (send gripper-inside :locate (float-vector best-x best-y (+ best-z (/ gr-z 2.0))) :world) ;; change position absolutely
          (send gripper-inside :orient best-rot :z :world) ;; change rotation absolutely
          (send stick-l :translate (float-vector 0 0 (- (/ (+ max-z min-z) 2.0) (aref (send stick-l :worldpos) 2))) :world)
          (send stick-r :translate (float-vector 0 0 (- (/ (+ max-z min-z) 2.0) (aref (send stick-r :worldpos) 2))) :world)
          (send stick-center :translate (float-vector 0 0 (- (/ (+ max-z min-z) 2.0) (aref (send stick-center :worldpos) 2))) :world)
          ;; (objects (append (list *fetch* pc stick-l stick-r stick-center) region)) ;; for debug view
          (if visualize-searching
            (progn
              ;; (send *irtviewer* :objects (append (list *fetch* pc) region))
              ;; (send *irtviewer* :redraw)
              (unix:usleep (* 100 1000))
              (send *irtviewer* :objects (append (list *fetch* best-pc-inside-gripper best-pc-outside-gripper) region))
              (send *irtviewer* :viewer :viewing :look viewpoint viewtarget #f(0 0 1)) ;; fix camera position and posture
              (format t "max-inside-point: ~A" max-inside-point)
              ;; (send *irtviewer* :change-background (float-vector (/ max-inside-point 30.0) 0 0)) ;; 30.0 is given by human, tuned parameter.
              (send *irtviewer* :draw-objects :flush t)
              ;; (send *irtviewer* :viewer :viewsurface :string 5 20 (format nil "points:"))
              ;; (send *irtviewer* :viewer :viewsurface :string 5 50 (format nil "~A" max-inside-point))
                (send *irtviewer* :viewer :viewsurface :string 30 20 (format nil "Grids in"))
                (send *irtviewer* :viewer :viewsurface :string 30 50 (format nil "gripper: ~A" max-inside-point))
              (send *irtviewer* :viewer :viewsurface :flush)
              )
            (objects (append (list *fetch* pc) region)))
          max-inside-point)
        (progn
          (format t "cannot find graspable point.~%")
          0))
      )))


;; sample
;; decide-grope-point occupied-pointcloud target-region :x-offset 80 :y-offset 80 :z-offset 60
(defun decide-grope-point (pc target-region &key (x-offset 0) (y-offset 0) (z-offset 0) (visualize-searching nil))
  (let ((hand-pos nil) (x nil) (y nil) (z nil) (fetch-copy nil) (points 0) (max-points 0)
        (min-x (+ (reduce #'min (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))) x-offset))
        (max-x (- (reduce #'max (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))) x-offset))
        (min-y (+ (reduce #'min (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))) y-offset))
        (max-y (- (reduce #'max (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))) y-offset))
        (min-z (+ (reduce #'min (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices))) z-offset))
        (max-z (- (reduce #'max (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices))) z-offset)))
    (setq points (search-graspable-point pc :min-x min-x :max-x max-x :min-y min-y :max-y max-y :min-z min-z :max-z max-z :iter-count 500 :visualize-searching visualize-searching))
    (setq max-points points)
    (setq fetch-copy (copy-object *fetch*))
    (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos))
    (print hand-pos)
    (setq x (aref hand-pos 0))
    (setq y (aref hand-pos 1))
    (setq z (aref hand-pos 2))
    ;; (setq points (search-graspable-point pc :min-x (max min-x (- x 100)) :max-x (min max-x (+ x 100)) :min-y (max min-y (- y 100)) :max-y (min max-y (+ y 100)) :min-z min-z :max-z max-z :iter-count 500 :visualize-searching visualize-searching))
    ;; (when (< max-points points)
    ;;   (setq fetch-copy (copy-object *fetch*))
    ;;   (setq max-points points)
    ;;   (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)))
    ;; (print (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)) ;; hand position
    ;; (setq x (aref hand-pos 0))
    ;; (setq y (aref hand-pos 1))
    ;; (setq z (aref hand-pos 2))
    ;; (setq points (search-graspable-point pc :min-x (max min-x (- x 50)) :max-x (min max-x (+ x 50)) :min-y (max min-y (- y 50)) :max-y (min max-y (+ y 50)) :min-z min-z :max-z max-z :iter-count 500 :visualize-searching visualize-searching))
    ;; (when (< max-points points)
    ;;   (setq fetch-copy (copy-object *fetch*))
    ;;   (setq max-points points)
    ;;   (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)))
    ;; (print (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)) ;; hand position
    ;; (setq *fetch* fetch-copy)
    (objects (list *fetch* pc))
    (list max-points *fetch*)
    ))


;; NOTE arguments of this function means region of the target space, not octomap region.
;; return map of where is not explored very much
;; (unknown-distribution :min-x *min-x* :max-x *max-x* :min-y *min-y* :max-y *max-y* :min-z *min-z* :max-z *max-z* :block-size 50.0)
(defun unknown-distribution (occupied-pointcloud unknown-pointcloud
                             &key (min-x *min-x*) (max-x *max-x*) (min-y *min-y*) (max-y *max-y*) (min-z *min-z*) (max-z *max-z*) (block-size 50.0)
                             (view-octomap-2d nil) (view-unknown-distribution t))
  (unless (and (eq (mod (round (- *max-x* *min-x*)) (round block-size)) 0)
               (eq (mod (round (- *max-y* *min-y*)) (round block-size)) 0))
    (format t "target region is not a multiple of block-size~%")
    (return-from unknown-distribution nil))
  (let* ((occupied-pc occupied-pointcloud)
         (unknown-pc unknown-pointcloud)
         (octomap-2d (apply #'matrix (make-list (round (/ (- max-x min-x) *resolution*)) :initial-element (make-list (round (/ (- max-y min-y) *resolution*)) :initial-element 0))))
         direction-arrow)
    (when (eq occupied-pc nil)
      (format t "cannot find occupied grids~%")
      (setq occupied-pc (instance pointcloud :init)))
    (when (eq unknown-pc nil)
      (format t "cannot find unknown grids~%")
      (setq unknown-pc (instance pointcloud :init)))
    ;; look up occupied pointcloud firstly
    (dotimes (i (send occupied-pc :size))
      (let* ((occupied (matrix-row (send occupied-pc :points) i))
             (x (round (/ (- (aref occupied 0) min-x (/ *resolution* 2.0)) *resolution*)))
             (y (round (/ (- (aref occupied 1) min-y (/ *resolution* 2.0)) *resolution*)))) ;; x and y are index
        ;; (if (and (< min-x (aref occupied 0) max-x) (< min-y (aref occupied 1) max-y) (< min-z (aref occupied 2) max-z))
        (if (and (<= 0 x) (< x (car (array-dimensions octomap-2d))) (<= 0 y) (< y (cadr (array-dimensions octomap-2d))))
          (setf (aref octomap-2d x y) -1))))
    ;; look up unknown pointcloud secondly
    (dotimes (j (send unknown-pc :size))
      (let* ((unknown (matrix-row (send unknown-pc :points) j))
             (x (/ (round (- (aref unknown 0) min-x *resolution*)) *resolution*))
             (y (/ (round (- (aref unknown 1) min-y *resolution*)) *resolution*))) ;; x and y are index
        (if (and (<= 0 x) (< x (car (array-dimensions octomap-2d))) (<= 0 y) (< y (cadr (array-dimensions octomap-2d))) ;; target point is inside the target region?
                 (not (eq (aref octomap-2d x y) -1)))
          (setf (aref octomap-2d x y) (+ 1 (aref octomap-2d x y)))
          )))

    ;; visualization of octomap-2d
    (when view-octomap-2d
      (let ((cube nil) (vis-list nil) (max-num 0))
        (setq *resolution* 10)
        (dotimes (k (car (array-dimensions octomap-2d)))
          (dotimes (l (cadr (array-dimensions octomap-2d)))
            (unless (or (eq k 0) (eq l 0)) ;; this is deprecated, find the root of this bug
              (if (< max-num (aref octomap-2d k l))
                (setq max-num (aref octomap-2d k l))))))
        (dotimes (i (car (array-dimensions octomap-2d)))
          (dotimes (j (cadr (array-dimensions octomap-2d)))
            (unless (or (eq i 0) (eq j 0)) ;; this is deprecated, find the root of this bug
              (setq cube (make-cube *resolution* *resolution* *resolution*))
              (send cube :set-color (float-vector 0 0 (/ (max 0 (aref octomap-2d i j)) max-num)))
              (send cube :translate (float-vector (+ (* *resolution* (+ i 0.5)) *min-x*) (+ (* *resolution* (+ j 0.5)) *min-y*) *min-z*) :world)
              (setq vis-list (append vis-list (list cube))))))
        (setq *resolution* 10)
        (objects (append (list *fetch*) vis-list))
        (send *irtviewer* :change-background #f(0.3 0.3 0.1))
        (send *irtviewer* :redraw)
        (x::window-main-one)))

    ;; calculate unknown-distribution matrix. this shows rough distribution of unknown area, and visualize it.
    (let* ((x-size (round (/ (- max-x min-x) block-size))) ;; x-size of target region, not octomap region. divide octomap area into (x-size * y-size) areas
           (y-size (round (/ (- max-y min-y) block-size))) ;; y-size of target region, not octomap region.
           (dist-mat (apply #'matrix (make-list x-size :initial-element (make-list y-size :initial-element 0))))
           (max-unknown 0) max-unknown-place dist-vis-list cube
           (adjacent-max-unknown 0) adjacent-max-unknown-place
           (region-cube (send (make-cube (- max-x min-x) (- max-y min-y) (- max-z min-z))
                              :locate (float-vector (* 0.5 (+ max-x min-x)) (* 0.5 (+ max-y min-y)) (* 0.5 (+ max-z min-z)))))
           (region-edge (progn (send region-cube :worldcoords) (send region-cube :edges))))
      ;; octomap-2d is composed of very fine grids, so make it rougher and create dist-mat (distribution matrix of unknown grids)
      (dotimes (i (car (array-dimensions octomap-2d)))
        (dotimes (j (cadr (array-dimensions octomap-2d)))
          (let ((x-index (/ i (round (/ block-size *resolution*))))
                (y-index (/ j (round (/ block-size *resolution*)))))
            (setf (aref dist-mat x-index y-index)
                  (+ (aref octomap-2d i j) (aref dist-mat x-index y-index)))
        )))

      (when view-unknown-distribution
        ;; calculate the most unknown place
        (dotimes (k (car (array-dimensions dist-mat)))
          (dotimes (l (cadr (array-dimensions dist-mat)))
            (when (< max-unknown (aref dist-mat k l))
              (setq max-unknown-place (list k l))
              (setq max-unknown (aref dist-mat k l)))))
        ;; calculate the most unknown adjacent place
        (dotimes (i 4)
          (let ((x nil) (y nil))
            (cond ((eq i 0) (setq x 1) (setq y 0))
                  ((eq i 1) (setq x 0) (setq y 1))
                  ((eq i 2) (setq x -1) (setq y 0))
                  ((eq i 3) (setq x 0) (setq y -1)))
            ;; when ((+ (car max-unknown-place) x), (+ (cadr max-unknown-place) y)) is not out of dist-mat
            (when (not (or (< (+ (car max-unknown-place) x) 0) (<= (car (array-dimensions dist-mat)) (+ (car max-unknown-place) x))
                           (< (+ (cadr max-unknown-place) y) 0) (<= (cadr (array-dimensions dist-mat)) (+ (cadr max-unknown-place) y))))
              (when (< adjacent-max-unknown (aref dist-mat (+ (car max-unknown-place) x) (+ (cadr max-unknown-place) y)))
                (setq adjacent-max-unknown (aref dist-mat (+ (car max-unknown-place) x) (+ (cadr max-unknown-place) y)))
                (setq adjacent-max-unknown-place (list x y))))))

        (dotimes (k (car (array-dimensions dist-mat)))
          (dotimes (l (cadr (array-dimensions dist-mat)))
            (setq cube (make-cube (/ (- max-x min-x) x-size) (/ (- max-y min-y) y-size) *resolution*))
            (send cube :set-color (float-vector 0 0 (max (/ (aref dist-mat k l) (float max-unknown)) 0)))
            (send cube :translate (float-vector (+ (* (/ (- max-x min-x) x-size) (+ k 0.5)) min-x) (+ (* (/ (- max-y min-y) y-size) (+ l 0.5)) min-y) *min-z*) :world)
            (setq dist-vis-list (append dist-vis-list (list cube)))))
        ;; (objects (append (list *fetch*) dist-vis-list region-edge))
        (setq direction-arrow
              (make-arrow
               (float-vector (+ (* (/ (- max-x min-x) x-size) (+ (car max-unknown-place) 0.5)) min-x)
                             (+ (* (/ (- max-y min-y) y-size) (+ (cadr max-unknown-place) 0.5)) min-y)
                             (+ *min-z* 20)) ;; pos
               (float-vector (* (/ (- max-x min-x) x-size) (car adjacent-max-unknown-place))
                             (* (/ (- max-y min-y) y-size) (cadr adjacent-max-unknown-place))
                             0) ;; vec
               :color #f(1 0 0)))
        (objects (append (list *fetch*) dist-vis-list direction-arrow))
        ;; (send *irtviewer* :change-background #f(0.3 0.3 0.1))
        (send *irtviewer* :change-background #f(0.2 0.2 0.2))
        ;; (send *irtviewer* :viewer :viewing :look #f(640.0 0.0 1200) #f(640.0 0.0 550.0) #f(0 0 1)) ;; above view
        (send *irtviewer* :viewer :viewing :look #f(2635.65 -2264.36 2194.42) #f(-655.171 698.709 314.742) #f(0 0 1)) ;; side view
        (send *irtviewer* :draw-objects :flush t)
        (send *irtviewer* :redraw)
        ;; draw arrow to move hand
        (x::window-main-one))
      dist-mat)
    ))


(defun box-recog (boxes &key (offset 0))
  (let* ((box (car (send boxes :boxes)))
         (dims (ros::tf-point->pos (send box :dimensions)))
         (target-region nil))
    (setq target-region
          (make-cube (- (elt dims 0) offset) (- (elt dims 1) offset) (- (elt dims 2) offset))) ;; 300 is for microwave
    (if (not (and (boundp '*tfl*) *tfl*))
      (setq *tfl* (instance ros::transform-listener :init)))
    (send target-region :newcoords
          (send (send *tfl* :lookup-transform "/base_link" (send boxes :header :frame_id) (ros::time 0))
                :transform (ros::tf-pose->coords (send box :pose))))
    ;; (send target-region :translate (float-vector (* 0.5 (x-of-cube target-region)) 0 0) :world)
    (send target-region :worldcoords)
    target-region))

(defun collision-object-for-microwave (target-region)
  (let ((microwave-box-down nil) (microwave-box-up nil) (microwave-left nil) (microwave-right nil) (box-list nil)
        (microwave-door nil)
        (xoc (x-of-cube target-region)) (yoc (y-of-cube target-region)) (zoc (z-of-cube target-region))
        )
    ;; microwave box
    (setq microwave-box-down (make-cube xoc yoc (- (aref (send target-region :centroid) 2) (* 0.5 zoc))))
    (setq microwave-box-up (make-cube xoc yoc 700))
    (setq microwave-box-left (make-cube xoc 200 (+ (z-of-cube microwave-box-down) zoc (z-of-cube microwave-box-up))))
    (setq microwave-box-right (make-cube xoc 200 (+ (z-of-cube microwave-box-down) zoc (z-of-cube microwave-box-up))))
    (send microwave-box-down :locate (float-vector (+ (aref (send target-region :centroid) 0) (- (* 0.5 (x-of-cube microwave-box-down)) (* 0.5 xoc)))
                                                   (aref (send target-region :centroid) 1)
                                                   (* 0.5 (z-of-cube microwave-box-down))))
    (send microwave-box-up :locate (float-vector (+ (aref (send target-region :centroid) 0) (- (* 0.5 (x-of-cube microwave-box-up)) (* 0.5 xoc)))
                                                 (aref (send target-region :centroid) 1)
                                                 (+ (aref (send target-region :centroid) 2) (* 0.5 zoc) (* 0.5 (z-of-cube microwave-box-up)))))
    (send microwave-box-left :locate (float-vector (+ (aref (send target-region :centroid) 0) (- (* 0.5 (x-of-cube microwave-box-left)) (* 0.5 xoc)))
                                                   (+ (* 0.5 (+ yoc (y-of-cube microwave-box-left))) (aref (send target-region :centroid) 1))
                                                   (* 0.5 (z-of-cube microwave-box-left))))
    (send microwave-box-right :locate (float-vector (+ (aref (send target-region :centroid) 0) (- (* 0.5 (x-of-cube microwave-box-right)) (* 0.5 xoc)))
                                                    (+ (* -0.5 (+ yoc (y-of-cube microwave-box-right))) (aref (send target-region :centroid) 1))
                                                    (* 0.5 (z-of-cube microwave-box-left))))
    (setq box-list (list microwave-box-down microwave-box-up microwave-box-left microwave-box-right))

    ;; microwave door
    (setq microwave-door (make-cube 300 300 zoc))
    (send microwave-door :locate (v+ (send target-region :centroid)
                                     (scale 0.5 (float-vector (+ (* -1 (x-of-cube microwave-door)))
                                                              (+ (y-of-cube microwave-door) yoc)
                                                              0))))
    ;; (objects (append box-list (list microwave-door *fetch*)))
    (append box-list (list microwave-door))
    ))

(defun prepose-for-main (&key (type :all))
  (let ((av0 nil) (av1 nil) (av2 nil) (av3 nil))
    (setq av0 (send *fetch* :reset-pose))
    (setq av1 (send *fetch* :rarm :move-end-pos #f(100 -100 100) :world :use-torso nil))
    (send *fetch* :reset-pose)
    (send *fetch* :torso :waist-z :joint-angle *torso*)
    (send *fetch* :rarm :shoulder-y :joint-angle -80)
    (setq av3 (send *fetch* :rarm :inverse-kinematics
                    (make-coords :pos (float-vector -100 -400 950) :rpy (float-vector pi/2 (* pi 0.43) 0))
                    :use-torso nil
                    :debug-view nil
                    :min-loop 100
                    :revert-if-fail nil
                    :additional-nspace-list (mapcar #'(lambda (link refangle)
                                                        (eval `(list link #'(lambda () (* 0.01 (- ,refangle (send ,link :joint :joint-angle)))))))
                                                    (send-all (list (nth 5 (send *fetch* :rarm :joint-list))) :child-link) ;; 5th joint equals to (send *fetch* :rarm :wrist-p)
                                                    (list 0))))
    (setq av2 (send *fetch* :inverse-kinematics (make-coords :pos (float-vector 200 -350 950) :rpy (float-vector 0 pi pi)) :use-torso nil))

    (when (eq type :all)
      (while (not (send *ri* :angle-vector-sequence (list av1 av2 av3) (list 2500 2500 5000)))
        (format t "retry prepose-for-main~%")
        (unix:usleep (* 10 1000)))
      (send *fetch* :angle-vector av3)
      ;; (send *ri* :wait-interpolation)
      (unix:sleep 10)
      )
    (when (eq type :part)
      (while (not (send *ri* :angle-vector av3 3000))
        (format t "retry prepose-for-main~%")
        (unix:usleep (* 10 1000)))
      (send *fetch* :angle-vector av3)
      (send *ri* :wait-interpolation))
    (when (eq type :reverse)
      (while (not (send *ri* :angle-vector-sequence (list av3 av2 av1 av0) (list 5000 5000 2500 2500)))
        (format t "retry prepose-for-main~%")
        (unix:usleep (* 10 1000)))
      (send *fetch* :angle-vector av0)
      ;; (send *ri* :wait-interpolation)
      (unix:sleep 15)
      )
    ))

nil
