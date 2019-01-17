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
(setq *torso* 20)
;; direction accumulation
(setq *direction-accum* 0)
;; collision objects for moveit
(unless (boundp '*collision-objects*)
  (setq *collision-objects* nil))

(unless (boundp '*ri*)
  (set-fetch))
(unless (boundp '*irtviewer*)
  (fetch)
  (objects (list *fetch*)))

(warning-message 3 (format nil "~%fetch's pre-pose:~%  (progn (send *fetch* :reset-pose) (send *fetch* :torso :waist-z :joint-angle 290) (send *fetch* :rarm :shoulder-y :joint-angle -80) (send *fetch* :inverse-kinematics
            (make-coords :pos (float-vector 500 0 700))
            :use-torso nil
            :debug-view nil
            :min-loop 50
            :revert-if-fail nil
            :additional-nspace-list (mapcar #'(lambda (link refangle)
                                                (eval `(list link #'(lambda () (* 0.001 (- ,refangle (send ,link :joint :joint-angle)))))))
                                            (send-all (list (nth 1 (send *fetch* :rarm :joint-list))) :child-link)
                                            (list -90))))~%"))
(warning-message 3 (format nil "~%send av to *ri*:~%  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)~%"))

;; NOTE type is :setting, :main or :all
;; :setting does setting of this function, :main does main part of this function and :all does both of them.
;; NOTE do pre-pose before executing this function
(defun main-in-microwave (&key (type :all))
  (if (not (or (equal type :setting) (equal type :main) (equal type :all)))
    (progn (format t "argument :type is not correctly set~%") (return-from main nil)))
  (if (boundp '*co*)
    (send *co* :wipe-all))
  (setq *direction-accum* 0)
  (let ((boxes nil) (target-region nil) x-min x-max y-min y-max z-min z-max)
    (when (or (equal type :all) (equal type :setting))
      ;; settings for experiment
      ;; below is IK program for stretching wrist_flex_joint
      (send *fetch* :reset-pose)
      (send *fetch* :torso :waist-z :joint-angle 290)
      (send *fetch* :rarm :shoulder-y :joint-angle -80)
      (send *fetch* :inverse-kinematics
            (make-coords :pos (float-vector 500 (/ (+ *min-y* *max-y*) 2.0) (/ (+ *min-z* *max-z*) 2.0)))
            :use-torso nil
            :debug-view :no-message
            :min-loop 50
            :revert-if-fail nil
            :additional-nspace-list (mapcar #'(lambda (link refangle)
                                                (eval `(list link #'(lambda () (* 0.001 (- ,refangle (send ,link :joint :joint-angle)))))))
                                            (send-all (list (nth 5 (send *fetch* :rarm :joint-list))) :child-link)
                                            (list -90))
            )
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
      (setq *target-region* (copy-object target-region))
      (camera-stop)

      ;; set hand to initial experiment pose
      (send *fetch* :rarm :move-end-pos #f(0 400 0) :world :use-torso nil)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 4000)
      (send *ri* :wait-interpolation)
      ;; set environment as collision object for moveit
      (collision-object-for-microwave target-region)
      )

    (break)

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
             (unless (send *fetch* :rarm :move-end-pos (float-vector -100 0 0) :world)
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
    (prepose-for-main)

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
      ;; calculate most unknown area
      (dotimes (i (car (array-dimensions unknown-dist)))
        (dotimes (j (cadr (array-dimensions unknown-dist)))
          (when (< max-val (aref unknown-dist i j))
            (setq max-val (aref unknown-dist i j))
            (setq max-place (list i j)))))
      (setf (aref (send target-coords :worldpos) 0) min-x)
      (setf (aref (send target-coords :worldpos) 1) (+ min-y (/ block-size 2.0) (* (car max-place) block-size)))
      (setf (aref (send target-coords :worldpos) 2) (+ min-z (/ block-size 2.0) (* (cadr max-place) block-size)))

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
        ;; calculate direction of trace motion
        (setq direction (+ 90 (rad2deg (atan2 (cadr adjacent-max-place) (car adjacent-max-place)))))
        (print (list direction adjacent-max-place))
        ;; visualize the direction
        (setq direction-arrow
              (make-arrow
               (float-vector (- *min-x* 20)
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
                  (hand-cover-thick 30)
                  ;; when :gripper :position is 0.3, gripper-handside-distance is about 25
                  (gripper-handside-distance (- 35 (* 0.5 1000 (- (send *ri* :gripper :position) 0.01)))))
              ;; check if there is object near top of the grippers.
              (if (setq detected-frame (cadr (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                                                     :close-frame-list (list (list "/left" "/top") (list "/right" "/top")
                                                                             (list "/left" "/side1") (list "/right" "/side1")
                                                                             (list "/left" "/side2") (list "/right" "/side2"))
                                                     :close-thre (list 35 35
                                                                       hand-cover-thick hand-cover-thick
                                                                       (+ 15 gripper-handside-distance) ;; +10 is safety margin
                                                                       (+ 15 gripper-handside-distance))
                                                     :tm 7000)))
                (cond
                 ((equal detected-frame (list "/left" "/top")) ;; object is near left top
                  (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/top")) :data) cons)) 40) ;; both left and right finger top is near object, start moving horizontally
                    (format t "start moving horizontally~%")
                    (return))
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (+ gripper-handside-distance 10) 0))
                                :close-frame-list (list (list "/right" "/side2")) :close-thre gripper-handside-distance :tm 3000)
                    (format t "something is right of my gripper~%"))
                  (return))
                 ((equal detected-frame (list "/right" "/top")) ;; object is near right top
                  (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/top")) :data) cons)) 40) ;; both left and right finger top is near object, start moving horizontally
                    (format t "start moving horizontally~%")
                    (return))
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (* -1 (+ gripper-handside-distance 10)) 0))
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
;; NOTE the dump file below is too big. load the file only when you need
;; (load "package://octomap_server/euslisp/microwave-env.l")
;; NOTE

;; when using microwave-env.l, do sample code below
;; (search-graspable-point occupied-pointcloud :min-x 710 :max-x 810 :min-y -45 :max-y 215 :min-z 1000 :max-z 1150 :visualize-searching t :iter-count 100)
(defun search-graspable-point (pc &key (min-x 750) (max-x 850) (min-y -100) (max-y 100) (min-z 950) (max-z 1100) (iter-count 1000) (visualize-searching nil))
  (let* (stick-l stick-r stick-center
         random-y random-z random-rot
         best-x best-y best-z best-rot
         (gr-x 40.0) (gr-y 30.0) (gr-z 60.0) (gr-open 100.0) (hand-x 70.0)
         (gripper-l (send (make-cube gr-x gr-y gr-z) :locate (float-vector 0 (/ gr-open -2) 0)))
         (gripper-r (send (make-cube gr-x gr-y gr-z) :locate (float-vector 0 (/ gr-open 2) 0)))
         (gripper-inside (make-cube gr-x (- gr-open gr-y) gr-z))
         (point-list (send pc :point-list))
         (point-list-sorted nil) (point-list-inside nil) (point-list-outside nil)
         nearest-x
         (inside-point 0) (max-inside-point 0) (best-pc-inside-gripper nil) (best-pc-outside-gripper nil)
         (viewpoint #f(1528.4 1471.28 1304.32))
         (viewtarget #f(604.334 -129.251 831.657))
         )
    (send *fetch* :torso :waist-z :joint-angle 290)
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
      (setq stick-l (send (make-cube (- max-x min-x) gr-x gr-y) :locate (float-vector (/ (+ max-x min-x) 2.0) (/ gr-open -2) 0)))
      (send stick-l :set-color #f(1 0 0 0.5))
      (setq stick-r (send (make-cube (- max-x min-x) gr-x gr-y) :locate (float-vector (/ (+ max-x min-x) 2.0) (/ gr-open 2) 0)))
      (send stick-r :set-color #f(0 1 0 0.5))
      (setq stick-center (send (make-cube (- max-x min-x) hand-x (- gr-open gr-y)) :locate (float-vector (/ (+ max-x min-x) 2.0) 0 0)))
      (send stick-center :set-color #f(0 0 1 0.5))
      (send gripper-inside :assoc gripper-l)
      (send gripper-inside :assoc gripper-r)
      (send gripper-inside :rotate -pi/2 :y) ;; to change depth direction from z-axis to x-axis
      (send gripper-inside :assoc stick-l)
      (send gripper-inside :assoc stick-r)
      (send gripper-inside :assoc stick-center)

      ;; sort points in the order of z-height
      ;; exclude points which is out of target region
      ;; NOTE if execution of (search-graspable-point) is slow, let's make pointcloud sparse
      ;; NOTE by using (dotimes (point point-list) (when (eq (random 4) 0) ~~~))
      (dolist (point point-list)
        (when t ;; (eq (random 4) 0)
          (setq point-list-sorted (append point-list-sorted (list point)))
          (if (and (<= min-x (aref point 0) max-x) (<= min-y (aref point 1) max-y) (<= min-z (aref point 2) max-z))
            (setq point-list-inside (append point-list-inside (list point)))
            (setq point-list-outside (append point-list-outside (list point))))))
      (setq point-list-sorted (copy-object point-list-sorted))
      ;; (sort) is destructively sort
      (sort point-list-sorted #'(lambda (x y) (< (aref x 0) (aref y 0))))
      ;; search for most graspable point
      (dotimes (i iter-count)
        ;; reset variables
        (setq nearest-x nil)
        (setq inside-point 0)
        ;; move gripper randomly and get the highest point under the gripper
        ;; update objects related counting pointcloud
        (send stick-l :worldcoords)
        (send stick-r :worldcoords)
        (send stick-center :worldcoords)
        (send gripper-inside :worldcoords)

        ;; set random point to grope
        (setq random-y (+ (random (- max-y min-y)) min-y))
        (setq random-z (+ (random (- max-z min-z)) min-z))
        ;; NOTE if considering gravity direction, you can restrict random-rot. e.g. you can choose pi/2 for wrist angle range
        (setq random-rot (funcall #'(lambda (range) (- (random range) (/ range 2.0))) pi/2)) ;; maybe pi/2 does better than pi
        (send gripper-inside :locate (float-vector 0 random-y random-z) :world) ;; change position absolutely
        (send gripper-inside :orient random-rot :z :local) ;; change rotation absolutely

        ;; calc point at which the gripper and the hand collides with pointcloud
        (let ((l/r-flag nil) (center-flag nil))
          (dolist (point point-list-sorted)
            (when (and (not l/r-flag)
                       (or (equal (send stick-l :insidep point) :inside)
                           (equal (send stick-r :insidep point) :inside)))
              (if nearest-x
                (setq nearest-x (min nearest-x (aref point 0)))
                (setq nearest-x (aref point 0)))
              (setq l/r-flag t)
              (if (and l/r-flag center-flag)
                (return)))
            (when (and (not center-flag)
                       (equal (send stick-center :insidep point) :inside))
              (if nearest-x
                (setq nearest-x (min nearest-x (+ (aref point 0) gr-z)))
                (setq nearest-x (+ (aref point 0) gr-z)))
              (setq center-flag t)
              (if (and l/r-flag center-flag)
                (return)))))

        (when nearest-x
          ;; move the hand down until collision with the highest point, and count the points inside the gripper.
          (send gripper-inside :dissoc stick-l)
          (send gripper-inside :dissoc stick-r)
          (send gripper-inside :dissoc stick-center)
          (send gripper-inside :locate (float-vector 0 0 (* -1 (- nearest-x (/ gr-z 2.0)))) :local)
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
                (send *fetch* :angle-vector #f(290.0 -91.1404 64.5379 31.0038 101.51 96.5575 58.5683 102.242 0.0 0.0)) ;; pre-pose for IK
                (send *fetch* :inverse-kinematics
                      (make-coords :pos (float-vector (- nearest-x (/ gr-z 2.0)) random-y random-z)
                                   :rpy (float-vector 0 0 (* -1 random-rot)))
                      :use-torso nil)
                (send *irtviewer* :objects (append (list *fetch* pc) region))

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
                                                                                     :initial-element #f(0.3 0.3 0.5)))))
                  (setq pc-outside-gripper (instance pointcloud :init)))
                ;; (send *irtviewer* :redraw)
                (send *irtviewer* :objects (append (list *fetch* pc-inside-gripper pc-outside-gripper) region))
                (send *irtviewer* :viewer :viewing :look viewpoint viewtarget #f(0 0 1)) ;; fix camera position and posture
                ;; (send *irtviewer* :change-background (float-vector (/ inside-point 30.0) 0 0)) ;; 30.0 is given by human, tuned parameter.
                (send *irtviewer* :draw-objects :flush t)
                (send *irtviewer* :viewer :viewsurface :string 5 20 (format nil "points:"))
                (send *irtviewer* :viewer :viewsurface :string 5 50 (format nil "~A" inside-point))
                (send *irtviewer* :viewer :viewsurface :flush))

              (when (> inside-point max-inside-point)
                (setq max-inside-point inside-point)
                (setq best-x nearest-x)
                (setq best-y random-y)
                (setq best-z random-z)
                (setq best-rot random-rot)
                (when visualize-searching
                  (setq best-pc-inside-gripper (copy-object pc-inside-gripper))
                  (setq best-pc-outside-gripper (copy-object pc-outside-gripper))))))

          ;; redo the movement of gripper-l, gripper-r and gripper-inside
          (send gripper-inside :locate (float-vector 0 0 0) :world)
          (send stick-l :locate (float-vector (/ (+ max-x min-x) 2.0) (/ gr-open -2.0) 0) :world)
          (send stick-r :locate (float-vector (/ (+ max-x min-x) 2.0) (/ gr-open 2.0) 0) :world)
          (send stick-center :locate (float-vector (/ (+ max-x min-x) 2.0) 0 0) :world)
          (send gripper-inside :orient 0 :z :world) ;; change rotation absolutely
          (send stick-l :orient 0 :z :world)
          (send stick-r :orient 0 :z :world)
          (send stick-center :orient 0 :z :world)
          (send gripper-inside :rotate -pi/2 :y)
          (send gripper-inside :assoc stick-l)
          (send gripper-inside :assoc stick-r)
          (send gripper-inside :assoc stick-center)
          ))
      (format t "max-inside-point:~A~%" max-inside-point)

      (if best-x
        (progn
          (send *fetch* :angle-vector #f(290.0 -91.1404 64.5379 31.0038 101.51 96.5575 58.5683 102.242 0.0 0.0)) ;; pre-pose for IK
          (send *fetch* :inverse-kinematics
                (make-coords :pos (float-vector (- best-x (/ gr-z 2.0)) best-y best-z)
                             :rpy (float-vector 0 0 (* -1 best-rot))
                             :use-torso nil))
          (send gripper-inside :locate (float-vector (- best-x (/ gr-z 2.0)) best-y best-z) :world) ;; change position absolutely
          (send gripper-inside :orient best-rot :z :world) ;; change rotation absolutely
          (send stick-l :translate (float-vector (- (/ (+ max-x min-x) 2.0) (aref (send stick-l :worldpos) 0)) 0 0) :world)
          (send stick-r :translate (float-vector (- (/ (+ max-x min-x) 2.0) (aref (send stick-l :worldpos) 0)) 0 0) :world)
          (send stick-center :translate (float-vector (- (/ (+ max-x min-x) 2.0) (aref (send stick-l :worldpos) 0)) 0 0) :world)
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
              (send *irtviewer* :viewer :viewsurface :string 5 20 (format nil "points:"))
              (send *irtviewer* :viewer :viewsurface :string 5 50 (format nil "~A" max-inside-point))
              (send *irtviewer* :viewer :viewsurface :flush)
              )
            (objects (append (list *fetch* pc) region)))
          max-inside-point)
        (progn
          (format t "cannot find graspable point.~%")
          0))
      )))

;; sample for microwave
;; (decide-grope-point occupied-pointcloud target-region :x-offset 100 :y-offset 45 :z-offset 40 :visualize-searching t)
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
    (setq points (search-graspable-point pc :min-x min-x :max-x max-x :min-y (max min-y (- y 100)) :max-y (min max-y (+ y 100)) :min-z (max min-z (- z 100)) :max-z (min max-z (+ z 100)) :iter-count 500 :visualize-searching visualize-searching))
    (when (< max-points points)
      (setq fetch-copy (copy-object *fetch*))
      (setq max-points points)
      (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)))
    (print (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)) ;; hand position
    (setq x (aref hand-pos 0))
    (setq y (aref hand-pos 1))
    (setq z (aref hand-pos 2))
    (setq points (search-graspable-point pc :min-x min-x :max-x max-x :min-y (max min-y (- y 50)) :max-y (min max-y (+ y 50)) :min-z (max min-z (- z 50)) :max-z (min max-z (+ z 50)) :iter-count 500 :visualize-searching visualize-searching))
    (when (< max-points points)
      (setq fetch-copy (copy-object *fetch*))
      (setq max-points points)
      (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)))
    (print (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos)) ;; hand position
    (setq *fetch* fetch-copy)
    (objects (list *fetch* pc))
    (list max-points *fetch*)
    ))


;; NOTE arguments of this function means region of the target space, not octomap region.
;; return map of where is not explored very much
;; example code
;; (progn (unknown-distribution occupied-pointcloud unknown-pointcloud :min-x 570 :max-x 870 :min-y -84 :max-y 156 :min-z 1020 :max-z 1140 :block-size 30.0) (objects occupied-pointcloud))
(defun unknown-distribution (occupied-pointcloud unknown-pointcloud
                             &key (min-x *min-x*) (max-x *max-x*) (min-y *min-y*) (max-y *max-y*) (min-z *min-z*) (max-z *max-z*) (block-size 50.0)
                             (view-octomap-2d nil) (view-unknown-distribution t) (visualize-arrow nil))
  (unless (and (eq (mod (round (- *max-y* *min-y*)) (round block-size)) 0)
               (eq (mod (round (- *max-z* *min-z*)) (round block-size)) 0))
    (format t "target region is not a multiple of block-size~%")
    (return-from unknown-distribution nil))
  (let* ((occupied-pc occupied-pointcloud)
         (unknown-pc unknown-pointcloud)
         (octomap-2d (apply #'matrix (make-list (round (/ (- max-y min-y) *resolution*)) :initial-element (make-list (round (/ (- max-z min-z) *resolution*)) :initial-element 0))))
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
             (y (round (/ (- (aref occupied 1) min-y (/ *resolution* 2.0)) *resolution*)))
             (z (round (/ (- (aref occupied 2) min-z (/ *resolution* 2.0)) *resolution*)))
             ) ;; y and z are index
        ;; (if (and (< min-x (aref occupied 0) max-x) (< min-y (aref occupied 1) max-y) (< min-z (aref occupied 2) max-z))
        (if (and (<= 0 y) (< y (car (array-dimensions octomap-2d))) (<= 0 z) (< z (cadr (array-dimensions octomap-2d))))
          (setf (aref octomap-2d y z) -1))))
    ;; look up unknown pointcloud secondly
    (dotimes (j (send unknown-pc :size))
      (let* ((unknown (matrix-row (send unknown-pc :points) j))
             (y (/ (round (- (aref unknown 1) min-y *resolution*)) *resolution*))
             (z (/ (round (- (aref unknown 2) min-z *resolution*)) *resolution*))) ;; y and z are index
        (if (and (<= 0 y) (< y (car (array-dimensions octomap-2d))) (<= 0 z) (< z (cadr (array-dimensions octomap-2d))) ;; target point is inside the target region?
                 (not (eq (aref octomap-2d y z) -1)))
          (setf (aref octomap-2d y z) (+ 1 (aref octomap-2d y z)))
          )))

    ;; visualization of octomap-2d
    (when view-octomap-2d
      (let ((cube nil) (vis-list nil) (max-num 0))
        (setq *resolution* 10)
        (dotimes (k (car (array-dimensions octomap-2d)))
          (dotimes (l (cadr (array-dimensions octomap-2d)))
            (if (< max-num (aref octomap-2d k l))
              (setq max-num (aref octomap-2d k l)))))
        (dotimes (i (car (array-dimensions octomap-2d)))
          (dotimes (j (cadr (array-dimensions octomap-2d)))
            (setq cube (make-cube *resolution* *resolution* *resolution*))
            (send cube :set-color (float-vector (/ (max 0 (aref octomap-2d i j)) max-num) 0 0))
            (send cube :translate (float-vector *min-x* (+ (* *resolution* (+ i 0.5)) *min-y*) (+ (* *resolution* (+ j 0.5)) *min-z*)) :world)
            (setq vis-list (append vis-list (list cube)))))
        (setq *resolution* 10)
        (objects (append (list *fetch*) vis-list))
        (send *irtviewer* :change-background #f(0.3 0.3 0.1))
        (send *irtviewer* :redraw)
        (x::window-main-one)))

    ;; calculate unknown-distribution matrix. this shows rough distribution of unknown area, and visualize it.
    (let* ((y-size (round (/ (- max-y min-y) block-size))) ;; y-size of target region, not octomap region. divide octomap area into (y-size * z-size) areas
           (z-size (round (/ (- max-z min-z) block-size))) ;; z-size of target region, not octomap region.
           (dist-mat (apply #'matrix (make-list y-size :initial-element (make-list z-size :initial-element 0))))
           (max-unknown 0) max-unknown-place dist-vis-list cube
           (adjacent-max-unknown 0) adjacent-max-unknown-place
           (region-cube (send (make-cube (- max-x min-x) (- max-y min-y) (- max-z min-z))
                              :locate (float-vector (* 0.5 (+ max-x min-x)) (* 0.5 (+ max-y min-y)) (* 0.5 (+ max-z min-z)))))
           (region-edge (progn (send region-cube :worldcoords) (send region-cube :edges))))
      ;; octomap-2d is composed of very fine grids, so make it rougher and create dist-mat (distribution matrix of unknown grids)
      (dotimes (i (car (array-dimensions octomap-2d)))
        (dotimes (j (cadr (array-dimensions octomap-2d)))
          (let ((y-index (/ i (round (/ block-size *resolution*))))
                (z-index (/ j (round (/ block-size *resolution*)))))
            (setf (aref dist-mat y-index z-index)
                  (+ (aref octomap-2d i j) (aref dist-mat y-index z-index)))
        )))

      (when view-unknown-distribution
        ;; calculate the most unknown place
        (dotimes (k (car (array-dimensions dist-mat)))
          (dotimes (l (cadr (array-dimensions dist-mat)))
            (when (< max-unknown (aref dist-mat k l))
              (setq max-unknown-place (list k l))
              (setq max-unknown (aref dist-mat k l)))))
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ;; Start, replace: x->y, y->z ;;
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ;; End, replace: x->y, y->z ;;
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        (dotimes (k (car (array-dimensions dist-mat)))
          (dotimes (l (cadr (array-dimensions dist-mat)))
            (setq cube (make-cube *resolution* (/ (- max-y min-y) y-size) (/ (- max-z min-z) z-size)))
            (send cube :set-color (float-vector 0 0 (max (/ (aref dist-mat k l) (float max-unknown)) 0)))
            (send cube :translate (float-vector *min-x* (+ (* (/ (- max-y min-y) y-size) (+ k 0.5)) min-y) (+ (* (/ (- max-z min-z) z-size) (+ l 0.5)) min-z)) :world)
            (setq dist-vis-list (append dist-vis-list (list cube)))))
        ;; (objects (append (list *fetch*) dist-vis-list region-edge))
        (when visualize-arrow
          (setq direction-arrow
                (make-arrow
                 (float-vector (- *min-x* 20)
                               (+ (* (/ (- max-y min-y) y-size) (+ (car max-unknown-place) 0.5)) min-y)
                               (+ (* (/ (- max-z min-z) z-size) (+ (cadr max-unknown-place) 0.5)) min-z)) ;; pos
                 (float-vector 0
                               (* (/ (- max-y min-y) y-size) (car adjacent-max-unknown-place))
                               (* (/ (- max-z min-z) z-size) (cadr adjacent-max-unknown-place))) ;; vec
                 :color #f(1 0 0)))
          (objects (append (list *fetch*) dist-vis-list direction-arrow)))
        (objects (append (list *fetch*) dist-vis-list))
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
          (make-cube (- 300 offset) (- (elt dims 1) offset) (- (elt dims 2) offset))) ;; 300 is for microwave
    (if (not (and (boundp '*tfl*) *tfl*))
      (setq *tfl* (instance ros::transform-listener :init)))
    (send target-region :newcoords
          (send (send *tfl* :lookup-transform "/base_link" (send boxes :header :frame_id) (ros::time 0))
                :transform (ros::tf-pose->coords (send box :pose))))
    (send target-region :translate (float-vector (- (+ (* 1000 (send (send (send (car (send boxes :boxes)) :pose) :position) :x)) (* 0.5 (- 300 offset))) (aref (send target-region :worldpos) 0)) 0 0) :world)
    (setq *target-region* (copy-object target-region))
    target-region))

(defun collision-object-for-microwave (target-region)
  (let ((microwave-box-down nil) (microwave-box-up nil) (microwave-left nil) (microwave-right nil) (box-list nil)
        (microwave-door nil)
        (xoc (x-of-cube target-region)) (yoc (y-of-cube target-region)) (zoc (z-of-cube target-region))
        )
    (when (not (boundp '*co*))
      (setq *co* (instance collision-object-publisher :init))
      (send *co* :wipe-all))
    ;; microwave box
    (setq microwave-box-down (make-cube xoc yoc (- (aref (send target-region :centroid) 2) (* 0.5 zoc))))
    (setq microwave-box-up (make-cube xoc yoc 700))
    (setq microwave-box-left (make-cube xoc 200 (+ (z-of-cube microwave-box-down) zoc (z-of-cube microwave-box-up))))
    (setq microwave-box-right (make-cube xoc 200 (+ (z-of-cube microwave-box-down) zoc (z-of-cube microwave-box-up))))
    (send microwave-box-down :locate (float-vector (+ (aref (send target-region :centroid) 0) (* 0.5 (x-of-cube microwave-box-down))) 0 (* 0.5 (z-of-cube microwave-box-down))))
    (send microwave-box-up :locate (float-vector (+ (aref (send target-region :centroid) 0) (* 0.5 (x-of-cube microwave-box-up))) 0 (+ (aref (send target-region :centroid) 2) (* 0.5 zoc) (* 0.5 (z-of-cube microwave-box-up)))))
    (send microwave-box-left :locate (float-vector (+ (aref (send target-region :centroid) 0) (* 0.5 (x-of-cube microwave-box-left))) (* 0.5 (+ yoc (y-of-cube microwave-box-left))) (* 0.5 (z-of-cube microwave-box-left))))
    (send microwave-box-right :locate (float-vector (+ (aref (send target-region :centroid) 0) (* 0.5 (x-of-cube microwave-box-right))) (* -0.5 (+ yoc (y-of-cube microwave-box-right))) (* 0.5 (z-of-cube microwave-box-left))))
    (setq box-list (list microwave-box-down microwave-box-up microwave-box-left microwave-box-right))
    (let ((i 0) (name-list (list "box-down" "box-up" "box-left" "box-right")))
      (dolist (microwave-box box-list)
        (send *co* :add-object microwave-box :frame-id "base_link"
              :relative-pose (send microwave-box :copy-worldcoords)
              :object-id (nth i name-list))
        (setq i (+ i 1))))

    ;; microwave door
    (setq microwave-door (make-cube 300 300 zoc))
    (send microwave-door :locate (v+ (send target-region :centroid)
                                     (scale 0.5 (float-vector (+ (* -1 (x-of-cube microwave-door)))
                                                              (+ (y-of-cube microwave-door) yoc)
                                                              0))))
    (send *co* :add-object microwave-door :frame-id "base_link"
          :relative-pose (send microwave-door :copy-worldcoords)
          :object-id "microwave-door")
    ;; (objects (append box-list (list microwave-door *fetch*)))
    (setq *collision-objects* (append box-list (list microwave-door)))
    ))

(defun prepose-for-main ()
  (send *co* :wipe-all)
  (if *collision-objects*
    (dolist (object *collision-objects*)
      (send *co* :add-object object)))
  (send *fetch* :reset-pose)
  (send *fetch* :torso :waist-z :joint-angle 290)
  (send *fetch* :rarm :shoulder-y :joint-angle -80)
  (send *fetch* :inverse-kinematics
        (make-coords :pos (float-vector 500 (/ (+ *min-y* *max-y*) 2.0) (/ (+ *min-z* *max-z*) 2.0)))
        :use-torso nil
        :debug-view :no-message
        :min-loop 50
        :revert-if-fail nil
        :additional-nspace-list (mapcar #'(lambda (link refangle)
                                            (eval `(list link #'(lambda () (* 0.001 (- ,refangle (send ,link :joint :joint-angle)))))))
                                        (send-all (list (nth 5 (send *fetch* :rarm :joint-list))) :child-link)
                                        (list -90)))
  (send *ri* :angle-vector (send *fetch* :angle-vector) 6000)
  (send *co* :wipe-all)
  )

nil
