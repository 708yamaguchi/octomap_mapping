#!/usr/bin/env roseus

(load "package://bag_segmentation/euslisp/bag-manipulation-interface.l")
(load "package://bag_segmentation/euslisp/utils.l")


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
(setq *target-region* nil)

;; torso length
(setq *torso* 20)

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


;; type is :setting, :main or :all
;; :setting does setting of this function, :main does main part of this function and :all does both of them.
(defun main (&key (type :all))
  (if (not (or (equal type :setting) (equal type :main) (equal type :all)))
    (progn (format t "argument :type is not correctly set~%") (return-from main nil)))
  (let ((boxes nil) (target-region nil) x-min x-max y-min y-max z-min z-max)
    (when (or (equal type :all) (equal type :setting))
      ;; settings for experiment
      (send *fetch* :reset-pose)
      (send *fetch* :torso :waist-z :joint-angle 400)
      (send *fetch* :inverse-kinematics
            (make-coords :pos (float-vector (/ (+ *min-x* *max-x*) 2.0) (/ (+ *min-y* *max-y*) 2.0) 950) :rpy (float-vector 0 pi/2 0))) ;; 950 is tuned parameter
      (send *fetch* :head :look-at (float-vector (/ (+ *max-x* *min-x*) 2.0) (/ (+ *max-y* *min-y*) 2.0) (/ (+ *max-z* *min-z*) 2.0)))
      (send *fetch* :rarm :move-end-pos #f(0 -200 0) :world :use-torso nil) ;; move hand sideway to avoid occlusion
      (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
      (send *ri* :wait-interpolation)
      (send *ri* :start-grasp :wait t)
      (unix:usleep (* 500 1000))
      (send *ri* :set-init-proximities)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 4000)
      (camera-stop)
      (octomap-reset)
      (send *ri* :wait-interpolation)
      (camera-request)
      (unix:sleep 1)

      ;; use camera and decide target region
      (while t ;; wait for /HSI_color_filter/boxes comes once
        (setq boxes (one-shot-subscribe "/HSI_color_filter/boxes" jsk_recognition_msgs::BoundingBoxArray))
        (when (send boxes :boxes)
          (return))
        (unix:usleep (* 100 1000)))
      (setq target-region (box-recog boxes))

      (camera-stop)
      ;; set hand to initial experiment pose
      (send *fetch* :rarm :move-end-pos #f(0 200 0) :world :use-torso nil)
      (send *ri* :angle-vector (send *fetch* :angle-vector) 4000)
      (send *ri* :wait-interpolation))

    (if (equal type :setting)
      (return-from main t))

    ;; grasp object with octomap and grope. TODO simultaneous execution of (create-map) and (decide-grope-point) may be needed.
    ;; flowchart of this (do-until-key) is loop of following steps:
    ;; 1. first create octomap, and then try to grasp the target object with the same octomap up to 5 times.
    ;; 2. if success, lift the object and finish. else, retry from the first step.
    (do-until-key ;; do not return from this loop until the robot succeeds to grasp the target object
     (let ((detectp nil) (graspp nil))
       ;; create octomap by tracing environment, using proximity sensors on finger.
       (dotimes (i 30)
         (create-map target-region))

       ;; calculate the most graspable point. In the function below, *fetch* is set.
       (dotimes (i 5) ;; try to grasp the target object with the same octomap, up to 5 times.
         (decide-grope-point (get-octo-pointcloud :type "occupied") target-region)

         ;; extend hand to target object with pre-grasp motion by proximity sensors
         ;; first, move hand up
         (let ((ec-z (aref (send (send *fetch* :rarm :end-coords :copy-worldcoords) :worldpos) 2)))
           (send *fetch* :rarm :move-end-pos (float-vector 0 0 (- *max-z* ec-z)) :world)
           (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
           (send *ri* :wait-interpolation))
         ;; second, extend hand to target object with pre-grasp motion
         (while t
           (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                              *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
             (if (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                         :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :close-thre 30)
               (progn ;; if something is in front of the gripper, return from this while loop and try to pre-grasp
                 (setq detectp t)
                 (return)))
             (progn ;; if nothing is found, retry (decide-grope-point)
               (format t "cannot find object~%")
               (setq detectp nil)
               (return))))
         ;; TODO PLEASE WRITE (pre-grasp) function
         (if detectp ;; if something is in front of gripper, do pre-grasp
           (setq graspp (pre-grasp))
           (if graspp
             (return))))
       (if graspp ;; if pre-grasp is succeeded and the robot can grasp the target object, return from do-until-key loop
         (return))
       ))

    ;; finally, lift the target object up
    (send *fetch* :rarm :move-end-pos #f(0 0 300) :world)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 7000)
    (format "grope succeeded~%")
    ))


(defun box-recog (boxes)
  (let* ((box (car (send boxes :boxes)))
         (dims (ros::tf-point->pos (send box :dimensions)))
         (target-region nil))
    (setq target-region (make-cube (elt dims 0) (elt dims 1) (- *max-z* *min-z*)))
    (if (not (and (boundp '*tfl*) *tfl*))
      (setq *tfl* (instance ros::transform-listener :init)))
    (send target-region :newcoords
          (send (send *tfl* :lookup-transform "/base_link" (send boxes :header :frame_id) (ros::time 0))
                :transform (ros::tf-pose->coords (send box :pose))))
    (send target-region :translate (float-vector 0 0 (- (* 0.5 (+ *max-z* *min-z*)) (aref (send target-region :worldpos) 2))) :world)
    (setq *target-region* (copy-object target-region))
    target-region))


;; (progn (send *fetch* :reset-pose) (send *fetch* :inverse-kinematics (make-coords :pos (float-vector (/ (+ *min-x* *max-x*) 2.0) (/ (+ *min-y* *max-y*) 2.0) *max-z*) :rpy (float-vector 0 pi/2 0)))) :rotation-axis nil
;; (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
;; TODO: what method is best solution to create map?
(defun create-map (target-region)
  (unless target-region
    (format t "target-region is not set~%")
    (break))
  (let ((target-coords (make-coords :rpy (float-vector 0 pi/2 0))) ;; where to move fetch's hand
        (direction (rad2deg (random pi))) (count 0) (unknown-dist nil))
    (send *ri* :start-grasp :wait t)
    ;; first, moving hand up
    (format t "move hand up. preparing...~%")
    (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
    (send *fetch* :torso :waist-z :joint-angle 350)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
    ;; calculate where to grope, during moving hand up
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
           (block-size 50.0)
           (diff-x (/ (- (* (+ 1 (floor (/ (- max-x-round min-x-round) block-size))) block-size) (- max-x-round min-x-round)) 2.0))
           (diff-y (/ (- (* (+ 1 (floor (/ (- max-y-round min-y-round) block-size))) block-size) (- max-y-round min-y-round)) 2.0))
           )
      ;; calculate position of hand, x and y
      ;; correct min/max-x/y/z in order to adapt to (unknown-distribution)
      (setq min-x-round (- min-x-round diff-x))
      (setq max-x-round (+ max-x-round diff-x))
      (setq min-y-round (+ min-y-round diff-y))
      (setq max-y-round (+ max-y-round diff-y))
      (setq unknown-dist (unknown-distribution :min-x min-x :max-x max-x :min-y min-y :max-y max-y :min-z min-z :max-z max-z
                                               :block-size block-size))
      (dotimes (i (car (array-dimensions unknown-dist)))
        (dotimes (j (cadr (array-dimensions unknown-dist)))
          (when (< max-val (aref unknown-dist i j))
            (setq max-val (aref unknown-dist i j))
            (setq max-place (list i j)))))
      (setf (aref (send target-coords :worldpos) 0) (+ min-x (/ block-size 2.0) (* (car max-place) block-size)))
      (setf (aref (send target-coords :worldpos) 1) (+ min-y (/ block-size 2.0) (* (cadr max-place) block-size)))
      (setf (aref (send target-coords :worldpos) 2) max-z)
      ;; calculate angle of wrist, theta
      (dolist (x (list -1 0 1))
        (dolist (y (list -1 0 1))
          (when (not (or (and (eq x 0) (eq y 0)) ;; when ((+ (car max-place) x), (+ (cadr max-place) y)) is not out of unknown-dist
                         (< (+ (car max-place) x) 0) (<= (car (array-dimensions unknown-dist)) (+ (car max-place) x))
                         (< (+ (cadr max-place) y) 0) (<= (cadr (array-dimensions unknown-dist)) (+ (cadr max-place) y))))
            (when (< adjacent-max-val (aref unknown-dist (+ (car max-place) x) (+ (cadr max-place) y)))
              (setq adjacent-max-val (aref unknown-dist (+ (car max-place) x) (+ (cadr max-place) y)))
              (setq adjacent-max-place (list x y))))))
      (setq direction (+ 180 (* -1 (rad2deg (atan2 (cadr adjacent-max-place) (car adjacent-max-place))))))
      )
    (send *ri* :wait-interpolation)
    ;; calibration proximity sensor's param every time before groping.
    (sensor-calib)
    ;; move hand above the target object
    (send target-coords :translate (float-vector 0 0 (- 950 (aref (send target-coords :pos) 2))) :world)
    (send *fetch* :inverse-kinematics target-coords :use-torso nil)
    (send *fetch* :rarm :wrist-r :joint-angle (+ (send *fetch* :rarm :wrist-r :joint-angle) direction)) ;; rotate arm
    (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    ;; move hand down to target coords ;; these functions may not be needed any more
    ;; (format t "move hand to target coords~%")
    ;; (send *fetch* :torso :waist-z :joint-angle 50)
    ;; (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    ;; (send *ri* :wait-interpolation)

    (format t "start tracing~%")
    ;; move hand until detecting object. move-end-pos 10cm, 10cm, 10cm ...  gradually.
    (while (< count 3) ;; vertical hand move and horizontal hand move, alternately
      (format t "count: ~A~%" count)
      ;; move hand down
      (while t
        (format t "move hand down~%")
        (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                           *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
          (progn
            (format t "move hand until detecting object~%")
            ;; check if there is object near top of the grippers
            (let ((detected-frame nil) (min-distance nil))
              (if (setq detected-frame (cadr (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                                                     :close-frame-list (list (list "/left" "/top") (list "/right" "/top")
                                                                             (list "/left" "/side1") (list "/right" "/side1")
                                                                             (list "/left" "/side2") (list "/right" "/side2"))
                                                     :close-thre (list 30 30 60 60 35 35))))
                (cond
                 ((or (equal detected-frame (list "/left" "/top")) (equal detected-frame (list "/right" "/top"))) ;; object is near top
                  (return))
                 ((or (equal detected-frame (list "/left" "/side1")) (equal detected-frame (list "/right" "/side1"))) ;; object is near side1
                  (ros::spin-once)
                  (setq min-distance (min (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side1")) :data) cons))
                                          (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side1")) :data) cons))))
                  (format t "detect object at side1, so move hand~%")
                  ;; distance between gripper's side1 and back of the hand is about 60mm
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 0 (* -1 (- 60 min-distance))))
                                :close-frame-list (list (list "/left" "/side3") (list "/right" "/side3")) :close-thre 30)
                    (format t "cannot move hand forward any more~%")
                    (return-from create-map t)))
                 ((equal detected-frame (list "/left" "/side2"))
                  (ros::spin-once)
                  (setq min-distance (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side2")) :data) cons)))
                  (format t "detect object at left side2, so move hand~%")
                  ;; distance between left gripper's side2 and side of the hand is about 35mm (at start-grasp state)
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (- 35 min-distance) 0))
                                :close-frame-list (list (list "/left" "/side2")) :close-thre 35)
                    (format t "cannot move hand rightside any more~%")
                    (return-from create-map t)))
                 ((equal detected-frame (list "/right" "/side2"))
                  (ros::spin-once)
                  (setq min-distance (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side2")) :data) cons)))
                  (format t "detect object at right side2, so move hand~%")
                  ;; distance between left gripper's side2 and side of the hand is about 35mm (at start-grasp state)
                  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (* -1 (- 35 min-distance)) 0))
                                :close-frame-list (list (list "/right" "/side2")) :close-thre 35)
                    (format t "cannot move hand leftside any more~%")
                    (return-from create-map t)))
                 ))))
          (progn
            (format t "robot hand is outside of target region~%")
            (return-from create-map t))))
      ;; move hand horizontally
      (while t
        (format t "move hand sideway~%")
        (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                           *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
          (progn
            (format t "move hand until missing object~%")
            (let ((result (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(0 0 -100))
                                    :close-frame-list (list (list "/left" "/side3") (list "/right" "/side3"))
                                    :away-frame-list (list (list "/left" "/top") (list "/right" "/top"))
                                    :close-thre 50 :away-thre 50)))
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


;; ;; WIP.
;; (defun trace (target-coords-world) ;; argument(target-coords) is world coords, rotation of coords is NOT considered.
;;   (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
;;   (let* ((ec-world (send *fetch* :rarm :end-coords :copy-worldcoords))
;;          (target-coords-local (send (send ec-world :transformation (make-coords)) :transform (copy-object target-coords-world)))
;;          (move-pos (send target-coords-local :pos)) ;; move-pos is based on local coords
;;          (close-target-list nil) (away-target-list nil)) ;; sensor frame which is considered detection during tracing
;;     (print target-coords-world)
;;     (print ec-world)
;;     (print move-pos)
;;     (if (> (aref move-pos 0) 20) ;; if move direction is x-axis positive
;;       (setq close-target-list (append close-target-list (list (list "/left" "/top")) (list (list "/right" "/top")))))
;;     (if (> (aref move-pos 1) 20) ;; if move direction is y-axis positive
;;       (setq close-target-list (append close-target-list (list (list "/right" "/side2")))))
;;     (if (< (aref move-pos 1) -20) ;; if move direction is y-axis negative
;;       (setq close-target-list (append close-target-list (list (list "/left" "/side2")))))
;;     (if (> (aref move-pos 2) 20) ;; if move direction is z-axis positive
;;       (setq close-target-list (append close-target-list (list (list "/left" "/side1")) (list (list "/right" "/side1")))))
;;     (if (< (aref move-pos 2) -20) ;; if move direction is z-axis negative
;;       (setq close-target-list (append close-target-list (list (list "/left" "/side3")) (list (list "/right" "/side3")))))

;;     ;; trace flow:
;;     ;; 1. move toward the destination, 2. when detecting at side of gripper, move hand up, 3. when there is nothing under gripper, move hand down
;;     ;; repeat 1~3 above until hand reach the destinationv
;;     (print close-target-list)
;;     (setq target-coords-local target-coords-world)
;;     (setq away-target-list nil)

;;     ;; TODO : decide target-coords-local & close/away-target-list automatically
;;     (dotimes (i 2)
;;       (detect target-coords-local :close-frame-list close-target-list :away-frame-list )
;;       ;; (setq close-target-list hogehoge)
;;       (setq away-target-list close-frame-list)
;;       ;; calculate goal coords view from hand coords
;;       (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
;;       (setq ec-world (send *fetch* :rarm :end-coords :copy-worldcoords))
;;       (setq target-coords-local (send (send ec-world :transformation (make-coords)) :transform (copy-object target-coords-world)))
;;       )
;;     ;; TODO  end
;;     ;; test code
;;     ;; (progn (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
;;     ;;        (send *fetch* :rarm :move-end-pos #f(100 0 0) :world :use-torso nil)
;;     ;;        (trace (send *fetch* :rarm :end-coords :copy-worldcoords)))
;;     ))

;; before calling this function, necessary to subscribe "/unknown(frontier)_cells_vis_array"
;; (load "package://octomap_server/euslisp/occupied-and-unknown-cloud1.l")
;; (load "package://octomap_server/euslisp/occupied-and-unknown-cloud2.l")
;; only occupied and unknown pointcloud can be gotten from this function.
(defun get-octo-pointcloud (&key (type nil))
  (let ((octo-cloud nil) (eus-octo-cloud nil) (color nil) (colors nil))
    ;; subscribe octomap pointcloud
    (cond
     ;; ((equal type "free")
     ;;  (setq color (list 0 1 0))
     ;;  (setq octo-cloud (one-shot-subscribe "/octomap_free_point_cloud_centers" sensor_msgs::Pointcloud2)))
     ((equal type "occupied")
      (setq color (list 0 0 1))
      (setq octo-cloud (one-shot-subscribe "/octomap_point_cloud_centers" sensor_msgs::Pointcloud2)))
     ((equal type "unknown")
      (setq color (list 1 0 1))
      (setq octo-cloud (one-shot-subscribe "/octomap_unknown_point_cloud_centers" sensor_msgs::Pointcloud2)))
     ((equal type "frontier")
      (setq color (list 1 0 0))
      (setq octo-cloud (one-shot-subscribe "/octomap_frontier_point_cloud_centers" sensor_msgs::Pointcloud2)))
     ((equal type "potential")
      (setq color (list 1 0 0))
      (setq octo-cloud (one-shot-subscribe "/octomap_potential_point_cloud_centers" sensor_msgs::Pointcloud2)))
     (t
      (fotmat t "key argument type is not correctly set.~%"))
     )
    (setq eus-octo-cloud (make-eus-pointcloud-from-ros-msg octo-cloud))
    (if eus-octo-cloud
      (progn
        ;; set colors to octo-clouds, free -> green, occupied -> blue, unknown -> purple, frontier -> red
        (setq colors (make-list (send eus-octo-cloud :size) :initial-element (list 0 0 1)))
        (send eus-octo-cloud :colors (apply #'matrix colors))
        eus-octo-cloud))))


;; for debug, call (search-graspable-point *pc1)
(load "package://octomap_server/euslisp/av-and-groped-pc1.l")
(load "package://octomap_server/euslisp/av-and-groped-pc2.l")
(setq *pc1* (cadr av-and-groped-pc1))
(setq *pc2* (cadr av-and-groped-pc1))
;; TODO decide grope point based on height map. hint is in (calc-grope-point)
;; TODO use pyramid algorithm when searching grope point
(defun search-graspable-point (pc &key (min-x 300) (max-x 800) (min-y -300) (max-y 300) (min-z 400) (max-z 1000) (iter-count 1000))
  (let* (stick-l stick-r
         random-x random-y random-rot
         best-x best-y best-z best-rot
         (gr-x 40.0) (gr-y 30.0) (gr-z 60.0) (gr-open 100.0)
         (gripper-l (send (make-cube gr-x gr-y gr-z) :locate (float-vector 0 (/ gr-open -2) 0)))
         (gripper-r (send (make-cube gr-x gr-y gr-z) :locate (float-vector 0 (/ gr-open 2) 0)))
         (gripper-inside (make-cube gr-x (- gr-open gr-y) gr-z))
         (point-list (send pc :point-list))
         (point-list-sorted (copy-object point-list))
         highest-z
         (inside-point 0) (max-inside-point 0)
         )
    (send *fetch* :torso :waist-z :joint-angle 300)
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
      (send gripper-inside :assoc gripper-l)
      (send gripper-inside :assoc gripper-r)
      (send gripper-inside :assoc stick-l)
      (send gripper-inside :assoc stick-r)

      ;; sort points in the order of z-height
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
        (send gripper-inside :worldcoords)
        ;; set random point to grope
        (setq random-x (+ (random (- max-x min-x)) min-x))
        (setq random-y (+ (random (- max-y min-y)) min-y))
        (setq random-rot (- (random pi) pi/2))
        (send gripper-inside :locate (float-vector random-x random-y 0) :world) ;; change position absolutely
        (send gripper-inside :orient random-rot :z :world) ;; change rotation absolutely
        ;; calc point at which the gripper collides with pointcloud
        (dolist (point point-list-sorted)
          (when (or (equal (send stick-l :insidep point) :inside)
                    (equal (send stick-r :insidep point) :inside))
            (setq highest-z (aref point 2))
            (return)))

        (when highest-z
          ;; move the hand down until collision with the highest point, and count the points inside the gripper.
          (send gripper-inside :dissoc stick-l)
          (send gripper-inside :dissoc stick-r)
          (send gripper-inside :locate (float-vector 0 0 (+ highest-z (/ gr-z 2.0))) :local)
          ;; for visualization
          ;; (send *fetch* :inverse-kinematics
          ;;       (make-coords :pos (float-vector random-x random-y (+ highest-z (/ gr-z 2.0)))
          ;;                    :rpy (float-vector 0 pi/2 (* -1 random-rot))
          ;;                    :use-torso nil))
          ;; (objects (append (list *fetch* pc stick-l stick-r gripper-inside) region))
          ;; update objects related counting pointcloud
          (send gripper-inside :worldcoords)
          ;; count the point inside of gripper-inside
          (dolist (point point-list)
            (if (equal (send gripper-inside :insidep point) :inside)
              (setq inside-point (+ inside-point 1))))
          (when (> inside-point max-inside-point)
            (setq max-inside-point inside-point)
            (setq best-x random-x)
            (setq best-y random-y)
            (setq best-z highest-z)
            (setq best-rot random-rot))
          ;; redo the movement of gripper-l, gripper-r and gripper-inside
          (send gripper-inside :locate (float-vector 0 0 0) :world)
          (send stick-l :locate (float-vector 0 (/ gr-open -2) (/ (+ max-z min-z) 2.0)) :world)
          (send stick-r :locate (float-vector 0 (/ gr-open 2) (/ (+ max-z min-z) 2.0)) :world)
          (send gripper-inside :assoc stick-l)
          (send gripper-inside :assoc stick-r)
          ))
      (format t "max-inside-point:~A~%" max-inside-point)
      (if best-z
        (progn
          (send *fetch* :inverse-kinematics
                (make-coords :pos (float-vector best-x best-y (+ best-z (/ gr-z 2.0)))
                             :rpy (float-vector 0 pi/2 (* -1 best-rot))
                             :use-torso nil))
          (send gripper-inside :dissoc stick-l)
          (send gripper-inside :dissoc stick-r)
          (send gripper-inside :locate (float-vector best-x best-y (+ best-z (/ gr-z 2.0))) :world) ;; change position absolutely
          (send gripper-inside :orient best-rot :z :world) ;; change rotation absolutely
          (objects (append (list *fetch* pc) region))
          max-inside-point)
        (progn
          (format t "cannot find graspable point.~%")
          0))
      )))


;; sample
;; (decide-grope-point *pc2*)
(defun decide-grope-point (pc target-region)
  (let ((hand-pos nil) (x nil) (y nil) (z nil) (fetch-copy nil) (points 0) (max-points 0)
        (min-x (reduce #'min (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))))
        (max-x (reduce #'max (mapcar #'(lambda (x) (aref x 0)) (send target-region :vertices))))
        (min-y (reduce #'min (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))))
        (max-y (reduce #'max (mapcar #'(lambda (x) (aref x 1)) (send target-region :vertices))))
        (min-z (reduce #'min (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices))))
        (max-z (reduce #'max (mapcar #'(lambda (x) (aref x 2)) (send target-region :vertices)))))
    (setq max-points (search-graspable-point pc :min-x min-x :max-x max-x :min-y min-y :max-y max-y :min-z min-z :max-z max-z :iter-count 500))
    (setq fetch-copy (copy-object *fetch*))
    (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos))
    (print hand-pos)
    (setq x (aref hand-pos 0))
    (setq y (aref hand-pos 1))
    (setq z (aref hand-pos 2))
    (setq points (search-graspable-point pc :min-x (- x 200) :max-x (+ x 200) :min-y (- y 200) :max-y (+ y 200) :min-z (- z 200) :max-z (+ z 200) :iter-count 500))
    (when (< max-points points)
      (setq fetch-copy (copy-object *fetch*))
      (setq max-points points))
    (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos))
    (print hand-pos)
    (setq x (aref hand-pos 0))
    (setq y (aref hand-pos 1))
    (setq z (aref hand-pos 2))
    (search-graspable-point pc :min-x (- x 100) :max-x (+ x 100) :min-y (- y 100) :max-y (+ y 100) :min-z (- z 100) :max-z (+ z 100) :iter-count 500)
    (setq hand-pos (send (send *fetch* :rarm :end-coords :copy-worldcoords) :pos))
    (print hand-pos)
    (when (< max-points points)
      (setq fetch-copy (copy-object *fetch*))
      (setq max-points points))
    (setq *fetch* fetch-copy)
    (let ((cube (make-cube 200 200 200)) (region nil))
      (send cube :locate (float-vector x y z) :world)
      (send cube :worldcoords)
      (setq region (send cube :edges))
      (objects (append (list *fetch* pc) region)))
    ))


;; NOTE arguments of this function means region of the target space, not octomap region.
;; return map of where is not explored very much
;; (unknown-distribution :min-x *min-x* :max-x *max-x* :min-y *min-y* :max-y *max-y* :min-z *min-z* :max-z *max-z* :block-size 50.0)
(defun unknown-distribution (&key (min-x *min-x*) (max-x *max-x*) (min-y *min-y*) (max-y *max-y*) (min-z *min-z*) (max-z *max-z*) (block-size 50.0)
                                  (view-octomap-2d nil) (view-unknown-distribution t))
  (unless (and (eq (mod (round (- *max-x* *min-x*)) (round block-size)) 0)
               (eq (mod (round (- *max-y* *min-y*)) (round block-size)) 0)
               (eq (mod (round (- *max-z* *min-z*)) (round block-size)) 0))
    (format t "target region is not a multiple of block-size~%")
    (return-from unknown-distribution nil))
  (ros::spin-once)
  (let* ((occupied-pc (get-octo-pointcloud :type "occupied"))
         (unknown-pc (get-octo-pointcloud :type "unknown"))
         (octomap-2d (apply #'matrix (make-list (round (/ (- *max-x* *min-x*) *resolution*)) :initial-element (make-list (round (/ (- *max-y* *min-y*) *resolution*)) :initial-element 0)))))
    (when (eq occupied-pc nil)
      (format t "cannot find occupied grids~%")
      (setq occupied-pc (instance pointcloud :init)))
    (when (eq unknown-pc nil)
      (format t "cannot find unknown grids~%")
      (setq unknown-pc (instance pointcloud :init)))
    ;; look up occupied pointcloud firstly
    (dotimes (i (send occupied-pc :size))
      (let* ((occupied (matrix-row (send occupied-pc :points) i))
             (x (round (/ (- (aref occupied 0) *min-x* (/ *resolution* 2.0)) *resolution*)))
             (y (round (/ (- (aref occupied 1) *min-y* (/ *resolution* 2.0)) *resolution*)))) ;; x and y are index
        (setf (aref octomap-2d x y) -1)))
    ;; look up unknown pointcloud secondly
    (dotimes (j (send unknown-pc :size))
      (let* ((unknown (matrix-row (send unknown-pc :points) j))
             (x (/ (round (- (aref unknown 0) *min-x* *resolution*)) *resolution*))
             (y (/ (round (- (aref unknown 1) *min-y* *resolution*)) *resolution*))) ;; x and y are index
        (if (not (eq (aref octomap-2d x y) -1.0))
          (setf (aref octomap-2d x y) (+ 1 (aref octomap-2d x y)))
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
            (send cube :translate (float-vector (+ (* *resolution* (+ i 0.5)) *min-x*) (+ (* *resolution* (+ j 0.5)) *min-y*) *min-z*) :world)
            (setq vis-list (append vis-list (list cube)))))
        (setq *resolution* 10)
        (objects (append (list *fetch*) vis-list))
        (send *irtviewer* :change-background #f(0.3 0.3 0.1))
        (send *irtviewer* :redraw)))

    ;; calculate unknown-distribution matrix. this shows rough distribution of unknown area, and visualize it.
    (let* ((x-size (round (/ (- max-x min-x) block-size))) ;; x-size of target region, not octomap region. divide octomap area into (x-size * y-size) areas
           (y-size (round (/ (- max-y min-y) block-size))) ;; y-size of target region, not octomap region.
           (dist-mat (apply #'matrix (make-list x-size :initial-element (make-list y-size :initial-element 0))))
           (max-unknown 0) dist-vis-list cube)
      ;; octomap-2d is composed of very fine grids, so make it rougher and create dist-mat (distribution matrix of unknown grids)
      (dotimes (i (car (array-dimensions octomap-2d)))
        (dotimes (j (cadr (array-dimensions octomap-2d)))
          (if (and (<= (/ (- min-x *min-x*) *resolution*) i) (< i (/ (- max-x *min-x*) *resolution*))
                   (<= (/ (- min-y *min-y*) *resolution*) j) (< j (/ (- max-y *min-y*) *resolution*)))
            (progn
              (let ((x-index (floor (/ (- i (/ (- min-x *min-x*) *resolution*)) (/ block-size *resolution*))))
                    (y-index (floor (/ (- j (/ (- min-y *min-y*) *resolution*)) (/ block-size *resolution*)))))
                (setf (aref dist-mat x-index y-index)
                      (+ (aref octomap-2d i j) (aref dist-mat x-index y-index)))))
            )))
      (when view-unknown-distribution
        (dotimes (k (car (array-dimensions dist-mat)))
          (dotimes (l (cadr (array-dimensions dist-mat)))
            (if (< max-unknown (aref dist-mat k l))
              (setq max-unknown (aref dist-mat k l)))))
        (dotimes (k (car (array-dimensions dist-mat)))
          (dotimes (l (cadr (array-dimensions dist-mat)))
            (setq cube (make-cube (/ (- max-x min-x) x-size) (/ (- max-y min-y) y-size) *resolution*))
            (send cube :set-color (float-vector (max (/ (aref dist-mat k l) (float max-unknown)) 0) 0 0))
            (send cube :translate (float-vector (+ (* (/ (- max-x min-x) x-size) (+ k 0.5)) min-x) (+ (* (/ (- max-y min-y) y-size) (+ l 0.5)) min-y) *min-z*) :world)
            (setq dist-vis-list (append dist-vis-list (list cube)))))
        (objects (append (list *fetch*) dist-vis-list))
        (send *irtviewer* :change-background #f(0.3 0.3 0.1))
        (send *irtviewer* :redraw))
      dist-mat)
    ))



;; this function returns t if grasp is succeeded. Else, return nil.
(defun pre-grasp (target-coords)
  (let ((l-min *inf*) (r-min *inf*) (target-finger nil) (another-finger nil) (rotate-min *inf*) (rotate-angle 0) (interpolatingp nil))
    ;; first, find object
    (detect target-coords :close-frame-list (list (list "/left" "/top" ) (list "/right" "/top")) :close-thre 40)

    ;; open gripper and detect edge
    (send *ri* :grasp-with-condition
          0.1 ;; goal pos of gripper
          #'(lambda () (or (> (setq l-min (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/top")) :data) cons))) 50) ;; condition
                           (> (setq r-min (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/top")) :data) cons))) 50))))
    (if (> l-min r-min)
      (setq target-finger "/left")
      (setq target-finger "/right"))

    ;; measure direction of edge
    (send *fetch* :rarm :wrist-r :joint-angle (+ 90 (send *fetch* :rarm :wrist-r :joint-angle)))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 7000)
    (while (not (setq interplatingp (send *ri* :interpolatingp))) ;; wait until interpolatingp becomes t
      (ros::spin-once)
      (unix:usleep 10000))
    (while (setq interplatingp (send *ri* :interpolatingp))
      (ros::spin-once)
      (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance target-finger "/top")) :data) cons)) rotate-min)
        (setq rotate-min (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance target-finger "/top")) :data) cons)))
        (setq rotate-angle (aref (send *ri* :state :potentio-vector :wait-until-update t) 7))) ;; update wrist-r angle
      (unix:usleep (* 200 1000)))
    (send *fetch* :rarm :wrist-r :joint-angle rotate-angle)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    (send *ri* :wait-interpolation)

    ;; move hand horizontally such that target object come to center of grippers.
    (if (equal target-finger "/left")
      (progn (send *fetch* :rarm :move-end-pos #f(0 -100 0) :local :use-torso nil)
             (setq another-finger "/right"))
      (progn (send *fetch* :rarm :move-end-pos #f(0 100 0) :local :use-torso nil)
             (setq another-finger "/left")))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 7000)
    (while (not (setq interplatingp (send *ri* :interpolatingp))) ;; wait until interpolatingp becomes t
      (ros::spin-once)
      (unix:usleep 10000))
    (while (setq interplatingp (send *ri* :interpolatingp))
      (ros::spin-once)
      (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance target-finger "/top")) :data) cons))
               (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance another-finger "/top")) :data) cons)))
       (return)) ;; robot judges that the target object come to center of grippers
      (unix:usleep (* 200 1000)))

    ;; move hand down and grasp object
    (detect target-coords :close-frame-list (list (list "/left" "/top" ) (list "/right" "/top")) :close-thre 30)
    (send *ri* :start-grasp)
  ))


;; This is trial function, and do not use at experiment stage.
;; randomly extend hand to surface of table.
(defun random-grope ()
  (let* ((ec nil) (range 150.0)
         (rand-vec-x (- (random range) (/ range 2))) (rand-vec-y (- (random range) (/ range 2.0))) (rand-vec-rot (- (random 1.6) 0.8))
         )
    (send *ri* :start-grasp :wait t)
    (send *fetch* :inverse-kinematics (make-coords
                                       :pos (v+ (float-vector rand-vec-x rand-vec-y 0) (float-vector 400 0 150))
                                       :rpy (float-vector 0 pi/2 rand-vec-rot)))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (send *fetch* :rarm :move-end-pos #f(0 0 -200) :world :use-torso nil)
    (setq ec (send *fetch* :rarm :end-coords :copy-worldcoords))
    (detect ec :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :close-thre 30)
    (unix:sleep 4)
    ))

(defun inside-region (coords x-min x-max y-min y-max z-min z-max &optional (margin 0))
  (let ((x (aref (send coords :worldpos) 0))
        (y (aref (send coords :worldpos) 1))
        (z (aref (send coords :worldpos) 2)))
    (and (<= (- x-min margin) x (+ x-max margin))
         (<= (- y-min margin) y (+ y-max margin))
         (<= (- z-min margin) z (+ z-max margin)))))

(defun camera-request ()
  (ros::wait-for-service *camera-request-service*)
  (ros::service-call *camera-request-service* (instance std_srvs::EmptyRequest :init)))

(defun camera-stop ()
  (ros::wait-for-service *camera-stop-service*)
  (ros::service-call *camera-stop-service* (instance std_srvs::EmptyRequest :init)))

(defun octomap-reset ()
  (ros::wait-for-service *octomap-reset-service*)
  (ros::service-call *octomap-reset-service* (instance std_srvs::EmptyRequest :init)))

(defun sensor-calib (&optional (thre 90))
  (dotimes (i 100)
    (let ((min-temp *inf*) (min *inf*))
      (send *ri* :set-init-proximities)
      (unix:usleep (* 100 1000))
      (ros::spin-once)
      (dolist (l/r (list "/left" "/right"))
        (dolist (frame (list "/top" "/side1" "/side2" "/side3"))
          (setq min-temp (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance l/r frame)) :data) cons)))
          (when (< min-temp min)
            (setq min min-temp))))
      (when (< thre min)
        (format t "calibration succeeded~%")
        (return-from sensor-calib t))))
  (format t "failed to calibrate sensor~%"))

;; do not return pointcloud when loading this file
nil
