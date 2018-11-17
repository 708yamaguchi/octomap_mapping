#!/usr/bin/env roseus

(load "package://bag_segmentation/euslisp/bag-manipulation-interface.l")
(load "package://bag_segmentation/euslisp/utils.l")


;; set working region
(setq *region* (list (* 0.4 1000) (* 0.7 1000) (* -0.15 1000) (* 0.15 1000) (* 0.3 1000) (* 0.6 1000)))
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


(defun main ()
  ;; settings for experiment
  ;; TODO use camera and decide target region
  (camera-stop)
  (octomap-reset)

  ;; set initial pose for experiment
  (send *fetch* :reset-pose)
  (send *fetch* :torso :waist-z :joint-angle *torso*)
  (send *fetch* :inverse-kinematics
        (make-coords :pos (float-vector (/ (+ *min-x* *max-x*) 2.0) (/ (+ *min-y* *max-y*) 2.0) *max-z*) :rpy (float-vector 0 pi/2 0)))
  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
  (send *ri* :wait-interpolation)
  (send *ri* :start-grasp :wait t)
  (unix:usleep (* 500 1000))
  (send *ri* :set-init-proximities)

  ;; grasp object with octomap and grope. TODO simultaneous execution of (create-map) and (decide-grope-point) may be needed.
  ;; flowchart of this (do-until-key) is loop of following steps:
  ;; 1. first create octomap, and then try to grasp the target object with the same octomap up to 5 times.
  ;; 2. if success, lift the object and finish. else, retry from the first step.
  (do-until-key ;; do not return from this loop until the robot succeeds to grasp the target object
   (let ((detectp nil) (graspp nil))
     ;; create octomap by tracing environment, using proximity sensors on finger.
     (dotimes (i 30)
       (create-map))

     ;; calculate the most graspable point. In the function below, *fetch* is set.
     (dotimes (i 5) ;; try to grasp the target object with the same octomap, up to 5 times.
       (decide-grope-point (get-octo-pointcloud :type "occupied")
                           :min-x *min-x* :max-x *max-x* :min-y *min-y* :max-y *max-y* :min-z *min-z* :max-z *max-z*)

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
                       :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :thre 30)
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
  )

;; (send *fetch* :reset-pose)
;; (send *fetch* :inverse-kinematics (make-coords :pos (float-vector (/ (+ *min-x* *max-x*) 2.0) (/ (+ *min-y* *max-y*) 2.0) *max-z*) :rpy (float-vector 0 pi/2 0))) :rotation-axis nil
;; (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
;; TODO: what method is best solution to create map?
(defun create-map ()
  (let ((target-coords (make-coords :rpy (float-vector 0 pi/2 0))) ;; where to move fetch's hand
        (direction (rad2deg (random pi))) (count 0))
    (send *ri* :start-grasp :wait t)
    ;; decide where to grope from now
    (setf (aref (send target-coords :worldpos) 0) (+ *min-x* (random (- *max-x* *min-x*))))
    (setf (aref (send target-coords :worldpos) 1) (+ *min-y* (random (- *max-y* *min-y*))))
    (setf (aref (send target-coords :worldpos) 2) *max-z*)
    ;; first, moving hand up
    (format t "move hand up. preparing...~%")
    (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
    (send *fetch* :torso :waist-z :joint-angle 350)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    (sensor-calib) ;; calibration proximity sensor's param. TODO Is this method needed here?
    ;; move hand above the target object
    (send target-coords :translate (float-vector 0 0 (- 350 50)) :world)
    (send *fetch* :inverse-kinematics target-coords :use-torso nil)
    (send *fetch* :rarm :wrist-r :joint-angle (+ (send *fetch* :rarm :wrist-r :joint-angle) direction)) ;; rotate arm randomly
    (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    ;; move hand down to target coords
    (format t "move hand to target coords~%")
    (send *fetch* :torso :waist-z :joint-angle 50)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    (send *ri* :wait-interpolation)

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
            ;; if side1 is near object, stop moving hand down
            (ros::spin-once)
            (if (or (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side1")) :data) cons)) 50)
                    (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side1")) :data) cons)) 50)
                    (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/side2")) :data) cons)) 50)
                    (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/side2")) :data) cons)) 50))
              (return-from create-map t))
            (if (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                        :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :thre 30)
              (return)))
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
                                    :thre 50)))
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
            (return-from create-map t))
          (if (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(100 0 0))
                      :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :thre 30)
            (return)))
        (progn
          (format t "robot hand is outside of target region~%")
          (return-from create-map t))))
    ))


;; (defun create-map ()
;;   (send *ri* :start-grasp :wait t)
;;   (ros::spin-once)
;;   (let ((potential-pc nil) (next-pos nil) (next-pos-cube nil))
;;     (setq potential-pc (send (get-octo-pointcloud :type "potential") :points))
;;     ;; TODO decide where to explore next
;;     (setq next-pos (matrix-row potential-pc (random (car (array-dimensions potential-pc)))))
;;     (objects (list *fetch* (setq next-pos-cube (send (make-cube 20 20 20) :locate next-pos))))
;;     (send next-pos-cube :set-color #f(1 0 0))
;;     ;; TODO decide where to explore next
;;     (send *fetch* :inverse-kinematics
;;           (make-coords :pos (v+ next-pos #f(0 0 200)) :rpy (float-vector 0 pi/2 0)) :use-torso nil)
;;     (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
;;     (send *ri* :wait-interpolation)
;;     (send *fetch* :rarm :move-end-pos #f(0 0 -200) :world :use-torso nil)
;;     (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
;;     (setq ec (send *fetch* :rarm :end-coords :copy-worldcoords))
;;     (detect ec :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :thre 30)
;;     (send *irtviewer* :draw-objects :flush t)
;;     (unix:sleep 4)
;;     )
;;   )


;; WIP.
(defun trace (target-coords-world) ;; argument(target-coords) is world coords, rotation of coords is NOT considered.
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (let* ((ec-world (send *fetch* :rarm :end-coords :copy-worldcoords))
         (target-coords-local (send (send ec-world :transformation (make-coords)) :transform (copy-object target-coords-world)))
         (move-pos (send target-coords-local :pos)) ;; move-pos is based on local coords
         (close-target-list nil) (away-target-list nil)) ;; sensor frame which is considered detection during tracing
    (print target-coords-world)
    (print ec-world)
    (print move-pos)
    (if (> (aref move-pos 0) 20) ;; if move direction is x-axis positive
      (setq close-target-list (append close-target-list (list (list "/left" "/top")) (list (list "/right" "/top")))))
    (if (> (aref move-pos 1) 20) ;; if move direction is y-axis positive
      (setq close-target-list (append close-target-list (list (list "/right" "/side2")))))
    (if (< (aref move-pos 1) -20) ;; if move direction is y-axis negative
      (setq close-target-list (append close-target-list (list (list "/left" "/side2")))))
    (if (> (aref move-pos 2) 20) ;; if move direction is z-axis positive
      (setq close-target-list (append close-target-list (list (list "/left" "/side1")) (list (list "/right" "/side1")))))
    (if (< (aref move-pos 2) -20) ;; if move direction is z-axis negative
      (setq close-target-list (append close-target-list (list (list "/left" "/side3")) (list (list "/right" "/side3")))))

    ;; trace flow:
    ;; 1. move toward the destination, 2. when detecting at side of gripper, move hand up, 3. when there is nothing under gripper, move hand down
    ;; repeat 1~3 above until hand reach the destinationv
    (print close-target-list)
    (setq target-coords-local target-coords-world)
    (setq away-target-list nil)

    ;; TODO : decide target-coords-local & close/away-target-list automatically
    (dotimes (i 2)
      (detect target-coords-local :close-frame-list close-target-list :away-frame-list )
      ;; (setq close-target-list hogehoge)
      (setq away-target-list close-frame-list)
      ;; calculate goal coords view from hand coords
      (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
      (setq ec-world (send *fetch* :rarm :end-coords :copy-worldcoords))
      (setq target-coords-local (send (send ec-world :transformation (make-coords)) :transform (copy-object target-coords-world)))
      )
    ;; TODO  end
    ;; test code
    ;; (progn (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
    ;;        (send *fetch* :rarm :move-end-pos #f(100 0 0) :world :use-torso nil)
    ;;        (trace (send *fetch* :rarm :end-coords :copy-worldcoords)))
    ))

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
(defun decide-grope-point (pc &key (min-x 300) (max-x 800) (min-y -300) (max-y 300) (min-z 400) (max-z 1000))
  (let ((hand-pos nil) (x nil) (y nil) (z nil) (fetch-copy nil) (points 0) (max-points 0))
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
(defun unknown-distribution (&key (min-x 300) (max-x 800) (min-y -300) (max-y 300) (min-z 400) (max-z 1000) (block-size 50.0))
  (ros::spin-once)
  (let* ((occupied-pc (get-octo-pointcloud :type "occupied"))
         (unknown-pc (get-octo-pointcloud :type "unknown"))
         (octomap-2d (apply #'matrix (make-list (round (/ (- *max-x* *min-x*) *resolution*)) :initial-element (make-list (round (/ (- *max-y* *min-y*) *resolution*)) :initial-element 0))))
         (x-size (floor (/ (- max-x min-x) block-size))) ;; x-size of target region, not octomap region
         (y-size (floor (/ (- max-y min-y) block-size))) ;; y-size of target region, not octomap region
         (dist-mat (apply #'matrix (make-list x-size :initial-element (make-list y-size :initial-element 0)))))
    (dotimes (i (send occupied-pc :size))
      (let* ((occupied (matrix-row (send occupied-pc :points) i))
             (x (round (/ (- (aref occupied 0) *min-x* (/ *resolution* 2.0)) x-size)))
             (y (round (/ (- (aref occupied 1) *min-y* (/ *resolution* 2.0)) y-size)))) ;; x and y are index
        (setf (aref octomap-2d x y) -1)))
    (dotimes (j (send unknown-pc :size))
      (let* ((unknown (matrix-row (send unknown-pc :points) j))
             (x (round (/ (- (aref unknown 0) *min-x* *resolution*) x-size)))
             (y (round (/ (- (aref unknown 1) *min-y* *resolution*) y-size)))) ;; x and y are index
        (if (not (eq (aref octomap-2d x y) -1))
          (setf (aref octomap-2d x y) (+ 1 (aref octomap-2d x y))))))
    octomap-2d
    ))


(defun pre-grasp ()
  ;; if grasp is succeeded, return t. else, return nil
  )


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
    (detect ec :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :thre 30)
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
