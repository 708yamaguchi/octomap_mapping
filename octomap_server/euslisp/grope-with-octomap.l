#!/usr/bin/env roseus

(load "package://bag_segmentation/euslisp/bag-manipulation-interface.l")
(load "package://bag_segmentation/euslisp/utils.l")


;; set working region
(setq *region* (list (* 0.4 1000) (* 0.7 1000) (* -0.15 1000) (* 0.15 1000) (* 0.3 1000) (* 0.6 1000)))
(setq *resolution* 0.01)
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


(unless (boundp '*ri*)
  (set-fetch))
(unless (boundp '*irtviewer*)
  (fetch)
  (objects (list *fetch*)))


(defun main ()
  (send *fetch* :reset-pose)
  (send *fetch* :torso :waist-z :joint-angle *torso*)
  (send *fetch* :inverse-kinematics (make-coords :pos #f(600 0 600) :rpy (float-vector 0 pi/2 0)) :use-torso nil)
  (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
  (send *ri* :wait-interpolation)
  (send *ri* :start-grasp :wait t)

  ;; first grope
  (let ((ec nil))
    (send *fetch* :rarm :move-end-pos #f(0 0 -400) :world :use-torso nil)
    (setq ec (send *fetch* :rarm :end-coords :copy-worldcoords))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 12000)
    (detect ec :close-frame-list (list (list "/left" "/top") (list "/right" "/top")) :thre 30))

  (dotimes (i 30)
    ;; (random-grope)
    (create-map)
    )
  )

;; (send *fetch* :reset-pose)
;; (send *fetch* :inverse-kinematics (make-coords :pos (float-vector (/ (+ *min-x* *max-x*) 2.0) (/ (+ *min-y* *max-y*) 2.0) *max-z*) :rpy (float-vector 0 pi/2 0))) :rotation-axis nil
;; (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
;; TODO: what method is best solution to create map?
(defun create-map ()
  (let ((target-coords (make-coords :rpy (float-vector 0 pi/2 0))) ;; where to move fetch's hand
        (direction (rad2deg (random pi))) (count 0))
    ;; decide where to grope from now
    (setf (aref (send target-coords :worldpos) 0) (+ *min-x* (random (- *max-x* *min-x*))))
    (setf (aref (send target-coords :worldpos) 1) (+ *min-y* (random (- *max-y* *min-y*))))
    (setf (aref (send target-coords :worldpos) 2) *max-z*)
    ;; first, moving hand up
    (format t "move hand up. preparing...~%")
    (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
    (send *fetch* :rarm :move-end-pos (float-vector 0 0 (- *max-z* (aref (send *fetch* :rarm :end-coords :worldpos) 2))) :world)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    ;; then move hand to target coords
    (format t "move hand to target coords~%")
    (send *fetch* :reset-pose)
    (send *fetch* :inverse-kinematics target-coords)
    (send *fetch* :rarm :wrist-r :joint-angle (+ (send *fetch* :rarm :wrist-r :joint-angle) direction)) ;; rotate arm randomly
    (send *ri* :angle-vector (send *fetch* :angle-vector) 5000)
    (send *ri* :wait-interpolation)
    ;; (send *fetch* :rarm :move-end-pos (float-vector 0 0 -300) :world) ;; 300 is tuned-parameter, needed to modify
    ;; (setq target-coords (send *fetch* :rarm :end-coords :copy-worldcoords))
    ;; (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
    ;; (detect target-coords :close-frame-list (list (list "/left" "/top") (list "/right" "/top")))
    ;; (unix:sleep 3) ;; make sure that robot stops motion by :cancel-angle-vector, and wait for updating octomap
    (format t "start tracing~%")
    ;; move hand until detecting object. move-end-pos 10cm, 10cm, 10cm ...  gradually.
    ;; TODO: make this while loop more intelligent. more smooth tracing is needed.
    (while (< count 3) ;; vertical hand move and horizontal hand move, alternately
      (format t "count: ~A~%" count)
      ;; move hand down
      (while t
        (format t "move hand down~%")
        (if (inside-region (send *fetch* :rarm :end-coords :copy-worldcoords)
                           *min-x* *max-x* *min-y* *max-y* *min-z* *max-z* 20)
          (progn
            (format t "move hand until detecting object~%")
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
            (if (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(0 0 -100))
                        :close-frame-list (list (list "/left" "/side3") (list "/right" "/side3")) :thre 30)
              (return)))
          (progn
            (format t "robot hand is outside of target region~%")
            (return-from create-map t))))
      (setq count (+ 1 count))
      )

    ;; ;; before version
    ;; (let* ((region-x (- *max-x* *min-x*)) (region-y (- *max-y* *min-y*))
    ;;        (par (random (* 2 (+ region-x region-y))))
    ;;        target-x target-y ec direction wrist-rotate-angle)
    ;;   (if (<= par region-x)
    ;;     (progn
    ;;       (setq target-x (- *max-x* par))
    ;;       (setq target-y *max-y*)))
    ;;   (if (<= region-x par (+ region-x region-y))
    ;;     (progn
    ;;       (setq target-x *min-x*)
    ;;       (setq target-y (- *max-y* (- par region-x)))))
    ;;   (if (<= (+ region-x region-y) par (+ (* 2 region-x) region-y))
    ;;     (progn
    ;;       (setq target-x (+ (- par region-x region-y) *min-x*))
    ;;       (setq target-y *min-y*)))
    ;;   (if (<= (+ (* 2 region-x) region-y) par)
    ;;     (progn
    ;;       (setq target-x *max-x*)
    ;;       (setq target-y (+ (- par (* 2 region-x) region-y) *min-y*))))
    ;;   (setq ec (send *fetch* :rarm :end-coords :copy-worldcoords))
    ;;   (setq direction (send ec :inverse-transform-vector (float-vector target-x target-y (aref (send ec :worldpos) 2))))
    ;;   (setq wrist-rotate-angle (rad2deg (atan2 (aref direction 1) (* -1 (aref direction 2)))))
    ;;   (send *fetch* :rarm :wrist-r :joint-angle (+ (send *fetch* :rarm :wrist-r :joint-angle) wrist-rotate-angle))
    ;;   (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    ;;   (send *ri* :wait-interpolation)
    ;;   (setq target-coords (send *fetch* :rarm :end-coords :copy-worldcoords))
    ;;   (setf (aref (send target-coords :worldpos) 0) target-x)
    ;;   (setf (aref (send target-coords :worldpos) 1) target-y)
    ;;   (format t "target end-coords is ~A~%" target-coords)
    ;;   (detect target-coords :close-frame-list (list (list "/left" "/side3") (list "/right" "/side3")) :thre 30)
    ;;   (format t "now, end-coords is ~A~%" (send *fetch* :rarm :end-coords :copy-worldcoords))
    ;;   (unix:sleep 1) ;; make sure that robot stops motion by :cancel-angle-vector
    ;;   )
    ;; ;; before version
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
(defun get-octo-pointcloud (&key (type nil))
  (let ((octo-cloud nil) (eus-octo-cloud nil) (color nil) (colors nil))
    ;; subscribe octomap pointcloud
    (cond
     ;; ((equal type "free")
     ;;  (setq color (list 0 1 0))
     ;;  (setq octo-cloud (one-shot-subscribe "/octomap_free_point_cloud_centers" sensor_msgs::Pointcloud2)))
     ((equal type "occupied")
      (setq color (list 0 0 1))
      (setq octo-cloud (one-shot-subscribe "/octomap_occupied_point_cloud_centers" sensor_msgs::Pointcloud2)))
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


;; decide next grope point which is desirable for creating octomap
;; TODO this method is no more needed.
(defun calc-grope-point (pc)
  (let* ((point-list (send pc :point-list))
         (point-mask (make-matrix (round (/ (- *max-x* *min-x*) *resolution*))
                                  (round (/ (- *max-y* *min-y*) *resolution*))))
         (density-map (copy-object point-mask))
         (density-map-tile nil) (tile nil)
         (adjacent-point-num 0))
    ;; make 2D map above which at least one grid is occupied.
    (dolist (point point-list)
      (setf (aref point-mask
                  (round (/ (- (aref point 0) *min-x* (/ *resolution* 2.0)) *resolution*))
                  (round (/ (- (aref point 1) *min-y* (/ *resolution* 2.0)) *resolution*))) 1))
    ;; calc number of adjacent masked pixel for each pixel of 2D map (pointmask)
    (dotimes (x (length (matrix-column point-mask 0)))
      (dotimes (y (length (matrix-row point-mask 0)))
        (dolist (i (list -1 0 1))
          (dolist (j (list -1 0 1))
            (when (and (<= 0 (+ x i) (- (round (/ (- *max-x* *min-x*) *resolution*)) 1))
                       (<= 0 (+ y j) (- (round (/ (- *max-y* *min-y*) *resolution*)) 1)))
              (when (eq (round (aref point-mask (+ x i) (+ y j))) 1)
                (setq adjacent-point-num (+ adjacent-point-num 1))))))
        (setf (aref density-map x y) adjacent-point-num)
        (setq tile (make-cube *resolution* *resolution* 1))
        (send tile :set-color
              (if (eq adjacent-point-num 0)
                (float-vector 1 1 1)
                (float-vector (/ (* adjacent-point-num 1.0) 8.0) 0 0)))
        (send tile :locate (float-vector (+ *min-x* (/ *resolution* 2.0) (* x *resolution*))
                                         (+ *min-y* (/ *resolution* 2.0) (* y *resolution*))
                                         *min-z*) :world)
        (setq density-map-tile (append density-map-tile (list tile)))
        (setq adjacent-point-num 0)))
    ;; for visualization
    (objects (append (list *fetch* pc) density-map-tile))
    nil
    ))


;; for debug, call (search-graspable-point *pc1)
(load "package://octomap_server/euslisp/av-and-groped-pc1.l")
(load "package://octomap_server/euslisp/av-and-groped-pc2.l")
(setq *pc1* (cadr av-and-groped-pc1))
(setq *pc2* (cadr av-and-groped-pc1))
;; TODO decide grope point based on height map. hint is in (calc-grope-point)
;; TODO use pyramid algorithm when searching grope point
(defun search-graspable-point (pc)
  (let* (;*min-x* *max-x* *min-y* *max-y* *min-z* *max-z* region
         stick-l stick-r
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

    (let ((cube (make-cube (- *max-x* *min-x*) (- *max-y* *min-y*) (- *max-z* *min-z*))))
      (send cube :locate (float-vector (/ (+ *max-x* *min-x*) 2.0)
                                       (/ (+ *max-y* *min-y*) 2.0)
                                       (/ (+ *max-z* *min-z*) 2.0)) :world)
      (send cube :worldcoords)
      (setq region (send cube :edges)))

    ;; assoc the virtual gripper model
    (send gripper-l :set-color #f(1 0 0))
    (send gripper-r :set-color #f(0 1 0))
    (send gripper-inside :set-color #f(0 0 1))
    ;; stick is thexy-plane field occupied by gripper
    (setq stick-l (send (make-cube gr-x gr-y (- *max-z* *min-z*)) :locate (float-vector 0 (/ gr-open -2) (/ (+ *max-z* *min-z*) 2.0))))
    (send stick-l :set-color #f(1 0 0 0.5))
    (setq stick-r (send (make-cube gr-x gr-y (- *max-z* *min-z*)) :locate (float-vector 0 (/ gr-open 2) (/ (+ *max-z* *min-z*) 2.0))))
    (send stick-r :set-color #f(0 1 0 0.5))
    (send gripper-inside :assoc gripper-l)
    (send gripper-inside :assoc gripper-r)
    (send gripper-inside :assoc stick-l)
    (send gripper-inside :assoc stick-r)

    ;; sort points in the order of z-height
    ;; (sort) is destructively sort
    (sort point-list-sorted #'(lambda (x y) (> (aref x 2) (aref y 2))))
    ;; search for most graspable point
    (dotimes (i 1000)
      ;; reset variables
      (setq highest-z nil)
      (setq inside-point 0)
      ;; move gripper randomly and get the highest point under the gripper
      ;; update objects related counting pointcloud
      (send stick-l :worldcoords)
      (send stick-r :worldcoords)
      (send gripper-inside :worldcoords)
      ;; set random point to grope
      (setq random-x (+ (random (- *max-x* *min-x*)) *min-x*))
      (setq random-y (+ (random (- *max-y* *min-y*)) *min-y*))
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
        (send stick-l :locate (float-vector 0 (/ gr-open -2) (/ (+ *max-z* *min-z*) 2.0)) :world)
        (send stick-r :locate (float-vector 0 (/ gr-open 2) (/ (+ *max-z* *min-z*) 2.0)) :world)
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
        (objects (append (list *fetch* pc) region)))
      (fotmat t "cannot find graspable point.~%"))
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

;; do not return pointcloud when loading this file
nil
