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

;; NOTE before using this function, make sure that object is in front of top of finger
;; this function returns t if grasp is succeeded. Else, return nil.
(defun pre-grasp ()
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (send *ri* :start-grasp :wait t)
  (let ((l-min *inf*) (r-min *inf*) (target-finger nil) (another-finger nil) (rotate-max-distance 0) (rotate-angle 0) (interpolatingp nil))
    ;; open gripper and detect edge
    (send *ri* :grasp-with-condition
          0.05 ;; goal pos of gripper
          #'(lambda ()
              (ros::spin-once)
              (and (< (setq l-min (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/top")) :data) cons))) 50) ;; condition
                   (< (setq r-min (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/top")) :data) cons))) 50))))
    (if (> l-min r-min)
      (setq target-finger "/left")
      (setq target-finger "/right"))

    ;; measure direction of edge
    (send *fetch* :rarm :wrist-r :joint-angle (+ 179 (send *fetch* :rarm :wrist-r :joint-angle)))
    (send *ri* :angle-vector (send *fetch* :angle-vector) 10000)
    (while (not (setq interplatingp (send *ri* :interpolatingp))) ;; wait until interpolatingp becomes t
      (ros::spin-once)
      (unix:usleep 10000))
    (while (setq interplatingp (send *ri* :interpolatingp))
      (ros::spin-once)
      (when (< rotate-max-distance (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance target-finger "/top")) :data) cons)))
        (setq rotate-max-distance (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance target-finger "/top")) :data) cons)))
        (setq rotate-angle (aref (send *ri* :state :potentio-vector :wait-until-update t) 7))) ;; update wrist-r angle
      (unix:usleep (* 10 1000)))
    (send *fetch* :rarm :wrist-r :joint-angle rotate-angle)
    (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    (format t "most graspable angle is: ~A~%" rotate-angle)

    ;; move hand horizontally such that target object come to center of grippers.
    (send *ri* :stop-grasp :wait t)
    (dotimes (i 3)
      (ros::spin-once)
      (if (> (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/top")) :data) cons))
             (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/top")) :data) cons)))
        (send *fetch* :rarm :move-end-pos #f(15 10 0) :local :use-torso nil)
        (send *fetch* :rarm :move-end-pos #f(15 -10 0) :local :use-torso nil))
      (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)
      (send *ri* :wait-interpolation))
    (sensor-calib)

    ;; move hand down and grasp object
    (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate #f(50 0 0) :local)
            :close-frame-list (list (list "/left" "/side0" ) (list "/right" "/side0")) :close-thre 15)
    (send *ri* :start-grasp :wait nil :effort 60)
    (ros::spin-once)
    (unix:sleep 2)
    (< 0.012 (send *ri* :gripper :position)) ;; if grasp is succeeded, return t. else, return nil
  ))

;; return if coords (arg) is inside of x/y/z-min/max
(defun inside-region (coords x-min x-max y-min y-max z-min z-max &optional (margin 0))
  (let ((x (aref (send coords :worldpos) 0))
        (y (aref (send coords :worldpos) 1))
        (z (aref (send coords :worldpos) 2)))
    (and (<= (- x-min margin) x (+ x-max margin))
         (<= (- y-min margin) y (+ y-max margin))
         (<= (- z-min margin) z (+ z-max margin)))))

;; return region which is inscribed region of region (arg).length of returned region is (n * block-size)
(defun inscribed-region (region &optional (block-size 50))
  (let* ((min-x (reduce #'min (mapcar #'(lambda (x) (aref x 0)) (send region :vertices))))
         (max-x (reduce #'max (mapcar #'(lambda (x) (aref x 0)) (send region :vertices))))
         (min-y (reduce #'min (mapcar #'(lambda (x) (aref x 1)) (send region :vertices))))
         (max-y (reduce #'max (mapcar #'(lambda (x) (aref x 1)) (send region :vertices))))
         (min-z (reduce #'min (mapcar #'(lambda (x) (aref x 2)) (send region :vertices))))
         (max-z (reduce #'max (mapcar #'(lambda (x) (aref x 2)) (send region :vertices))))
         (min-x-round (round min-x))
         (max-x-round (round max-x))
         (min-y-round (round min-y))
         (max-y-round (round max-y))
         (min-z-round (round min-z))
         (max-z-round (round max-z))
         (diff-x (/ (- (- max-x-round min-x-round) (* (floor (/ (- max-x-round min-x-round) block-size)) block-size)) 2.0))
         (diff-y (/ (- (- max-y-round min-y-round) (* (floor (/ (- max-y-round min-y-round) block-size)) block-size)) 2.0))
         (diff-z (/ (- (- max-z-round min-z-round) (* (floor (/ (- max-z-round min-z-round) block-size)) block-size)) 2.0))
         (ret-cube (make-cube (- (- max-x-round diff-x) (+ min-x-round diff-x))
                              (- (- max-y-round diff-y) (+ min-y-round diff-y))
                              (- max-z min-z)))) ;; do not consider z
    (send ret-cube :locate (float-vector (aref (send region :worldpos) 0)
                                         (aref (send region :worldpos) 1)
                                         (aref (send region :worldpos) 2)))
    (send ret-cube :worldcoords) ;; update position of edges of ret-cube
    ret-cube))


;; service call via topic. this aims to record the timing of service call for rosbag play
(ros::advertise "/camera_request" std_msgs::Empty 1)
(ros::subscribe "/camera_request" std_msgs::Empty #'camera-request-cb)
(defun camera-request-cb (msg)
  (ros::wait-for-service *camera-request-service*)
  (ros::service-call *camera-request-service* (instance std_srvs::EmptyRequest :init)))
(defun camera-request ()
  (ros::publish "/camera_request" (instance std_msgs::Empty :init)))
;;
(ros::advertise "/camera_stop" std_msgs::Empty 1)
(ros::subscribe "/camera_stop" std_msgs::Empty #'camera-stop-cb)
(defun camera-stop-cb ()
  (ros::wait-for-service *camera-stop-service*)
  (ros::service-call *camera-stop-service* (instance std_srvs::EmptyRequest :init)))
(defun camera-stop ()
  (ros::publish "/camera_stop" (instance std_msgs::Empty :init)))
;;
(ros::advertise "/octomap_reset" std_msgs::Empty 1)
(ros::subscribe "/octomap_reset" std_msgs::Empty #'camera-request-cb)
(defun octomap-reset-cb ()
  (ros::wait-for-service *octomap-reset-service*)
  (ros::service-call *octomap-reset-service* (instance std_srvs::EmptyRequest :init)))
(defun octomap-reset ()
  (ros::publish "/octomap_reset" (instance std_msgs::Empty :init)))


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

(defun rotate-wrist-r (&optional angle)
  (ros::spin-once)
  (send *fetch* :angle-vector (send *ri* :state :potentio-vector :wait-until-update t))
  (when (numberp angle)
    (progn
      (send *fetch* :rarm :wrist-r :joint-angle
            (+ angle (send *fetch* :rarm :wrist-r :joint-angle)))
      (send *ri* :angle-vector (send *fetch* :angle-vector) 3000)))
  (send *fetch* :rarm :wrist-r :joint-angle))


;; Method for writing words on IRTviewer
;; See https://github.com/euslisp/jskeus/pull/530
(in-package "GL")
(defmethod glviewsurface
  (:string
   ;; (x y str &optional (fid x:font-courb24)) ;;x::font-helvetica-bold-12))
   (x y str &optional (fid x:font-courb24)) ;;x::font-helvetica-bold-12))
   (send self :makecurrent)
   (glMatrixMode GL_PROJECTION)
   (glPushMatrix)
   (send self :2d-mode)
   (unless (eq (get self :glxusexfont) fid)
     (setf (get self :glxusexfont) fid)
     (glxUseXfont fid 32 96 (+ 1000 32)))
   (glRasterPos2i (round x) (- (send self :height) (round y)))
   (glListBase 1000)
   (glCallLists (length str) GL_UNSIGNED_BYTE str)
   (send self :3d-mode)
   (glMatrixMode GL_PROJECTION)
   (glPopMatrix)
   (glMatrixMode GL_MODELVIEW)
      ))
