#!/usr/bin/env roseus

(load "package://octomap_server/euslisp/grope-in-box.l")

(defun sample-tracing ()
  (send *ri* :go-grasp :pos 0.03 :wait t)
  (let ((count 0))
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
                                                     :close-thre (list 25 25
                                                                       hand-cover-thick hand-cover-thick
                                                                       (+ 15 gripper-handside-distance) ;; +10 is safety margin
                                                                       (+ 15 gripper-handside-distance))
                                                     :tm 7000)))
                (cond
                 ;; NOTE select to use horizontal move caused by "/top" sensors
                 ;; ((equal detected-frame (list "/left" "/top")) ;; object is near left top
                 ;;  (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/right" "/top")) :data) cons)) 40) ;; both left and right finger top is near object, start moving horizontally
                 ;;    (format t "start moving horizontally~%")
                 ;;    (return))
                 ;;  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (+ gripper-handside-distance 10) 0))
                 ;;                :close-frame-list (list (list "/right" "/side2")) :close-thre gripper-handside-distance :tm 3000)
                 ;;    (format t "something is right of my gripper~%"))
                 ;;  (return))
                 ;; ((equal detected-frame (list "/right" "/top")) ;; object is near right top
                 ;;  (when (< (reduce #'min (coerce (send (car (send *ri* :get-proximity-distance "/left" "/top")) :data) cons)) 40) ;; both left and right finger top is near object, start moving horizontally
                 ;;    (format t "start moving horizontally~%")
                 ;;    (return))
                 ;;  (when (detect (send (send *fetch* :rarm :end-coords :copy-worldcoords) :translate (float-vector 0 (* -1 (+ gripper-handside-distance 10)) 0))
                 ;;                :close-frame-list (list (list "/left" "/side2")) :close-thre gripper-handside-distance :tm 3000)
                 ;;    (format t "something is left of my finger~%"))
                 ;;  (return))
                 ((or (equal detected-frame (list "/left" "/top")) (equal detected-frame (list "/right" "/top"))) ;; object is near left top
                  (return))
                 ;; NOTE select to use horizontal move caused by "/top" sensors

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
      )))
