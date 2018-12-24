#!/usr/bin/env roseus

(load "package://octomap_server/euslisp/grope-with-octomap.l")
(load "package://octomap_server/euslisp/box-env2.l")

(defun main ()
  (decide-grope-point occupied-pointcloud target-region :x-offset 40 :y-offset 40 :z-offset 20 :visualize-searching t)
  )

(objects (list *fetch*))
(format t "Press Enter key to start~%")

(do-until-key
 (x::window-main-one)
 (unix:sleep 1))

(main)

(exit)
