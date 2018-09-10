Usage of multicloud
-------------------

1. (for simulation) launch Octomap, Gazebo and Rviz.
```
roslaunch octomap_server octomap_mapping_multicloud.launch
```

2. Stop pointcloud.
```
rosservice call /multicloud_passthrough
```
