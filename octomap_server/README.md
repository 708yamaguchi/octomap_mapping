Usage of multicloud
-------------------

- (On real robot) launch Proximity sensors, Octomap and Rviz.
```bash
rossetfetch
roslaunch octomap_server octomap_mapping_multicloud.launch
```

- Reset Octomap mapping.
```bash
rosservice call /octomap_server_contact/reset
```

- Stop pointcloud from camera.
```bash
rosservice call /camera_passthrough
```

- Stop pointcloud from proximity sensors.
```bash
rosservice call /proximitycloud_passthrough
```

- (For simulation) launch Octomap, Gazebo and Rviz.
```bash
rossetlocal
roslaunch octomap_server octomap_mapping_multicloud.launch real_sensor:=false
```
