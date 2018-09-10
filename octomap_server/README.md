Usage of multicloud
-------------------

1. launch octomap.
```
roslaunch octomap_server octomap_mapping_multicloud.launch
```

2. Publish pointcloud.
```
roseus multicloud-publisher.l
(main)
```

3. Stop pointcloud.
```
rosservice call /multicloud_passthrough
```
