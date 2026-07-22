Hector SLAM:

1. roslaunch mushr_gazebo gazebo-sim.launch
2. roslaunch mushr_mapping hector.launch
3. rosrun mushr_bringup keyboard_teleop_node
4. rviz (add ‘Map’)

OctoMap:
1. roslaunch mushr_gazebo gazebo-sim.launch
2. rosrun mushr_bringup keyboard_teleop_node
3. rosrun mushr_gazebo ground_truth.py
4. roslaunch mushr_mapping octomap.launch
5. rviz (use odom and add ‘MarkerArray’)
