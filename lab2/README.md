### -- 2.8 --

cd launch/

ros2 launch three_turtles_launch.py

cd ~/ros2_ws/

ros2 run turtlesim turtle_teleop_key --ros-args --remap turtle1/cmd_vel:=/turtlesim1/turtle1/cmd_vel

### -- 2.9 --

colcon edit my_custom_package FullNameSumService.srv

### -- 2.10 --

ros2 run turtlesim turtlesim_node

ros2 run py_pubsub turtle_controller

ros2 topic pub /cmd_text std_msgs/String "data: 'move_forward'"
