# Robotics
ros2 service call /clear std_srvs/srv/Empty

## Простые команды запуска

ros2 run turtlesim turtlesim_node

ros2 run turtlesim turtle_teleop_key

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

### -- 3.1 --

ros2 run service_full_name service

ros2 run service_full_name client Иванов Иван Иванович
