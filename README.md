# Robotics

ros2 pkg create --build-type ament_cmake <NAME_DIRECT>

colcon build --packages-select <NAME_DIRECT>

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

### -- 3.2 --

colcon build --packages-select python_turtle_commands

ros2 run python_turtle_commands server

ros2 run python_turtle_commands client

### -- 3.3 --

cd /ros2_ws/bag_files/

ros2 bag play turtle_cmd_vel.mcap

ros2 topic echo /turtle1/pose >> pose_speed_x1.yaml

ros2 bag play turtle_cmd_vel.mcap --rate 2.0

ros2 topic echo /turtle1/pose >> pose_speed_x2.yaml

