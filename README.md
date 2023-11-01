# Robotics

ros2 pkg create --build-type ament_python <NAME_DIRECT>

colcon build --packages-select <NAME_DIRECT>

source install/setup.bash

## Простые команды запуска

ros2 run turtlesim turtlesim_node

ros2 run turtlesim turtle_teleop_key
