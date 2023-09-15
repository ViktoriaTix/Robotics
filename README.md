# Robotics
source install/local_setup.bash

ros2 service call /clear std_srvs/srv/Empty


ros2 run turtlesim turtlesim_node

ros2 run turtlesim turtle_teleop_key
