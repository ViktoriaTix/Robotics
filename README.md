# Robotics
ros2 service call /clear std_srvs/srv/Empty


ros2 run turtlesim turtlesim_node

ros2 run turtlesim turtle_teleop_key
-- 10 --
ros2 topic pub /cmd_text std_msgs/String "data: 'move_forward'"

