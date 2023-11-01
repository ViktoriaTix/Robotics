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

### -- 3.5 --

colcon build --packages-select move_to_goal 

ros2 run move_to_goal move 1.0 1.0 90
