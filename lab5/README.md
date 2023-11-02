### -- 5.1 (5.2) --

colcon build --packages-select sam_bot_description

ros2 launch sam_bot_description display.launch.py

### -- 5.3 --

colcon build --packages-select task3

ros2 launch task3 diff_drive.launch.py

ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/robot/cmd_vel

### -- 5.4 --

colcon build --packages-select circle_movement

ros2 launch circle_movement circle_movement.launch.py

### -- 5.5 --

colcon build --packages-select robot_bibi

ros2 launch robot_bibi diff_drive.launch.py

