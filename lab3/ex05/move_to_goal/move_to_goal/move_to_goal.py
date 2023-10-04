from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from rclpy.node import Node
import rclpy
import sys
import math
import time

class TurtleBot(Node):

     def __init__(self):
         super().__init__('move_to_goal')
         self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
         self.subscriber = self.create_subscription(Pose, '/turtle1/pose', self.update_pose, 10)
         
         self.pose = Pose()
         
         self.timer = self.create_timer(0.1, self.move2goal)
         self.goal_x = float(sys.argv[1])
         self.goal_y = float(sys.argv[2])
         self.goal_theta = float(sys.argv[3]) * math.pi / 180


     def update_pose(self, data):
         self.pose = data

     def move2goal(self):
         msg = Twist()

         # Расстояние до цели и угол до цели
         distance = math.sqrt((self.goal_x - self.pose.x) ** 2 + (self.goal_y - self.pose.y) ** 2)
         angle = math.atan2(self.goal_y - self.pose.y, self.goal_x - self.pose.x)

         # Пропорциональный контроль для линейной и угловой скорости
         linear_vel = 1.0 * distance
         angular_vel = 4.0 * (angle - self.pose.theta)

         msg.linear.x = linear_vel
         msg.angular.z = angular_vel
         self.publisher.publish(msg)
         
         if distance < 0.1 and abs(angle) > 0.1:
            msg.angular.z = self.goal_theta
            self.publisher.publish(msg)
            
            # Добавляем счетчик и цикл для доворачивания к желаемому углу
            count = 0
            while count < 9:  
                self.publisher.publish(msg)
                time.sleep(0.1)  
                count += 1  
            
            msg.linear.x = 0.0  # Останавливаем линейное движение
            msg.angular.z = 0.0  # Останавливаем вращение
            
            self.get_logger().info("Goal Reached!! ")
            self.timer.cancel()
            self.publisher.publish(msg)
            quit()

def main(args=None):
    rclpy.init(args=args)
    x = TurtleBot()
    rclpy.spin(x)
    x.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

