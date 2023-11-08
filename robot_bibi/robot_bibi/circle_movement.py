import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class FigureEightPublisher(Node):
    def __init__(self):
        super().__init__('publisher')
        self.publisher = self.create_publisher(Twist, '/bibi/cmd_vel', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.time = 0

    def timer_callback(self):
        twist = Twist()
        twist.linear.x = -1.0
        twist.angular.z = 0.5         
        self.publisher.publish(twist)
        

def main(args=None):
    rclpy.init(args=args)
    figure_eight = FigureEightPublisher()
    rclpy.spin(figure_eight)
    figure_eight.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

