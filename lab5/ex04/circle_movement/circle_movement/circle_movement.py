import rclpy, sys
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CirclePublisher(Node):

    def __init__(self):
        super().__init__('publisher')
        self.publisher = self.create_publisher(Twist, '/robot/cmd_vel', 10)
        timer_period = 0.5  
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):

        twist = Twist()
        twist.linear.x = 1.
        twist.angular.z = 0.5         
        self.publisher.publish(twist)
        
        
def main(args=None):
    rclpy.init(args=args)

    circling = CirclePublisher()

    rclpy.spin(circling)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    
