import rclpy
from std_msgs.msg import String
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class Publisher(Node):

    def __init__(self):
        super().__init__('publisher')
        self.publisher = self.create_publisher(Twist, '/bibi/cmd_vel', 10)
        self.subscription = self.create_subscription(Image, "/depth/image", self.listener_callback, 10)
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.bridge = CvBridge()
        self.is_obstacle = 0  
        
    def timer_callback(self):
        twist = Twist()
        twist.linear.x = -1.0 if not self.is_obstacle else 0.0
        self.publisher.publish(twist)

    def listener_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            print(f"CvBridgeError: {e}")
            return

        depth_array = np.array(depth_image, dtype=np.float32)
        h, w = depth_array.shape
        print(depth_array.shape, depth_array)

        self.is_obstacle = 1 if depth_array[120][w // 2] < 3 else 0

def main(args=None):
    rclpy.init(args=args)
    circling = Publisher()
    rclpy.spin(circling)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

