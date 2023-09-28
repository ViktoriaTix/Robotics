import rclpy
from rclpy.node import Node
from my_custom_package.srv import FullNameSumService

class ServiceNameNode(Node):

    def __init__(self):
        super().__init__('service_name')
        self.service = self.create_service(
            FullNameSumService, 'summ_full_name', self.handle_service_request
        )

    def handle_service_request(self, request, response):
        last_name = request.last_name
        name = request.name
        first_name = request.first_name

        full_name = f"{last_name} {name} {first_name}"

        response.full_name = full_name
        return response

def main(args=None):
    rclpy.init(args=args)
    node = ServiceNameNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

