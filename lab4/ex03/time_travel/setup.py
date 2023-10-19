from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'time_travel'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vikatop',
    maintainer_email='v.tikhonova@g.nsu.ru',
    description='Learning tf2 with rclpy',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'turtle_tf2_broadcaster = time_travel.turtle_tf2_broadcaster:main',
            'turtle_tf2_listener = time_travel.turtle_tf2_listener:main',
        ],
    },
)
