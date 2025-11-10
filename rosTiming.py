"""
In [2]: ApproximateTimeSynchronizer?
Init signature: ApproximateTimeSynchronizer(fs, queue_size, slop, allow_headerless=False)
Docstring:     
Approximately synchronizes messages by their timestamps.

:class:`ApproximateTimeSynchronizer` synchronizes incoming message filters
by the timestamps contained in their messages' headers. The API is the same
as TimeSynchronizer except for an extra `slop` parameter in the constructor
that defines the delay (in seconds) with which messages can be synchronized.
The ``allow_headerless`` option specifies whether to allow storing
headerless messages with current ROS time instead of timestamp. You should
avoid this as much as you can, since the delays are unpredictable.
File:           /opt/ros/humble/local/lib/python3.10/dist-packages/message_filters/__init__.py
Type:           type
Subclasses:     
"""

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SlopExample(Node):
    def __init__(self):
        super().__init__('slop_example')

        # Wrap subscribers with message_filters.Subscriber
        image_sub = Subscriber(self, Image, '/camera/image_raw')
        imu_sub = Subscriber(self, Imu, '/imu/data')

        # ApproximateTimeSynchronizer takes (subs, queue_size, slop)
        # slop = max time difference (in seconds) to consider messages "synced"
        sync = ApproximateTimeSynchronizer(
            [image_sub, imu_sub],
            queue_size=10,
            slop=0.05  # 50 milliseconds
        )
        sync.registerCallback(self.synced_callback)

    def synced_callback(self, image_msg, imu_msg):
        t_img = image_msg.header.stamp
        t_imu = imu_msg.header.stamp
        dt = abs((t_img.sec + t_img.nanosec * 1e-9) -
                 (t_imu.sec + t_imu.nanosec * 1e-9))
        self.get_logger().info(f"Synced pair — slop: {dt:.6f} s")

def main(args=None):
    rclpy.init(args=args)
    node = SlopExample()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
