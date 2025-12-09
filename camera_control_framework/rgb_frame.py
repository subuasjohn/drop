#!/usr/bin/env python3

import cv2
import pickle
import rclpy
import threading
import time
import yaml
from avMav import Av
from custom_msgs.msg import UInt8MultiArrayCustom
from cv_bridge import CvBridge
from lib.cameraControls import CameraConfig, DriverControls
# from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
# from std_msgs.msg import UInt8MultiArray

"""
Add compression later on

from sensor_msgs.msg import CompressedImage
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = cv2.imencode('.jpg', frame)[1].tobytes()
        self.pub.publish(msg)
"""

class RgbFrame:
    # def __init__(self, node: Node | None = None, autospin = True, args = {}):
    def __init__(self, node: Node | None = None, autospin = True, args = None):
        """
        If 'node' is None, RgbFrame will create and manage its own Node.
        If 'node' is provided, it attaches to that node instead.
        If autospin is True, starts a background spin thread when self-managed.
        """
        self.args = args or {}
        self.node = node
        self.autospin = autospin
        self._own_node = node is None

        ## Create the node
        if self._own_node:
            if not rclpy.ok():
                rclpy.init()
            self.node = Node('rgb_frame')
        self._register_topics()

        self.pub_rate_hz = None          # publish rate (observer-side)
        self.pub_rate_period = None
        self.height = None               # driver-side height
        self.device = None               # driver-side device
        self.width = None                # driver-side width
        self.fps = None                  # driver-side fps

        ## Tie in avMav
        self.aV = Av(self.node)

        ## Default params
        self._declare_and_load_params()

        ## Camera integration
        self.cap_lock = threading.Lock()
        self.bridge = CvBridge()
        self.cam_cfg = CameraConfig(device = self.device,
                                    fps = self.fps,
                                    height = self.height,
                                    width = self.width)
        self.cam_drv = DriverControls(self.cam_cfg)

        with self.cap_lock:
            self.cap = self.cam_drv.open_opencv_capture()
            if self.cap is None:
                self.node.get_logger().error(f'[!] Failed to open camera device {self.device}')

        ## Handle timers
        self.timer = self.node.create_timer(self.pub_rate_period, self._timer_callback)

        ## kickstart
        if self._own_node and autospin:
            threading.Thread(target = lambda: rclpy.spin(self.node), daemon = True).start()
            self.node.get_logger().info('[~] rgb_frame running (self-spinning)')

            ### Odd bug not being able to use this with multiple resolution changes...
            # self.executor = MultiThreadedExecutor()
            # self.executor.add_node(self.node)

            # self._spin_thread = threading.Thread(target = self.executor.spin, daemon = True)
            # self._spin_thread.start()

            # self.node.get_logger().info("[~] rgb_frame running (self-spinning)")


    def _config_callback(self, msg: String):
        raw = msg.data.strip()

        ## yaml parsing
        try:
            cfg = yaml.safe_load(raw)
            if isinstance(cfg, dict):
                self._set_config(device = cfg.get('device'),
                                 height = cfg.get('height'),
                                 pub_rate = cfg.get('pub_rate'),
                                 width = cfg.get('width'),
                                 fps = cfg.get('fps'))
                return
        except Exception as E:
            self.node.get_logger().warn("[!] yaml parsing failed, reverting to fallback")

        ## singular fallback
        parts = raw.replace('=', ' ').split()
        if len(parts) == 2:
            key, val = parts[0].lower(), parts[1]
            if key == 'device':
                self._set_config(device = val)
                return
            elif key == 'height':
                self._set_config(height = int(val))
                return
            elif key == 'pub_rate':
                self._set_config(pub_rate = int(val))
                return
            elif key == 'width':
                self._set_config(width = int(val))
                return
            elif key == 'fps':
                self._set_config(fps = int(val))
                return
            else:
                return

        ## invalid handling
        self.node.get_logger().error(f"[!] Invalid device: '{msg.data}'")


    def _declare_and_load_params(self):
        if not self.node.has_parameter('pub_rate'):
            self.node.declare_parameter('pub_rate', 30)
        if not self.node.has_parameter('fps'):
            self.node.declare_parameter('fps', 30)
        if not self.node.has_parameter('height'):
            self.node.declare_parameter('height', 480)
        if not self.node.has_parameter('device'):
            self.node.declare_parameter('device', '/dev/video0')
        if not self.node.has_parameter('width'):
            self.node.declare_parameter('width', 640)
        
        if self.args.get('pub_rate') is not None:
            self.pub_rate_hz = int(self.args.get('pub_rate'))
        else:
            self.pub_rate_hz = int(self.node.get_parameter('pub_rate').value)
        self.pub_rate_period = 1.0 / self.pub_rate_hz

        if self.args.get('fps') is not None:
            self.fps = int(self.args.get('fps'))
        else:
            self.fps = int(self.node.get_parameter('fps').value)

        if self.args.get('height') is not None:
            self.height = int(self.args.get('height'))
        else:
            self.height = self.node.get_parameter('height').value

        if self.args.get('device') is not None:
            self.device = self.args.get('device')
        else:
            self.device = self.node.get_parameter('device').value

        if self.args.get('width') is not None:
            self.width = int(self.args.get('width'))
        else:
            self.width = self.node.get_parameter('width').value  


    def _register_topics(self):
        qos = QoSProfile(reliability = ReliabilityPolicy.BEST_EFFORT,
                         history = HistoryPolicy.KEEP_LAST,
                         depth = 10)
        self.cam_pub = self.node.create_publisher(Image, '/rgb_frame/image_raw', qos)
        self.cam_combined_pub = self.node.create_publisher(UInt8MultiArrayCustom,
                                                           '/rgb_frame/combined',
                                                           qos)
        self.config_sub = self.node.create_subscription(String, '/rgb_frame/config',
                                                        self._config_callback,
                                                        qos)


    def _set_config(self, device = None, height = None, pub_rate = None, width = None):
        """Unified function to reconfigure camera settings dynamically."""
        # # recreate_capture = False

        # ## device
        # if device is not None and device != self.device:
        #     self.node.get_logger().info(f'[~] Changing device -> {device}')
        #     self.device = device
        #     recreate_capture = True

        # ## resolution
        # if width is not None and height is not None:
        #     self.width = int(width)
        #     self.height = int(height)
        #     self.node.get_logger().info(f'[~] Resolution -> {self.width}x{self.height}')
        #     recreate_capture = True

        # ## recreate
        # if recreate_capture:
        #     try:
        #         with self.cap_lock:
        #             if self.cap is not None:
        #                 self.cap.release()
        #     except Exception as E:
        #         self.node.get_logger().warn(f'[!] Failed to release camera device: {E}')

        #     try:
        #         with self.cap_lock:
        #             self.cap = cv2.VideoCapture(self.device)

        #             if not self.cap.isOpened():
        #                 self.node.get_logger().info(f'[!] Failed to open camera device {self.device}')
        #                 return

        #             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        #             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        #             self.cap.set(cv2.CAP_PROP_FPS, self.pub_rate_hz)
        #         self.node.get_logger().info(f'[~] Reopened device {self.device}')
        #     except Exception as E:
        #         self.node.get_logger().error(f'[!] Could not reopen device {E}')

    def _reopen_capture(self, cfg: CameraConfig):
        with self.cap_lock:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(cfg.device)
            if not cap.isOpened():
                self.node.get_logger().error(f'[!] Failed to open camera device {cfg.device}')
                return None
            self.cam_drv.apply_opencv_settings(cap)
            self.cap = cap
            return self.cap


    def _set_config(self, device = None, height = None, pub_rate = None, width = None, fps = None):
        """Reconfigure camera settings. pub_rate handled elsewhere; driver changes use DriverControls."""
        # Driver-facing overrides
        overrides = {}
        if device is not None and device != self.device:
            overrides["device"] = device
            self.node.get_logger().info(f'[~] Changing device -> {device}')
        if width is not None and height is not None:
            new_w, new_h = int(width), int(height)
            if (new_w, new_h) != (self.width, self.height):
                overrides["width"] = new_w
                overrides["height"] = new_h
                self.node.get_logger().info(f'[~] Resolution -> {new_w}x{new_h}')
        if fps is not None:
            new_fps = int(fps)
            if new_fps > 0 and new_fps != self.fps:
                overrides["fps"] = new_fps
                self.node.get_logger().info(f'[~] Driver FPS -> {new_fps}')

        if overrides:
            updated_drv, restart_needed, reopened = self.cam_drv.update_with_restart_check(
                overrides,
                auto_restart=True,
                reopen_callback=self._reopen_capture,
            )
            self.cam_drv = updated_drv
            self.device = self.cam_drv.config.device
            self.width = self.cam_drv.config.width
            self.height = self.cam_drv.config.height
            self.fps = self.cam_drv.config.fps
            if not restart_needed and "fps" in overrides:
                with self.cap_lock:
                    if self.cap is not None:
                        self.cam_drv.apply_opencv_settings(self.cap)
                self.node.get_logger().info(f'[~] CAP_PROP_FPS -> {self.fps}')

        ## pub_rate
        if pub_rate is not None:
            new_pub_rate = int(pub_rate)
            if new_pub_rate > 0:

                ## store the pub_rate
                self.pub_rate_hz = new_pub_rate
                self.pub_rate_period = 1.0 / new_pub_rate

                ## update parameter
                self.node.set_parameters([rclpy.parameter.Parameter('pub_rate', value = new_pub_rate)])

                ## Abstracted to cameraControls
                ## pub_rate change
                # try:
                #     with self.cap_lock:
                #         self.cap.set(cv2.CAP_PROP_FPS, new_pub_rate)
                #     self.node.get_logger().info(f'[~] CAP_PROP_FPS -> {new_pub_rate}')
                # except:
                #     self.node.get_logger().warn('[!] Could not set CAP_PROP_FPS')

                ## reset timer
                try:
                    # self.timer.destroy()
                    self.timer.cancel()
                    self.timer = self.node.create_timer(self.pub_rate_period, self._timer_callback)
                    self.node.get_logger().info(f'[~] FPS -> {new_pub_rate}')
                except Exception as E:
                    self.node.get_logger().warn('[!] Could not set CAP_PROP_FPS')


    def _timer_callback(self):
        """Controls publish rate"""
        rVal = False
        with self.cap_lock:
            ret, frame = self.cap.read()
        now = self.node.get_clock().now().to_msg()
        if not ret:
            return
        try:
            if self.cam_combined_pub.get_subscription_count() > 0:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    return
                obj = {'alt': self.aV._alt.get('rel'),
                       'pub_rate': self.pub_rate_hz,
                       'frame': jpeg.tobytes(),
                       'height': self.height,
                       'lat': self.aV._gps.get('lat'),
                       'lon': self.aV._gps.get('lon'),
                       'width': self.width}
                payload = pickle.dumps(obj)
                msg = UInt8MultiArrayCustom()
                msg.data = list(payload)
                msg.header.stamp = now
                self.cam_combined_pub.publish(msg)
        except Exception as E:
            rVal = True
            self.node.get_logger().info(f'{E}')
        try:
            if self.cam_pub.get_subscription_count() > 0:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding = 'bgr8')
                msg.header.stamp = now
                self.cam_pub.publish(msg)
        except Exception as E:
            rVal = True
            self.node.get_logger().info(f'{E}')
        if rVal is True:
            return


    def shutdown(self):
        if self._own_node:
            ## Stop executor first -- Doesn't apply in this case until the MultiThread is brought back in
            try:
                self.executor.shutdown()
            except:
                pass

            ## Destroy the node
            try:
                self.node.destroy_node()
            except:
                pass

            ## Shutdown rclpy
            rclpy.shutdown()


def main(args = None):
    rclpy.init(args = args)
    rgb = RgbFrame()
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rgb.shutdown()

if __name__ == '__main__':
    main()

# ros2 topic pub /rgb_frame/config std_msgs/String "data: '{pub_rate: 50, height: 640, width: 480}'" --once
# ros2 topic pub /rgb_frame/config std_msgs/String "data: 'pub_rate: 20'" --once
