#!/usr/bin/env python3

import cv2
import gi
import math
import rclpy
import threading
# import queue
import time
import pickle
import numpy as np
import yaml
from avMav import Av
from custom_msgs.msg import UInt8MultiArrayCustom
from cv_bridge import CvBridge
from lib.cameraControls import GStreamerDecodeChain
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
# from turbojpeg import TurboJPEG
# from std_msgs.msg import UInt8MultiArray
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

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

class RtspFrame:
    # def __init__(self, node: Node | None = None, autospin = True, args = {}):
    def __init__(self, node: Node | None = None, autospin = True, args = None):
        """
        If 'node' is None, RtspFrame will create and manage its own Node.
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
            self.node = Node('rtsp_frame')
        self._register_topics()

        self.pub_rate_hz = None
        self.pub_rate_period = None
        self.height = None
        self.rtsp_url = None
        self.width = None

        ## Tie in avMav
        self.aV = Av(self.node)

        ## Default params
        self._declare_and_load_params()

        ## GStreamer
        self.decode_chain = GStreamerDecodeChain(width = self.width,
                                                 height = self.height,
                                                 format = 'BGR',
                                                 decoder = 'avdec_h264')

        ## Camera integration
        self.bridge = CvBridge()
        # self.jpeg = TurboJPEG()

        # self._frame_queue = queue.Queue(maxsize = 5)
        self._frame_lock = threading.Lock()
        self._pipeline_lock = threading.Lock()
        self._config_lock = threading.Lock()
        self._latest_frame = None
        self._latest_ts = None

        self._last_frame_t = None
        self._stop_flag = threading.Event()
        self._pipeline_thread = None
        self._glib_loop = None
        self._pipeline = None
        self._appsink = None

        ## Init GStreamer once
        try:
            Gst.init(None)
        except Exception as E:
            self.node.get_logger().error('[!] GStreamer not able to be initialized')

        ## Keep retrying
        self._pipeline_thread = threading.Thread(target = self._rtsp_thread, daemon = True)
        self._pipeline_thread.start()

        ## Handle timers
        self.timer = self.node.create_timer(self.pub_rate_period, self._timer_callback)

        ## kickstart
        if self._own_node and autospin:
            # threading.Thread(target = lambda: rclpy.spin(self.node), daemon = True).start()
            self.executor = MultiThreadedExecutor()
            self.executor.add_node(self.node)

            self._spin_thread = threading.Thread(target = self.executor.spin, daemon = True)
            self._spin_thread.start()

            self.node.get_logger().info("[~] rtsp_frame running (self-spinning)")


    def _config_callback(self, msg: String):
        raw = msg.data.strip()

        ## yaml parsing
        try:
            cfg = yaml.safe_load(raw)
            if isinstance(cfg, dict):
                self._set_config(height = cfg.get('height'),
                                 pub_rate = cfg.get('pub_rate'),                                 
                                 rtsp_url = cfg.get('rtsp_url'),
                                 width = cfg.get('width'))
                return
        except:
            pass
        
        ## singular fallback
        parts = raw.replace('=', ' ').split()
        if len(parts) == 2:
            key, val = parts[0].lower(), parts[1]
            if key == 'pub_rate':
                self._set_config(pub_rate = int(val))
                return
            elif key == 'height':
                self._set_config(height = int(val))
                return
            elif key == 'rtsp_url':
                self._set_config(rtsp_url = val)
                return
            elif key == 'width':
                self._set_config(width = int(val))
                return
            else:
                return

        ## invalid handling
        self.node.get_logger().error(f"[!] Invalid input: '{msg.data}'")


    def _declare_and_load_params(self):
        if not self.node.has_parameter('pub_rate'):
            self.node.declare_parameter('pub_rate', 30)
        if not self.node.has_parameter('height'):
            self.node.declare_parameter('height', 480)
        if not self.node.has_parameter('rtsp_url'):
            self.node.declare_parameter('rtsp_url', 'rtsp://127.0.0.1:8554/video')
        if not self.node.has_parameter('width'):
            self.node.declare_parameter('width', 640)

        if self.args.get('pub_rate') is not None:
            self.pub_rate_hz = int(self.args.get('pub_rate'))
        else:
            self.pub_rate_hz = int(self.node.get_parameter('pub_rate').value)
        self.pub_rate_period = 1.0 / self.pub_rate_hz

        if self.args.get('height') is not None:
            self.height = int(self.args.get('height'))
        else:
            self.height = self.node.get_parameter('height').value
        if self.args.get('rtsp_url') is not None:
            self.rtsp_url = self.args.get('rtsp_url')
        else:
            self.rtsp_url = self.node.get_parameter('rtsp_url').value
        if self.args.get('width') is not None:
            self.width = int(self.args.get('width'))
        else:
            self.width  = self.node.get_parameter('width').value


    def _on_bus_message(self, bus, message):
        mtype = message.type
        if mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.node.get_logger().error(f'[~] GST ERROR from {message.src.get_name()}: {err.message}')
            if debug:
                self.node.get_logger().error(f'[~]   debug: {debug}')
            self._restart_pipeline()
        elif mtype == Gst.MessageType.EOS:
            self.node.get_logger().warn('[~] GST EOS (end-of-stream)')
            self._restart_pipeline()


    def _on_new_sample(self, sink):
        if self._stop_flag.is_set():
            return Gst.FlowReturn.FLUSHING

        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        success, mapinfo = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            w = caps.get_structure(0).get_value('width')
            h = caps.get_structure(0).get_value('height')
            if (self.width is None) or (self.height is None):
                self.width, self.height = int(w), int(h)

            arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
            frame = arr.reshape((int(h), int(w), 3))

            ## Keep fresh
            now_ts = time.time()
            with self._frame_lock:
                self._latest_frame = frame.copy()
                self._latest_ts = now_ts
                self._last_frame_t = now_ts

        except Exception:
            pass
        finally:
            buf.unmap(mapinfo)
        return Gst.FlowReturn.OK


    def _register_topics(self):
        qos = QoSProfile(reliability = ReliabilityPolicy.BEST_EFFORT,
                         history = HistoryPolicy.KEEP_LAST,
                         depth = 10)
        self.cam_pub = self.node.create_publisher(Image, '/rtsp_frame/image_raw', qos)
        self.cam_combined_pub = self.node.create_publisher(UInt8MultiArrayCustom, '/rtsp_frame/combined', qos)
        self.config_sub = self.node.create_subscription(String, '/rtsp_frame/config',
                                                        self._config_callback,
                                                        qos)


    def _rtsp_thread(self):
        """
        Runs a GLib mainloop with a GStreamer pipeline that pushes frames into _frame_queue.
        Auto-retries on stream errors/disconnects.
        """
        RETRY_DELAY = 2.0
        # decoder = self._select_decoder() # Bring in later on
        # print(decoder)
        while not self._stop_flag.is_set():
            try:
                chain = self.decode_chain.build()
                desc = f"rtspsrc location={self.rtsp_url} latency=0 drop-on-latency=true ! {chain}"
                # desc = (f'rtspsrc location={self.rtsp_url} latency=0 drop-on-latency=true ! '
                #         f'rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! '
                #         f'video/x-raw,format=BGR,width={self.width},height={self.height} ! '
                #         f'appsink name=appsink emit-signals=true sync=false drop=true max-buffers=1')

                with self._pipeline_lock:
                    self._pipeline = Gst.parse_launch(desc)
                    self._appsink = self._pipeline.get_by_name('appsink')
                    self._appsink.connect('new-sample', self._on_new_sample)

                ## Errors
                bus = self._pipeline.get_bus()
                bus.add_signal_watch()
                bus.connect('message', self._on_bus_message)

                ## Start pipeline
                self._pipeline.set_state(Gst.State.PLAYING)
                self.node.get_logger().info(f'[~] Pipeline PLAYING: {self.rtsp_url}')

                ## Run loop
                self._glib_loop = GLib.MainLoop()
                self._glib_loop.run()

            except Exception as E:
                self.node.get_logger().error(f'[~] Pipeline error: {E}')

            ## Cleanup
            try:
                if self._pipeline:
                    self._pipeline.set_state(Gst.State.NULL)
                self._pipeline = None
                self._appsink = None
            except Exception:
                pass

            if not self._stop_flag.is_set():
                self.node.get_logger().warn(f'[~] Reconnecting in {RETRY_DELAY:.1f}s...')
                time.sleep(RETRY_DELAY)


    def _restart_pipeline(self):
        with self._pipeline_lock:
            try:
                # quit safely on GLib loop thread
                if self._glib_loop:
                    GLib.idle_add(self._glib_loop.quit)
                if self._pipeline:
                    self._pipeline.set_state(Gst.State.NULL)
            except Exception as E:
                self.node.get_logger().error(f'[~] Unable to restart pipeline {E}')


    def _select_decoder(self):
        # Nvidia
        if Gst.ElementFactory.find("nvh264dec"):
            return "nvh264dec"

        # Intel VAAPI
        if Gst.ElementFactory.find("vaapih264dec"):
            return "vaapih264dec"

        # Raspberry Pi (modern)
        if Gst.ElementFactory.find("v4l2h264dec"):
            return "v4l2h264dec"

        # Raspberry Pi (older)
        if Gst.ElementFactory.find("omxh264dec"):
            return "omxh264dec"

        # CPU fallback
        return "avdec_h264"



    def _set_config(self, height = None, pub_rate = None, rtsp_url = None, width = None):
        """Unified function to reconfigure RTSP capture settings dynamically."""
        restart_needed = False

        ## pub_rate
        if pub_rate is not None:
            try:
                new_pub_rate_hz = int(pub_rate)
                if new_pub_rate_hz <= 0:
                    raise ValueError
            except Exception:
                self.node.get_logger().error(f'[!] Invalid pub_rate: {pub_rate}')
                new_pub_rate_hz = None

            if new_pub_rate_hz and new_pub_rate_hz != self.pub_rate_hz:
                old_pub_rate_hz = self.pub_rate_hz
                self.pub_rate_hz = new_pub_rate_hz
                self.pub_rate_period = 1.0 / new_pub_rate_hz

                # update ROS parameter
                self.node.set_parameters([rclpy.parameter.Parameter('pub_rate', value = new_pub_rate_hz)])

                # update timer period
                self.timer.cancel()
                self.timer = self.node.create_timer(self.pub_rate_period, self._timer_callback)
                self.node.get_logger().info(f'[~] Updated pub_rate: {old_pub_rate_hz} Hz -> {new_pub_rate_hz} Hz')


        ## url
        if rtsp_url is not None:
            if rtsp_url != self.rtsp_url:
                self.node.get_logger().info(f"[~] Changing RTSP URL:\n  {self.rtsp_url} -> {rtsp_url}")
                self.rtsp_url = rtsp_url
                restart_needed = True

        ## resolution
        if width is not None:
            try:
                width = int(width)
                if width > 0 and width != self.width:
                    self.node.get_logger().info(f"[~] Width: {self.width} -> {width}")
                    self.width = width
                    restart_needed = True
            except:
                self.node.get_logger().error(f"[!] Invalid width: {width}")

        if height is not None:
            try:
                height = int(height)
                if height > 0 and height != self.height:
                    self.node.get_logger().info(f"[~] Height: {self.height} -> {height}")
                    self.height = height
                    restart_needed = True
            except:
                self.node.get_logger().error(f"[!] Invalid height: {height}")

        ## restart
        if restart_needed:
            self.node.get_logger().warn("[~] Restarting GStreamer pipeline due to config change...")
            self.decode_chain = self.decode_chain.update(width=self.width, height=self.height)
            self._restart_pipeline()


    def _timer_callback(self):
        """Controls publish rate"""
        rVal = False

        # Grab latest frame (if any) under lock
        with self._frame_lock:
            frame = self._latest_frame
            ts = self._latest_ts
        if frame is None:
            return

        now = self.node.get_clock().now().to_msg()
        try:
            if self.cam_combined_pub.get_subscription_count() > 0:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if not ret:
                    return
                obj = {'alt': self.aV._alt.get('rel'),
                       'pub_rate': self.pub_rate_hz,
                       'frame': jpeg.tobytes(),
                       # 'frame': self.jpeg.encode(frame, quality=85),
                       'height': self.height,
                       'lat': self.aV._gps.get('lat'),
                       'lon': self.aV._gps.get('lon'),
                       'width': self.width}
                payload = pickle.dumps(obj)
                msg = UInt8MultiArrayCustom()
                msg.data = list(payload)
                msg.header.stamp = now
                self.cam_combined_pub.publish(msg)
        except Exception:
            rVal = True

        try:
            if self.cam_pub.get_subscription_count() > 0:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding = 'bgr8')
                msg.header.stamp = now
                self.cam_pub.publish(msg)
        except Exception:
            rVal = True

        if rVal is True:
            return


    def shutdown(self):
        ## gstreamer first
        try:
            self._stop_flag.set()
            if self._glib_loop is not None:
                try:
                    self._glib_loop.quit()
                except Exception:
                    pass
            if self._pipeline is not None:
                try:
                    self._pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
                self._pipeline_thread.join(timeout=2.0)
        except Exception:
            pass

        if self._own_node:
            ## Stop executor first
            try:
                self.executor.shutdown()
            except Exception:
                pass

            ## Destroy the node
            try:
                self.node.destroy_node()
            except Exception:
                pass

            ## Shutdown rclpy
            rclpy.shutdown()


def main(args = None):
    rclpy.init(args = args)
    rtsp = RtspFrame()
    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rtsp.shutdown()

if __name__ == '__main__':
    main()

# ros2 topic pub /rtsp_frame/config std_msgs/String "data: '{pub_rate: 50, height: 1280, width: 720}'" --once
# ros2 topic pub /rtsp_frame/config std_msgs/String "data: 'pub_rate: 20'" --once
