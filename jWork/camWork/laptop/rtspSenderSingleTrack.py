#!/usr/bin/env python3
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

Gst.init(None)

class RTSPServer:
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        self.factory = GstRtspServer.RTSPMediaFactory()
        self.factory.set_launch(
            "( v4l2src device=/dev/video0 ! videoconvert ! x264enc tune=zerolatency bitrate=4000 speed-preset=superfast key-int-max=30 ! rtph264pay config-interval=1 pt=96 name=pay0 )"
        )
        self.factory.set_shared(True)
        self.server.get_mount_points().add_factory("/cam", self.factory)
        self.server.attach(None)
        print("RTSP server running at rtsp://127.0.0.1:8554/cam")

GObject.threads_init()
server = RTSPServer()
loop = GObject.MainLoop()
loop.run()
