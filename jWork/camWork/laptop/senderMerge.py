#!/usr/bin/env python3
import gi, time
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, GLib

Gst.init(None)

class RTSPServer:
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        mounts = self.server.get_mount_points()

        # --- Video RTSP ---
        video_factory = GstRtspServer.RTSPMediaFactory()
        video_factory.set_launch(
            "( v4l2src device=/dev/video0 ! videoconvert ! "
            "x264enc tune=zerolatency bitrate=4000 speed-preset=superfast key-int-max=30 ! "
            "rtph264pay name=pay0 pt=96 )"
        )
        video_factory.set_shared(True)
        mounts.add_factory("/cam", video_factory)

        # --- Metadata RTSP ---
        meta_factory = GstRtspServer.RTSPMediaFactory()
        meta_factory.set_launch(
            "( appsrc name=meta_src is-live=true format=time do-timestamp=true "
            "caps=\"application/x-rtp,media=application,encoding-name=X-GST-META,clock-rate=90000\" "
            "! rtpstreampay name=pay0 pt=127 )"
        )
        meta_factory.set_shared(True)
        meta_factory.connect("media-configure", self.on_meta_configure)
        mounts.add_factory("/cam_meta", meta_factory)

        self.server.attach(None)
        print("✅ RTSP server running:")
        print("   Video:    rtsp://127.0.0.1:8554/cam")
        print("   Metadata: rtsp://127.0.0.1:8554/cam_meta")

    def on_meta_configure(self, factory, media):
        """Start pushing metadata once the RTSP media pipeline exists."""
        pipeline = media.get_element()
        meta_src = pipeline.get_by_name("meta_src")
        if not meta_src:
            print("❌ meta_src not found")
            return

        print("✅ meta_src ready, starting metadata push loop")

        def push_metadata():
            text = f"counter={int(time.time())}"
            buf = Gst.Buffer.new_wrapped(text.encode("utf-8"))
            ts = int(time.time() * Gst.SECOND)
            buf.pts = buf.dts = ts
            ret = meta_src.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                print("⚠️ push-buffer failed:", ret)
            print("📤 Pushed metadata:", text)
            return True

        GLib.timeout_add(50, push_metadata)  # ~20 fps

GObject.threads_init()
server = RTSPServer()
loop = GLib.MainLoop()
loop.run()
