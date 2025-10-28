#!/usr/bin/env python3
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

VIDEO_URI = "rtsp://127.0.0.1:8554/cam"
META_URI  = "rtsp://127.0.0.1:8554/cam_meta"

class RTSPClient:
    def __init__(self, video_uri, meta_uri):
        self.pipeline = Gst.Pipeline.new("rtsp-client")

        # --- Video ---
        self.video_src = Gst.ElementFactory.make("rtspsrc", "video_src")
        self.video_src.set_property("location", video_uri)
        self.video_src.connect("pad-added", self.on_video_pad)
        self.pipeline.add(self.video_src)

        self.v_queue = Gst.ElementFactory.make("queue", "v_queue")
        self.depay = Gst.ElementFactory.make("rtph264depay", "v_depay")
        self.decoder = Gst.ElementFactory.make("avdec_h264", "v_decoder")
        self.convert = Gst.ElementFactory.make("videoconvert", "v_convert")
        self.vsink = Gst.ElementFactory.make("autovideosink", "vsink")
        for e in [self.v_queue, self.depay, self.decoder, self.convert, self.vsink]:
            self.pipeline.add(e)
        self.v_queue.link(self.depay)
        self.depay.link(self.decoder)
        self.decoder.link(self.convert)
        self.convert.link(self.vsink)

        # --- Metadata ---
        self.meta_src = Gst.ElementFactory.make("rtspsrc", "meta_src")
        self.meta_src.set_property("location", meta_uri)
        self.meta_src.connect("pad-added", self.on_meta_pad)
        self.pipeline.add(self.meta_src)

        self.m_queue = Gst.ElementFactory.make("queue", "m_queue")
        self.m_sink = Gst.ElementFactory.make("appsink", "m_sink")
        self.m_sink.set_property("emit-signals", True)
        self.m_sink.set_property("max-buffers", 5)
        self.m_sink.set_property("drop", True)
        self.m_sink.connect("new-sample", self.on_meta_sample)
        self.pipeline.add(self.m_queue)
        self.pipeline.add(self.m_sink)
        self.m_queue.link(self.m_sink)

    def on_video_pad(self, src, pad):
        if "video" in pad.query_caps(None).to_string():
            pad.link(self.v_queue.get_static_pad("sink"))
            print("→ Linked video track")

    def on_meta_pad(self, src, pad):
        if "application" in pad.query_caps(None).to_string():
            pad.link(self.m_queue.get_static_pad("sink"))
            print("→ Linked metadata track")

    def on_meta_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            data = buf.extract_dup(0, buf.get_size())
            print("📩 Metadata received:", data.decode())
        return Gst.FlowReturn.OK

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        GLib.MainLoop().run()

if __name__ == "__main__":
    client = RTSPClient(VIDEO_URI, META_URI)
    client.run()
