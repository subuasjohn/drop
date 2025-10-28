#!/usr/bin/env python3
import gi, time
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

class RTSPClient:
    def __init__(self, video_uri, meta_uri):
        # --- Video pipeline ---
        self.video_pipeline = Gst.Pipeline.new("video-pipeline")
        self.video_src = Gst.ElementFactory.make("rtspsrc", "video_src")
        self.video_src.set_property("location", video_uri)
        self.video_src.set_property("latency", 50)
        self.video_src.connect("pad-added", self.on_video_pad)

        self.video_queue = Gst.ElementFactory.make("queue", "video_queue")
        self.depay = Gst.ElementFactory.make("rtph264depay", "depay")
        self.decoder = Gst.ElementFactory.make("avdec_h264", "decoder")
        self.convert = Gst.ElementFactory.make("videoconvert", "convert")
        self.vsink = Gst.ElementFactory.make("autovideosink", "vsink")

        for e in [self.video_src, self.video_queue, self.depay, self.decoder, self.convert, self.vsink]:
            self.video_pipeline.add(e)
        self.video_queue.link(self.depay)
        self.depay.link(self.decoder)
        self.decoder.link(self.convert)
        self.convert.link(self.vsink)

        # --- Metadata pipeline ---
        self.meta_pipeline = Gst.Pipeline.new("meta-pipeline")
        self.meta_src = Gst.ElementFactory.make("rtspsrc", "meta_src")
        self.meta_src.set_property("location", meta_uri)
        self.meta_src.set_property("latency", 50)
        self.meta_src.connect("pad-added", self.on_meta_pad)

        self.meta_queue = Gst.ElementFactory.make("queue", "meta_queue")
        self.jitter = Gst.ElementFactory.make("rtpjitterbuffer", "jitter")
        self.demux = Gst.ElementFactory.make("rtpssrcdemux", "demux")
        self.metasink = Gst.ElementFactory.make("fakesink", "metasink")
        self.metasink.set_property("signal-handoffs", True)
        self.metasink.connect("handoff", self.on_meta_handoff)

        for e in [self.meta_src, self.meta_queue, self.jitter, self.demux, self.metasink]:
            self.meta_pipeline.add(e)
        self.meta_queue.link(self.jitter)
        self.jitter.link(self.demux)
        # demux -> metasink linked automatically via caps

    # --- Video pad ---
    def on_video_pad(self, src, pad):
        caps = pad.query_caps(None).to_string()
        if "video" in caps:
            pad.link(self.video_queue.get_static_pad("sink"))
            print("→ Linked video track")


    def on_meta_pad(self, src, pad):
        caps = pad.query_caps(None).to_string()
        print("📡 Meta pad added:", caps)   # <-- add this
        if "application" in caps:
            pad.link(self.meta_queue.get_static_pad("sink"))
            print("→ Linked metadata track")


    def on_meta_handoff(self, sink, buffer, pad):
        ts = buffer.pts if buffer.pts != Gst.CLOCK_TIME_NONE else buffer.dts
        sec = ts / Gst.SECOND if ts != Gst.CLOCK_TIME_NONE else None
        success, mapinfo = buffer.map(Gst.MapFlags.READ)
        if success:
            data = bytes(mapinfo.data).decode("utf-8", errors="ignore")
            timestamp = f"{sec:.3f}s" if sec is not None else "no-ts"
            print(f"📩 Metadata @ {timestamp}: {data}")
            buffer.unmap(mapinfo)

    def run(self):
        self.video_pipeline.set_state(Gst.State.PLAYING)
        self.meta_pipeline.set_state(Gst.State.PLAYING)
        loop = GLib.MainLoop()
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
        self.video_pipeline.set_state(Gst.State.NULL)
        self.meta_pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    client = RTSPClient(
        video_uri="rtsp://127.0.0.1:8554/cam",
        meta_uri="rtsp://127.0.0.1:8554/cam_meta"
    )
    client.run()
