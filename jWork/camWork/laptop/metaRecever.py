#!/usr/bin/env python3
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

UDP_PORT = 5004

def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        data = buf.extract_dup(0, buf.get_size())
        print("📩 Metadata received:", data.decode())
    return Gst.FlowReturn.OK

pipeline = Gst.parse_launch(
    f"udpsrc port={UDP_PORT} "
    f"! appsink name=sink emit-signals=true max-buffers=10 drop=true"
)

sink = pipeline.get_by_name("sink")
sink.connect("new-sample", on_new_sample)

pipeline.set_state(Gst.State.PLAYING)
print(f"✅ Metadata client listening on udp://127.0.0.1:{UDP_PORT}")
GLib.MainLoop().run()
