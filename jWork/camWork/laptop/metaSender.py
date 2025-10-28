#!/usr/bin/env python3
import gi, time
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

Gst.init(None)

UDP_HOST = "127.0.0.1"
UDP_PORT = 5004

pipeline = Gst.parse_launch(
    f"appsrc name=meta_src is-live=true format=time do-timestamp=true "
    f"! udpsink host={UDP_HOST} port={UDP_PORT} sync=false"
)

meta_src = pipeline.get_by_name("meta_src")

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

# Push ~20 fps
GLib.timeout_add(50, push_metadata)

pipeline.set_state(Gst.State.PLAYING)
print(f"✅ Metadata server running at udp://{UDP_HOST}:{UDP_PORT}")
GLib.MainLoop().run()
