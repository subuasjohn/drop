gst-launch-1.0 -v rtspsrc location=rtsp://127.0.0.1:8554/cam latency=50 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
