import cv2
import numpy as np
import subprocess

width, height = 4608, 2592

# Start FFmpeg subprocess
ffmpeg_cmd = [
    'ffmpeg',
    '-f', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', '15',  # Frame rate
    '-i', '-',  # Read from stdin
    '-f', 'rtsp',
    '-rtsp_transport', 'tcp',
    'rtsp://0.0.0.0:8554/live.sdp'
]
ffmpeg = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Your capture loop
while True:
    frame = get_processed_bgr_frame()  # Replace with your Bayer → BGR decode logic
    ffmpeg.stdin.write(frame.tobytes())
