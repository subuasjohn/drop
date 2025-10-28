#!/usr/bin/env python3
"""
Capture one frame from /dev/video0 at max resolution (4608x2592), pixelformat=RG10 using v4l2-ctl,
then interactively adjust red, green, and blue gains for white balance and save output.

Requires v4l2-ctl installed and accessible.

Run: python3 capture_and_balance.py
"""

import subprocess
import numpy as np
import cv2
import os

# Parameters
device = "/dev/video0"
width, height = 4032, 3040
raw_file = "frame.raw"
output_file = "output_balanced_final.png"

def capture_frame():
    print(f"Setting format: {width}x{height}, pixelformat=RG10 on {device}")
    fmt_cmd = [
        "v4l2-ctl",
        "--device", device,
        "--set-fmt-video",
        f"width={width},height={height},pixelformat=RG10"
    ]
    subprocess.run(fmt_cmd, check=True)

    print(f"Capturing one frame to {raw_file}...")
    capture_cmd = [
        "v4l2-ctl",
        "--device", device,
        "--stream-mmap=1",
        "--stream-count=1",
        f"--stream-to={raw_file}"
    ]
    subprocess.run(capture_cmd, check=True)

def load_and_process_raw():
    expected_size = width * height * 2  # 2 bytes per pixel (16 bits)
    actual_size = os.path.getsize(raw_file)
    if actual_size != expected_size:
        raise ValueError(f"Unexpected raw size: got {actual_size}, expected {expected_size}")

    with open(raw_file, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint16).reshape((height, width))

    # Even though format is RG10, it's stored as 16-bit, so treat as unpacked 16-bit.
    raw_16bit = raw

    # Normalize and gamma correct (gamma=1/2.2)
    img_norm = raw_16bit.astype(np.float32) / 65535.0
    gamma = 1 / 2.2
    img_gamma = np.power(img_norm, gamma)

    # Convert to 8-bit for OpenCV Bayer conversion
    img_8bit = (img_gamma * 255).astype(np.uint8)

    # Convert BayerRG to BGR color image
    bgr = cv2.cvtColor(img_8bit, cv2.COLOR_BayerRG2BGR)

    return bgr

def interactive_white_balance(bgr):
    last_balanced = None

    def update_gains(_):
        nonlocal last_balanced
        r_gain = cv2.getTrackbarPos('Red Gain x100', 'Adjust Gains') / 100.0
        g_gain = cv2.getTrackbarPos('Green Gain x100', 'Adjust Gains') / 100.0
        b_gain = cv2.getTrackbarPos('Blue Gain x100', 'Adjust Gains') / 100.0

        b, g, r = cv2.split(bgr)
        r_balanced = cv2.multiply(r, r_gain)
        g_balanced = cv2.multiply(g, g_gain)
        b_balanced = cv2.multiply(b, b_gain)

        balanced = cv2.merge([b_balanced, g_balanced, r_balanced])
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)

        # Resize for preview (e.g., max width 1280)
        preview_width = 1280
        scale = preview_width / balanced.shape[1]
        preview_height = int(balanced.shape[0] * scale)
        preview_img = cv2.resize(balanced, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

        last_balanced = balanced
        cv2.imshow('Adjust Gains', preview_img)

    cv2.namedWindow('Adjust Gains')
    cv2.createTrackbar('Red Gain x100', 'Adjust Gains', 110, 300, update_gains)
    cv2.createTrackbar('Green Gain x100', 'Adjust Gains', 100, 300, update_gains)
    cv2.createTrackbar('Blue Gain x100', 'Adjust Gains', 100, 300, update_gains)

    update_gains(0)

    print("Adjust white balance gains using sliders. Press 'q' to save and quit.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return last_balanced

def main():
    try:
        capture_frame()
        bgr = load_and_process_raw()
    except Exception as e:
        print(f"Error processing raw frame: {e}")
        return

    balanced = interactive_white_balance(bgr)

    if balanced is not None:
        cv2.imwrite(output_file, balanced)
        print(f"Saved balanced output to {output_file}")

    if os.path.exists(raw_file):
        os.remove(raw_file)

if __name__ == "__main__":
    main()
