#!/usr/bin/env python3
"""
Shared helpers for capture nodes.

Driver-facing
- Manage a CameraConfig of device/width/height/fps/etc...
- DriverControls uses OpenCV to set format/FPS on the device.

Observer-side
- GStreamerCaps builds caps strings for pipelines.
- GStreamerDecodeChain builds the downstream decode/convert/appsink chain.
"""

import cv2
from typing import Dict, Optional

class CameraConfig:
    def __init__(self,
                 controls: Optional[Dict[str, int]] = None,
                 device: str = "/dev/video0",
                 fourcc: Optional[str] = None,
                 fps: int = 30,
                 height: int = 480,
                 width: int = 640):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fourcc = fourcc
        self.controls = controls

    def with_overrides(self, overrides: Dict) -> "CameraConfig":
        """Return a copy with provided keys applied if present."""
        return CameraConfig(device = overrides.get("device", self.device),
                            width = int(overrides.get("width", self.width)),
                            height = int(overrides.get("height", self.height)),
                            fps = int(overrides.get("fps", self.fps)),
                            fourcc = overrides.get("fourcc", self.fourcc),
                            controls = overrides.get("controls", self.controls))


class DriverControls:
    """Driver-modifying helpers built on OpenCV."""

    # Changing these typically requires reopening the device.
    RESTART_KEYS = ("device", "width", "height", "fourcc")

    def __init__(self, config: CameraConfig):
        self.config = config
        try:
            import cv2  # type: ignore
        except Exception:
            cv2 = None  # type: ignore
        self.cv2 = cv2

    def update(self, overrides: Dict) -> "DriverControls":
        """Return a new DriverControls with config merged from overrides."""
        return DriverControls(self.config.with_overrides(overrides))

    def requires_restart(self, overrides: Dict) -> bool:
        """
        Check whether provided overrides touch fields that usually require reopen.
        Caller can decide whether to act on this.
        """
        for key in self.RESTART_KEYS:
            if key in overrides and overrides[key] is not None:
                if overrides[key] != getattr(self.config, key):
                    return True
        return False

    def update_with_restart_check(
        self,
        overrides: Dict,
        auto_restart: bool = False,
        reopen_callback=None,
    ):
        """
        Merge overrides, report if restart is needed, optionally trigger a reopen callback.

        Returns (new_controls, restart_needed, reopened_capture)
        - new_controls: a DriverControls with merged config.
        - restart_needed: bool indicating whether a reopen is advised.
        - reopened_capture: result of reopen_callback(...) if invoked, else None.
        """
        restart_needed = self.requires_restart(overrides)
        updated = self.update(overrides)

        reopened = None
        if restart_needed and auto_restart and reopen_callback:
            reopened = reopen_callback(updated.config)

        return updated, restart_needed, reopened

    def apply_opencv_settings(self, cap):
        """
        Apply width/height/fps/fourcc to an open cv2.VideoCapture.
        Returns True if the capture appears usable.
        """
        if self.cv2 is None:
            raise RuntimeError("OpenCV not available but apply_opencv_settings was called")

        if self.config.fourcc:
            try:
                fourcc_val = self.cv2.VideoWriter_fourcc(*self.config.fourcc)
                cap.set(self.cv2.CAP_PROP_FOURCC, fourcc_val)
            except Exception:
                pass

        cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        cap.set(self.cv2.CAP_PROP_FPS, self.config.fps)
        return cap.isOpened()

    def open_opencv_capture(self):
        """
        Open and configure an OpenCV capture based on current config.
        Note: this touches the camera driver state.
        """
        if self.cv2 is None:
            raise RuntimeError("OpenCV not available but open_opencv_capture was called")

        cap = self.cv2.VideoCapture(self.config.device)
        ok = self.apply_opencv_settings(cap)
        return cap if ok else None


class GStreamerCaps:
    """
    Observer/consumer helper for building a caps string to request width/height/format.
    This does not assemble a full pipeline and does not change camera driver settings.
    """

    def __init__(self, width: int = 640, height: int = 480, format: str = "BGR"):
        self.width = int(width)
        self.height = int(height)
        self.format = format

    def update(self, width: Optional[int] = None, height: Optional[int] = None, format: Optional[str] = None) -> "GStreamerCaps":
        """Return a new caps builder with updated dimensions/format."""
        return GStreamerCaps(
            width=self.width if width is None else width,
            height=self.height if height is None else height,
            format=self.format if format is None else format,
        )

    def build(self) -> str:
        """Return a caps string for use in a pipeline."""
        return (
            "video/x-raw,format={fmt},width={w},height={h}"
        ).format(
            fmt=self.format,
            w=self.width,
            h=self.height,
        )


class GStreamerDecodeChain:
    """
    Observer/consumer helper for building the decode/convert/appsink chain.
    Abstracts out source details like RTSP or UDP; callers prepend their source.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        format: str = "BGR",
        decoder: str = "avdec_h264",
    ):
        self.width = int(width)
        self.height = int(height)
        self.format = format
        self.decoder = decoder

    def update(self, **kwargs) -> "GStreamerDecodeChain":
        """Return a new chain with updated width/height/format/decoder."""
        return GStreamerDecodeChain(
            width=kwargs.get("width", self.width),
            height=kwargs.get("height", self.height),
            format=kwargs.get("format", self.format),
            decoder=kwargs.get("decoder", self.decoder),
        )

    def build(self) -> str:
        """
        Build the downstream chain (depay/parse/decoder/convert/scale/caps/appsink).
        This does not change camera driver settings.
        """
        return (
            "rtph264depay ! h264parse ! {dec} ! videoconvert ! videoscale ! "
            "video/x-raw,format={fmt},width={w},height={h} ! "
            "appsink name=appsink emit-signals=true sync=false drop=true max-buffers=1"
        ).format(
            dec=self.decoder,
            fmt=self.format,
            w=self.width,
            h=self.height,
        )
