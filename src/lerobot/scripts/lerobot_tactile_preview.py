#!/usr/bin/env python3
"""Start a local web server that shows a live tactile heatmap preview.

Usage::

    lerobot-tactile-preview --port /dev/ttyACM0 --http-port 8765
    # Open http://localhost:8765 in a browser.
"""

from __future__ import annotations

import argparse
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2
import numpy as np

from lerobot.cameras.tactile import TactileVisualizer
from lerobot.cameras.tactile.driver import TactileSensorDriver
from lerobot.cameras.tactile.runtime import TactileRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a live tactile heatmap preview page."
    )
    parser.add_argument(
        "--port", type=str, default="/dev/ttyACM0", help="Serial port of the tactile controller."
    )
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--mode", choices=("auto_push", "distributed_poll"), default="auto_push")
    parser.add_argument("--distributed-scale", type=float, default=0.1)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--heatmap-vmin", type=float, default=0.0)
    parser.add_argument("--heatmap-vmax", type=float, default=25.5)
    parser.add_argument(
        "--heatmap-colormap",
        choices=TactileVisualizer.available_colormaps(),
        default="turbo",
    )
    parser.add_argument("--heatmap-gamma", type=float, default=0.75)
    parser.add_argument("--rgb-vmax-fz", type=float, default=25.5)
    parser.add_argument("--rgb-vmax-shear", type=float, default=12.8)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibration-samples", type=int, default=50)
    parser.add_argument("--calibration-interval", type=float, default=0.05)
    parser.add_argument("--calibration-warmup-frames", type=int, default=20)
    parser.add_argument("--calibration-reducer", choices=("median", "mean"), default="median")
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--refresh-ms", type=int, default=250)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def placeholder_frame(message: str, size: int = 256) -> np.ndarray:
    image = np.zeros((size * 2, size * 2, 3), dtype=np.uint8)
    cv2.putText(image, "Tactile Preview", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(image, message, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return image


def make_handler(
    runtime: TactileRuntime,
    visualizer: TactileVisualizer,
    refresh_ms: int,
    logger: logging.Logger,
):
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self.serve_index()
                return
            if path == "/frame.jpg":
                self.serve_frame()
                return
            if path == "/healthz":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"ok")
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            logger.info("preview %s", fmt % args)

        def serve_index(self) -> None:
            html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Tactile Preview</title>
  <style>
    body {{
      margin: 0; font-family: monospace; background: #101418; color: #e6edf3;
      display: grid; place-items: center; min-height: 100vh;
    }}
    .wrap {{ width: min(92vw, 900px); }}
    img {{ width: 100%; border: 1px solid #2f3b45; background: #000; display: block; }}
    .hint {{ margin-top: 12px; opacity: 0.8; }}
  </style>
</head>
<body>
  <div class="wrap">
    <img id="frame" src="/frame.jpg" alt="tactile preview">
    <div class="hint">Auto-refreshing every {refresh_ms} ms</div>
  </div>
  <script>
    const frame = document.getElementById("frame");
    setInterval(() => {{ frame.src = "/frame.jpg?t=" + Date.now(); }}, {refresh_ms});
  </script>
</body>
</html>
"""
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def serve_frame(self) -> None:
            snapshot = runtime.get_snapshot(copy_snapshot=True)
            if snapshot is None:
                image = placeholder_frame("Waiting for tactile data ...", size=visualizer.output_size)
            else:
                image = visualizer.render_snapshot(snapshot, calibrated=True)

            ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Failed to encode frame")
                return

            body = encoded.tobytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PreviewHandler


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("tactile-preview")

    driver = TactileSensorDriver(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        enable_distributed=args.mode == "auto_push",
        distributed_scale=args.distributed_scale,
        logger=logger,
    )
    runtime = TactileRuntime(
        driver=driver,
        read_mode=args.mode,
        read_timeout=args.timeout,
        poll_interval=args.poll_interval,
        calibration_samples=args.calibration_samples,
        calibration_interval=args.calibration_interval,
        calibration_warmup_frames=args.calibration_warmup_frames,
        calibration_reducer=args.calibration_reducer,
        logger=logger,
    )
    visualizer = TactileVisualizer(
        output_size=args.output_size,
        heatmap_vmin=args.heatmap_vmin,
        heatmap_vmax=args.heatmap_vmax,
        heatmap_colormap=args.heatmap_colormap,
        heatmap_gamma=args.heatmap_gamma,
        rgb_vmax_fz=args.rgb_vmax_fz,
        rgb_vmax_shear=args.rgb_vmax_shear,
    )

    try:
        runtime.start()
        if runtime.wait_for_frame(timeout=3.0) is None:
            logger.warning("No tactile frames yet, preview will show a waiting image.")
        elif args.calibrate:
            runtime.calibrate()

        handler = make_handler(runtime, visualizer, args.refresh_ms, logger)
        server = ThreadingHTTPServer((args.host, args.http_port), handler)
        logger.info("Tactile preview at http://%s:%s", args.host, args.http_port)
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping tactile preview server")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
