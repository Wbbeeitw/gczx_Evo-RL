#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np


def _ensure_repo_src_on_path() -> None:
    src_root = Path(__file__).resolve().parents[2]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)


_ensure_repo_src_on_path()

from lerobot.xiaomi.record_bi_piper_tactile import (
    TactileRuntimeBundle,
    TactileSideCamera,
    _load_tactile_classes,
    parse_bool,
    parse_optional_path,
)
from lerobot.xiaomi.tactile_defaults import (
    TACTILE_DEFAULT_CALIBRATE,
    TACTILE_DEFAULT_CALIBRATION_INTERVAL,
    TACTILE_DEFAULT_CALIBRATION_REDUCER,
    TACTILE_DEFAULT_CALIBRATION_SAMPLES,
    TACTILE_DEFAULT_CALIBRATION_WARMUP_FRAMES,
    TACTILE_DEFAULT_HEATMAP_COLORMAP,
    TACTILE_DEFAULT_HEATMAP_GAMMA,
    TACTILE_DEFAULT_HEATMAP_VMAX,
    TACTILE_DEFAULT_HEATMAP_VMIN,
    TACTILE_DEFAULT_MODE,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a browser-based preview page for left/right Xiaomi tactile heatmaps."
    )
    parser.add_argument("--xiaomi-xr0-root", type=parse_optional_path, default=None)
    parser.add_argument("--left-tactile-port", required=True, help="Left tactile controller serial port.")
    parser.add_argument("--right-tactile-port", required=True, help="Right tactile controller serial port.")
    parser.add_argument("--tactile.baudrate", "--tactile-baudrate", dest="tactile_baudrate", type=int, default=921600)
    parser.add_argument("--tactile.timeout", "--tactile-timeout", dest="tactile_timeout", type=float, default=1.0)
    parser.add_argument(
        "--tactile.mode",
        "--tactile-mode",
        dest="tactile_mode",
        choices=("auto_push", "distributed_poll"),
        default=TACTILE_DEFAULT_MODE,
    )
    parser.add_argument(
        "--tactile.distributed_scale",
        "--tactile-distributed-scale",
        dest="tactile_distributed_scale",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tactile.output_size",
        "--tactile-output-size",
        dest="tactile_output_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--tactile.calibrate",
        "--tactile-calibrate",
        dest="tactile_calibrate",
        type=parse_bool,
        default=TACTILE_DEFAULT_CALIBRATE,
    )
    parser.add_argument(
        "--tactile.calibration_samples",
        "--tactile-calibration-samples",
        dest="tactile_calibration_samples",
        type=int,
        default=TACTILE_DEFAULT_CALIBRATION_SAMPLES,
    )
    parser.add_argument(
        "--tactile.calibration_interval",
        "--tactile-calibration-interval",
        dest="tactile_calibration_interval",
        type=float,
        default=TACTILE_DEFAULT_CALIBRATION_INTERVAL,
    )
    parser.add_argument(
        "--tactile.calibration_warmup_frames",
        "--tactile-calibration-warmup-frames",
        dest="tactile_calibration_warmup_frames",
        type=int,
        default=TACTILE_DEFAULT_CALIBRATION_WARMUP_FRAMES,
    )
    parser.add_argument(
        "--tactile.calibration_reducer",
        "--tactile-calibration-reducer",
        dest="tactile_calibration_reducer",
        choices=("median", "mean"),
        default=TACTILE_DEFAULT_CALIBRATION_REDUCER,
    )
    parser.add_argument(
        "--tactile.heatmap_vmin",
        "--tactile-heatmap-vmin",
        dest="tactile_heatmap_vmin",
        type=float,
        default=TACTILE_DEFAULT_HEATMAP_VMIN,
    )
    parser.add_argument(
        "--tactile.heatmap_vmax",
        "--tactile-heatmap-vmax",
        dest="tactile_heatmap_vmax",
        type=float,
        default=TACTILE_DEFAULT_HEATMAP_VMAX,
    )
    parser.add_argument(
        "--tactile.heatmap_colormap",
        "--tactile-heatmap-colormap",
        dest="tactile_heatmap_colormap",
        default=TACTILE_DEFAULT_HEATMAP_COLORMAP,
    )
    parser.add_argument(
        "--tactile.heatmap_gamma",
        "--tactile-heatmap-gamma",
        dest="tactile_heatmap_gamma",
        type=float,
        default=TACTILE_DEFAULT_HEATMAP_GAMMA,
    )
    parser.add_argument(
        "--tactile.poll_interval",
        "--tactile-poll-interval",
        dest="tactile_poll_interval",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tactile.log_level",
        "--tactile-log-level",
        dest="tactile_log_level",
        default="INFO",
    )
    parser.add_argument("--page-title", default="BiTactile Heatmap Preview")
    parser.add_argument("--show-fps", type=parse_bool, default=True)
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind address.")
    parser.add_argument("--http-port", type=int, default=8765, help="HTTP port for the preview page.")
    parser.add_argument("--refresh-ms", type=int, default=250, help="Browser refresh interval in milliseconds.")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for the browser frame endpoint.")
    return parser


def _make_tactile_camera(side_name: str, port: str, bundle: TactileRuntimeBundle, args: argparse.Namespace) -> TactileSideCamera:
    return TactileSideCamera(
        side_name=side_name,
        port=port,
        runtime_bundle=bundle,
        baudrate=args.tactile_baudrate,
        timeout=args.tactile_timeout,
        mode=args.tactile_mode,
        distributed_scale=args.tactile_distributed_scale,
        output_size=args.tactile_output_size,
        heatmap_vmin=args.tactile_heatmap_vmin,
        heatmap_vmax=args.tactile_heatmap_vmax,
        heatmap_colormap=args.tactile_heatmap_colormap,
        heatmap_gamma=args.tactile_heatmap_gamma,
        poll_interval=args.tactile_poll_interval,
        calibrate=args.tactile_calibrate,
        calibration_samples=args.tactile_calibration_samples,
        calibration_interval=args.tactile_calibration_interval,
        calibration_warmup_frames=args.tactile_calibration_warmup_frames,
        calibration_reducer=args.tactile_calibration_reducer,
        log_level=args.tactile_log_level,
    )


def _stack_preview(left_rgb: np.ndarray, right_rgb: np.ndarray, *, title: str, fps_text: str | None) -> np.ndarray:
    header_h = 48
    width = max(left_rgb.shape[1], right_rgb.shape[1])
    canvas = np.zeros((header_h + left_rgb.shape[0] + right_rgb.shape[0], width, 3), dtype=np.uint8)
    canvas[header_h : header_h + left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
    canvas[header_h + left_rgb.shape[0] :, : right_rgb.shape[1]] = right_rgb

    cv2.putText(canvas, title, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(
        canvas,
        "Top: left tactile   Bottom: right tactile",
        (12, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        1,
    )
    if fps_text is not None:
        cv2.putText(canvas, fps_text, (max(width - 120, 12), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    return canvas


class PreviewState:
    def __init__(self, *, title: str, show_fps: bool) -> None:
        self.title = title
        self.show_fps = show_fps
        self._lock = threading.Lock()
        self._last_frame_t: float | None = None
        self._smoothed_fps = 0.0

    def render_canvas(self, left_camera: TactileSideCamera, right_camera: TactileSideCamera) -> np.ndarray:
        with self._lock:
            left_rgb = left_camera.read_image()
            right_rgb = right_camera.read_image()

            fps_text = None
            if self.show_fps:
                now_t = time.perf_counter()
                if self._last_frame_t is not None:
                    dt = max(now_t - self._last_frame_t, 1e-6)
                    fps = 1.0 / dt
                    self._smoothed_fps = fps if self._smoothed_fps <= 0.0 else (0.9 * self._smoothed_fps + 0.1 * fps)
                    fps_text = f"{self._smoothed_fps:5.1f} FPS"
                self._last_frame_t = now_t

            return _stack_preview(left_rgb, right_rgb, title=self.title, fps_text=fps_text)


def _encode_jpeg(image_rgb: np.ndarray, jpeg_quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 1, 100))],
    )
    if not ok:
        raise RuntimeError("Failed to encode tactile preview frame as JPEG.")
    return encoded.tobytes()


def make_handler(
    *,
    left_camera: TactileSideCamera,
    right_camera: TactileSideCamera,
    preview_state: PreviewState,
    refresh_ms: int,
    jpeg_quality: int,
    logger: logging.Logger,
) -> type[BaseHTTPRequestHandler]:
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                self._serve_index()
                return
            if path == "/frame.jpg":
                self._serve_frame()
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

        def _serve_index(self) -> None:
            html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{preview_state.title}</title>
  <style>
    body {{
      margin: 0;
      font-family: monospace;
      background: #101418;
      color: #e6edf3;
      display: grid;
      place-items: center;
      min-height: 100vh;
    }}
    .wrap {{
      width: min(92vw, 900px);
    }}
    img {{
      width: 100%;
      border: 1px solid #2f3b45;
      background: #000;
      display: block;
    }}
    .hint {{
      margin-top: 12px;
      opacity: 0.8;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <img id="frame" src="/frame.jpg" alt="bi tactile preview">
    <div class="hint">
      Top: left tactile | Bottom: right tactile<br>
      Auto-refreshing every {refresh_ms} ms
    </div>
  </div>
  <script>
    const frame = document.getElementById("frame");
    const refresh = () => {{
      frame.src = "/frame.jpg?t=" + Date.now();
    }};
    setInterval(refresh, {refresh_ms});
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

        def _serve_frame(self) -> None:
            image = preview_state.render_canvas(left_camera, right_camera)
            body = _encode_jpeg(image, jpeg_quality)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return PreviewHandler


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.tactile_log_level.upper(), logging.INFO))
    logger = logging.getLogger("bi-tactile-preview")

    tactile_runtime_cls, tactile_driver_cls, tactile_visualizer_cls = _load_tactile_classes(args.xiaomi_xr0_root)
    bundle = TactileRuntimeBundle(
        runtime_cls=tactile_runtime_cls,
        driver_cls=tactile_driver_cls,
        visualizer_cls=tactile_visualizer_cls,
    )
    left_camera = _make_tactile_camera("left", args.left_tactile_port, bundle, args)
    right_camera = _make_tactile_camera("right", args.right_tactile_port, bundle, args)
    preview_state = PreviewState(title=args.page_title, show_fps=args.show_fps)

    server: ThreadingHTTPServer | None = None
    try:
        left_camera.start()
        right_camera.start()

        handler = make_handler(
            left_camera=left_camera,
            right_camera=right_camera,
            preview_state=preview_state,
            refresh_ms=args.refresh_ms,
            jpeg_quality=args.jpeg_quality,
            logger=logger,
        )
        server = ThreadingHTTPServer((args.host, args.http_port), handler)
        logger.info("Bi-tactile preview available at http://%s:%s", args.host, args.http_port)
        logger.info("Open the page locally or use SSH port forwarding if the script runs on a remote server.")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping bi-tactile preview server")
    finally:
        if server is not None:
            server.server_close()
        try:
            left_camera.stop()
        finally:
            right_camera.stop()


if __name__ == "__main__":
    main()
