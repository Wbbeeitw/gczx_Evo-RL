#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TactileCamera: a LeRobot Camera backed by a tactile sensor controller."""

from __future__ import annotations

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_tactile import TactileCameraConfig
from .driver import TactileSensorDriver
from .runtime import TactileRuntime
from .visualizer import TactileVisualizer

logger = logging.getLogger(__name__)


class TactileCamera(Camera):
    """A virtual camera that renders tactile pressure data as RGB images.

    This camera implements the standard LeRobot :class:`Camera` interface
    so it can be used transparently in any robot configuration that
    accepts a ``cameras`` dict.

    Under the hood it:
    1. Opens a serial connection to the XIA-MI tactile controller.
    2. Runs a background thread polling the sensor.
    3. On each ``async_read()``, renders a calibrated 2×2 panel image
       (Fz heatmap + RGB force-map for both fingertip sensors).

    Example
    -------
    >>> from lerobot.cameras.tactile import TactileCamera, TactileCameraConfig
    >>> cfg = TactileCameraConfig(port="/dev/ttyACM0", output_size=256)
    >>> cam = TactileCamera(cfg)
    >>> cam.connect()
    >>> img = cam.async_read()   # (512, 512, 3) uint8 RGB
    >>> cam.disconnect()
    """

    def __init__(self, config: TactileCameraConfig):
        super().__init__(config)
        self.config: TactileCameraConfig = config

        # Derive the effective image shape from output_size.
        sz = config.output_size
        self._image_shape = (sz * 2, sz * 2, 3)  # (H, W, C)

        self._driver: TactileSensorDriver | None = None
        self._runtime: TactileRuntime | None = None
        self._visualizer: TactileVisualizer | None = None

        # Threading (mirrors OpenCVCamera pattern).
        self._read_thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock: Lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._new_frame_event: Event = Event()

    # ------------------------------------------------------------------
    # Camera interface
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._runtime is not None and self._runtime._running

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Tactile sensors are not auto-discoverable; return empty list."""
        return []

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """Open the serial port, start the runtime, and optionally calibrate.

        Parameters
        ----------
        warmup : bool
            If True, wait until at least one valid frame has been captured.
        """
        cfg = self.config

        self._driver = TactileSensorDriver(
            port=cfg.port,
            baudrate=cfg.baudrate,
            timeout=cfg.timeout,
            enable_distributed=cfg.enable_distributed,
            distributed_scale=cfg.distributed_scale,
            logger=logger,
        )

        self._runtime = TactileRuntime(
            driver=self._driver,
            read_mode=cfg.read_mode,
            read_timeout=cfg.timeout,
            poll_interval=cfg.calibration_interval,
            calibration_samples=cfg.calibration_samples,
            calibration_interval=cfg.calibration_interval,
            calibration_warmup_frames=cfg.calibration_warmup_frames,
            calibration_reducer=cfg.calibration_reducer,
            logger=logger,
        )

        self._visualizer = TactileVisualizer(
            output_size=cfg.output_size,
            heatmap_vmin=cfg.heatmap_vmin,
            heatmap_vmax=cfg.heatmap_vmax,
            heatmap_colormap=cfg.heatmap_colormap,
            heatmap_gamma=cfg.heatmap_gamma,
            rgb_vmax_fz=cfg.rgb_vmax_fz,
            rgb_vmax_shear=cfg.rgb_vmax_shear,
        )

        self._runtime.start()

        if warmup:
            frame = self._runtime.wait_for_frame(timeout=3.0)
            if frame is None:
                self._runtime.stop()
                raise ConnectionError(
                    f"TactileCamera({cfg.port}): no frames received during warmup"
                )
            if cfg.calibrate_on_connect:
                self._runtime.calibrate()

        self._start_read_thread()
        logger.info("TactileCamera(%s) connected (image=%s)", cfg.port, self._image_shape)

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        """Synchronous read — delegates to ``async_read``."""
        return self.async_read()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Return the latest rendered frame, blocking if necessary."""
        if self._read_thread is None or not self._read_thread.is_alive():
            raise RuntimeError("TactileCamera read thread is not running.")

        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"TactileCamera({self.config.port}): timed out waiting for frame "
                f"after {timeout_ms} ms."
            )

        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(
                f"TactileCamera({self.config.port}): event set but no frame available."
            )
        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 1000) -> NDArray[Any]:
        """Non-blocking peek at the most recent rendered frame."""
        if self._read_thread is None or not self._read_thread.is_alive():
            raise RuntimeError("TactileCamera read thread is not running.")

        with self._frame_lock:
            frame = self._latest_frame

        if frame is None:
            raise RuntimeError(
                f"TactileCamera({self.config.port}): no frames captured yet."
            )
        return frame

    def disconnect(self) -> None:
        """Stop the runtime and release the serial port."""
        if not self.is_connected and self._read_thread is None:
            raise DeviceNotConnectedError(
                f"TactileCamera({self.config.port}) is not connected."
            )

        # Stop read thread first.
        if self._stop_event is not None:
            self._stop_event.set()
        if self._read_thread is not None and self._read_thread.is_alive():
            self._read_thread.join(timeout=2.0)
        self._read_thread = None
        self._stop_event = None

        # Then stop the runtime.
        if self._runtime is not None:
            self._runtime.stop()
            self._runtime = None

        self._driver = None
        self._visualizer = None

        with self._frame_lock:
            self._latest_frame = None
            self._new_frame_event.clear()

        logger.info("TactileCamera(%s) disconnected.", self.config.port)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _start_read_thread(self) -> None:
        self._stop_event = Event()
        self._read_thread = Thread(
            target=self._read_loop,
            name=f"tactile-camera-{self.config.port}",
            daemon=True,
        )
        self._read_thread.start()
        time.sleep(0.1)

    def _read_loop(self) -> None:
        """Background loop: poll tactile runtime → render → store frame."""
        assert self._runtime is not None
        assert self._visualizer is not None
        assert self._stop_event is not None

        failure_count = 0
        while not self._stop_event.is_set():
            try:
                snapshot = self._runtime.get_snapshot(copy_snapshot=True)
                if snapshot is None:
                    time.sleep(0.01)
                    continue

                # Render calibrated snapshot → BGR image → convert to RGB
                bgr = self._visualizer.render_snapshot(snapshot, calibrated=True)
                rgb = bgr[..., ::-1].copy()  # BGR → RGB

                with self._frame_lock:
                    self._latest_frame = rgb
                self._new_frame_event.set()
                failure_count = 0

            except Exception as exc:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning(
                        "TactileCamera(%s) read loop error: %s",
                        self.config.port,
                        exc,
                    )
                    time.sleep(0.05)
                else:
                    logger.error(
                        "TactileCamera(%s) exceeded max consecutive failures.",
                        self.config.port,
                    )
                    break
