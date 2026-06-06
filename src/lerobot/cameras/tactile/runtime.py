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

"""Background thread that polls the tactile driver and maintains latest frame."""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

import numpy as np

from .driver import TactileSensorDriver
from .types import SENSOR_NAMES, TactileFrame, TactileSnapshot


class TactileRuntime:
    """Runs a background thread that polls the tactile driver and caches the
    latest calibrated frame.

    Parameters
    ----------
    driver : TactileSensorDriver
        Initialized (but not yet opened) driver instance.
    read_mode : str
        ``"auto_push"`` (default) or ``"distributed_poll"``.
    read_timeout : float
        Per-read timeout in seconds.
    poll_interval : float
        Sleep between poll attempts in distributed mode.
    calibration_samples : int
        Number of samples collected during calibration.
    calibration_interval : float
        Wait between calibration samples.
    calibration_warmup_frames : int
        Frames to discard before collecting calibration data.
    calibration_reducer : str
        ``"mean"`` or ``"median"``.
    logger : Optional[logging.Logger]
    """

    def __init__(
        self,
        driver: TactileSensorDriver,
        read_mode: str = "auto_push",
        read_timeout: float = 0.1,
        poll_interval: float = 0.1,
        calibration_samples: int = 50,
        calibration_interval: float = 0.05,
        calibration_warmup_frames: int = 20,
        calibration_reducer: str = "median",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if read_mode not in {"auto_push", "distributed_poll"}:
            raise ValueError(f"Unsupported tactile read mode: {read_mode}")
        if calibration_reducer not in {"mean", "median"}:
            raise ValueError(f"Unsupported calibration reducer: {calibration_reducer}")

        self.driver = driver
        self.read_mode = read_mode
        self.read_timeout = read_timeout
        self.poll_interval = poll_interval
        self.calibration_samples = calibration_samples
        self.calibration_interval = calibration_interval
        self.calibration_warmup_frames = max(int(calibration_warmup_frames), 0)
        self.calibration_reducer = calibration_reducer
        self.logger = logger or logging.getLogger(__name__)

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[TactileFrame] = None
        self._frame_event = threading.Event()
        self.offsets: Dict[str, np.ndarray] = {
            name: np.zeros(3, dtype=np.float32) for name in SENSOR_NAMES
        }
        self.distributed_offsets: Dict[str, Optional[np.ndarray]] = {
            name: None for name in SENSOR_NAMES
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the driver and start the background thread."""
        if self._running:
            return

        if self.read_mode == "auto_push":
            self.driver.initialize()
        else:
            self.driver.open()

        self._running = True
        self._thread = threading.Thread(target=self._loop, name="tactile-runtime", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and close the driver."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self.driver.close()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def wait_for_frame(
        self,
        timeout: float = 3.0,
        after_timestamp: Optional[float] = None,
    ) -> Optional[TactileFrame]:
        """Block until a frame newer than *after_timestamp* is available."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = self.get_latest_frame(copy_frame=True)
            if frame is not None and (after_timestamp is None or frame.timestamp > after_timestamp):
                return frame
            time.sleep(0.01)
        return None

    def get_latest_frame(self, copy_frame: bool = True) -> Optional[TactileFrame]:
        """Return the most recent raw frame (thread-safe)."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy_frame else self._latest_frame

    def get_snapshot(self, copy_snapshot: bool = True) -> Optional[TactileSnapshot]:
        """Return a calibrated snapshot (thread-safe)."""
        frame = self.get_latest_frame(copy_frame=True)
        if frame is None:
            return None

        with self._lock:
            offsets = {name: value.copy() for name, value in self.offsets.items()}
            distributed_offsets = {
                name: None if value is None else value.copy()
                for name, value in self.distributed_offsets.items()
            }

        calibrated_force: Dict[str, np.ndarray] = {}
        calibrated_distributed: Dict[str, Optional[np.ndarray]] = {}
        for name in SENSOR_NAMES:
            calibrated_force[name] = frame.get_force(name) - offsets[name]
            sensor = frame.sensors.get(name)
            if sensor is None or sensor.distributed is None:
                calibrated_distributed[name] = None
                continue
            offset = distributed_offsets.get(name)
            if offset is None or offset.shape != sensor.distributed.shape:
                calibrated_distributed[name] = sensor.distributed.copy()
            else:
                calibrated_distributed[name] = sensor.distributed - offset

        snapshot = TactileSnapshot(
            frame=frame,
            calibrated_force=calibrated_force,
            offsets=offsets,
            calibrated_distributed=calibrated_distributed,
            distributed_offsets=distributed_offsets,
        )
        return snapshot if not copy_snapshot else snapshot.copy()

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        sample_count: Optional[int] = None,
        sample_interval: Optional[float] = None,
        warmup_frames: Optional[int] = None,
        reducer: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Collect baseline readings and compute zero-point offsets.

        Returns the computed force offsets dict.
        """
        if not self._running:
            raise RuntimeError("Tactile runtime must be started before calibration")

        sample_count = sample_count or self.calibration_samples
        sample_interval = (
            sample_interval if sample_interval is not None else self.calibration_interval
        )
        warmup_frames = (
            self.calibration_warmup_frames if warmup_frames is None else max(int(warmup_frames), 0)
        )
        reducer = self.calibration_reducer if reducer is None else reducer
        if reducer not in {"mean", "median"}:
            raise ValueError(f"Unsupported calibration reducer: {reducer}")

        samples = {name: [] for name in SENSOR_NAMES}
        distributed_samples = {name: [] for name in SENSOR_NAMES}
        last_timestamp: Optional[float] = None

        for _ in range(warmup_frames):
            frame = self.wait_for_frame(
                timeout=max(self.read_timeout * 10.0, 1.0),
                after_timestamp=last_timestamp,
            )
            if frame is None:
                raise RuntimeError("Timed out during calibration warmup")
            last_timestamp = frame.timestamp
            time.sleep(sample_interval)

        for _ in range(sample_count):
            frame = self.wait_for_frame(
                timeout=max(self.read_timeout * 10.0, 1.0),
                after_timestamp=last_timestamp,
            )
            if frame is None:
                raise RuntimeError("Timed out during calibration sampling")
            last_timestamp = frame.timestamp
            for name in SENSOR_NAMES:
                sensor = frame.sensors.get(name)
                if sensor is None:
                    continue
                samples[name].append(sensor.force.copy())
                if sensor.distributed is not None:
                    distributed_samples[name].append(sensor.distributed.copy())
            time.sleep(sample_interval)

        offsets: Dict[str, np.ndarray] = {}
        distributed_offsets: Dict[str, Optional[np.ndarray]] = {}
        with self._lock:
            for name in SENSOR_NAMES:
                if samples[name]:
                    self.offsets[name] = self._reduce_calibration_samples(samples[name], reducer)
                else:
                    self.offsets[name] = np.zeros(3, dtype=np.float32)
                offsets[name] = self.offsets[name].copy()

                if distributed_samples[name]:
                    self.distributed_offsets[name] = self._reduce_calibration_samples(
                        distributed_samples[name], reducer
                    )
                else:
                    self.distributed_offsets[name] = None
                distributed_offsets[name] = (
                    None
                    if self.distributed_offsets[name] is None
                    else self.distributed_offsets[name].copy()
                )

        self.logger.info(
            "Calibrated tactile offsets with %s reducer: force=%s",
            reducer,
            {k: v.tolist() for k, v in offsets.items()},
        )
        return offsets

    def reset_calibration(self) -> None:
        """Reset all offsets to zero."""
        with self._lock:
            for name in SENSOR_NAMES:
                self.offsets[name] = np.zeros(3, dtype=np.float32)
                self.distributed_offsets[name] = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_snapshot_npz(self, path: str, snapshot: Optional[TactileSnapshot] = None) -> str:
        """Save a snapshot as a compressed .npz file."""
        snapshot = snapshot or self.get_snapshot(copy_snapshot=True)
        if snapshot is None:
            raise RuntimeError("No tactile snapshot available to save")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload: Dict[str, np.ndarray] = {}
        for name in SENSOR_NAMES:
            payload[f"{name}_offset"] = snapshot.offsets[name]
            payload[f"{name}_calibrated_force"] = snapshot.calibrated_force[name]
            distributed_offset = snapshot.distributed_offsets.get(name)
            if distributed_offset is not None:
                payload[f"{name}_distributed_offset"] = distributed_offset
            calibrated_distributed = snapshot.calibrated_distributed.get(name)
            if calibrated_distributed is not None:
                payload[f"{name}_calibrated_distributed"] = calibrated_distributed
            sensor = snapshot.frame.sensors.get(name)
            if sensor is None:
                continue
            payload[f"{name}_force"] = sensor.force
            if sensor.distributed is not None:
                payload[f"{name}_distributed"] = sensor.distributed

        np.savez_compressed(path, timestamp=snapshot.frame.timestamp, **payload)
        return path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                if self.read_mode == "auto_push":
                    frame = self.driver.read_frame(timeout=self.read_timeout)
                else:
                    frame = None  # distributed poll not used in camera mode
            except Exception as exc:
                self.logger.warning("Tactile runtime read failed: %s", exc)
                time.sleep(max(self.poll_interval, 0.1))
                continue

            if frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._frame_event.set()

            if self.read_mode == "distributed_poll":
                time.sleep(self.poll_interval)

    @staticmethod
    def _reduce_calibration_samples(samples: list[np.ndarray], reducer: str) -> np.ndarray:
        stacked = np.stack(samples, axis=0)
        if reducer == "median":
            return np.median(stacked, axis=0).astype(np.float32)
        return np.mean(stacked, axis=0).astype(np.float32)
