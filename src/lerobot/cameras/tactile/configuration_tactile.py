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

"""Configuration dataclass for TactileCamera."""

from dataclasses import dataclass, field

from ..configs import CameraConfig


@CameraConfig.register_subclass("tactile")
@dataclass
class TactileCameraConfig(CameraConfig):
    """Configuration for a tactile-sensor camera.

    A tactile camera reads pressure data from an XIA-MI tactile sensor
    controller over RS-485 serial and renders it as a 2×2 heatmap image.

    Example usage inside a robot config::

        cameras:
          tactile_left:
            type: tactile
            port: /dev/ttyACM0
            output_size: 256
            heatmap_colormap: turbo
            calibrate_on_connect: true

    Attributes
    ----------
    port : str
        Serial device path, e.g. ``/dev/ttyACM0``.
    baudrate : int
        Serial baud rate (default 921600).
    timeout : float
        Serial read timeout in seconds.
    read_mode : str
        ``"auto_push"`` (default) — sensor streams data automatically.
    enable_distributed : bool
        Whether to read per-taxel pressure distributions.
    distributed_scale : float
        Scaling factor for distributed pressure bytes.
    output_size : int
        Side length of each rendered sub-panel in pixels.  The final
        image will be ``(2*output_size, 2*output_size, 3)``.
    heatmap_vmin : float
        Lower bound of the Fz heatmap range.
    heatmap_vmax : float
        Upper bound of the Fz heatmap range.
    heatmap_colormap : str
        OpenCV colormap name (``turbo``, ``inferno``, ``plasma``, …).
    heatmap_gamma : float
        Gamma correction applied after range normalisation.
    rgb_vmax_fz : float
        Upper bound for the RGB-map Fz channel.
    rgb_vmax_shear : float
        Upper bound for the RGB-map Fx/Fy channels.
    calibrate_on_connect : bool
        Run zero-point calibration automatically on connect.
    calibration_samples : int
        Number of samples collected during calibration.
    calibration_interval : float
        Wait between calibration samples.
    calibration_warmup_frames : int
        Frames to discard before collecting calibration data.
    calibration_reducer : str
        ``"mean"`` or ``"median"``.
    """

    # --- Serial ---
    port: str
    baudrate: int = 921600
    timeout: float = 1.0
    read_mode: str = "auto_push"
    enable_distributed: bool = True
    distributed_scale: float = 0.1

    # --- Visualisation ---
    output_size: int = 64
    heatmap_vmin: float = 0.0
    heatmap_vmax: float = 25.5
    heatmap_colormap: str = "turbo"
    heatmap_gamma: float = 0.75
    rgb_vmax_fz: float = 25.5
    rgb_vmax_shear: float = 12.8

    # --- Calibration ---
    calibrate_on_connect: bool = True
    calibration_samples: int = 50
    calibration_interval: float = 0.05
    calibration_warmup_frames: int = 20
    calibration_reducer: str = "median"

    # CameraConfig overrides — tactile camera always produces images at
    # (2*output_size, 2*output_size, 3).  These are set in __post_init__.
    fps: int | None = field(default=10, init=False, repr=False)
    width: int | None = field(default=512, init=False, repr=False)
    height: int | None = field(default=512, init=False, repr=False)

    def __post_init__(self) -> None:
        # Force camera properties to match the rendered image.
        sz = self.output_size * 2
        object.__setattr__(self, "fps", 10)
        object.__setattr__(self, "width", sz)
        object.__setattr__(self, "height", sz)

        if self.read_mode not in {"auto_push", "distributed_poll"}:
            raise ValueError(f"Unsupported read_mode: {self.read_mode}")
        if self.calibration_reducer not in {"mean", "median"}:
            raise ValueError(f"Unsupported calibration_reducer: {self.calibration_reducer}")
