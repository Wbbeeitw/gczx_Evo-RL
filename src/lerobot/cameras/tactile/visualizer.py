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

"""Convert tactile pressure arrays into RGB heatmap / force-map images."""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np
from scipy.interpolate import griddata

from .types import TactileFrame, TactileSnapshot


class TactileVisualizer:
    """Render tactile distributed pressure data as visual heatmap images.

    Each call to :meth:`render_snapshot` or :meth:`render_frame` produces
    a ``(2*output_size, 2*output_size, 3)`` BGR image containing:

    - Row 1: Fz heatmaps for index / middle.
    - Row 2: RGB force-vector maps for index / middle.

    Parameters
    ----------
    output_size : int
        Side length of each sub-panel in pixels (default 256).
    heatmap_vmin : float
        Lower bound of the heatmap range.
    heatmap_vmax : float
        Upper bound of the heatmap range.
    heatmap_colormap : str
        OpenCV colormap name (``turbo``, ``inferno``, ``plasma``, …).
    heatmap_gamma : float
        Gamma correction applied after range normalisation (< 1.0 makes
        low-pressure changes more visible).
    rgb_vmax_fz : float
        Upper bound for the RGB-map Fz channel.
    rgb_vmax_shear : float
        Upper bound for the RGB-map Fx / Fy channels.
    """

    COLORMAPS = {
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "cividis": cv2.COLORMAP_CIVIDIS,
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "bone": cv2.COLORMAP_BONE,
    }

    def __init__(
        self,
        output_size: int = 256,
        heatmap_vmin: float = 0.0,
        heatmap_vmax: float = 25.5,
        heatmap_colormap: str = "turbo",
        heatmap_gamma: float = 0.75,
        rgb_vmax_fz: float = 25.5,
        rgb_vmax_shear: float = 12.8,
    ) -> None:
        self.output_size = output_size
        self.heatmap_vmin = heatmap_vmin
        self.heatmap_vmax = heatmap_vmax
        self.heatmap_colormap = heatmap_colormap.lower()
        self.heatmap_gamma = heatmap_gamma
        self.rgb_vmax_fz = rgb_vmax_fz
        self.rgb_vmax_shear = rgb_vmax_shear

        if self.heatmap_colormap not in self.COLORMAPS:
            supported = ", ".join(sorted(self.COLORMAPS))
            raise ValueError(
                f"Unsupported heatmap colormap '{heatmap_colormap}'. Supported: {supported}"
            )

    @classmethod
    def available_colormaps(cls) -> list[str]:
        return sorted(cls.COLORMAPS)

    # ------------------------------------------------------------------
    # Coordinate generation
    # ------------------------------------------------------------------

    def gen_coords(self, point_count: int) -> np.ndarray:
        """Generate (x, y) normalised coordinates for *point_count* taxels."""
        if point_count == 68:
            coords = []
            rows, cols = 9, 8
            for row in range(rows):
                for col in range(cols):
                    if (row == 0 and col == 0) or (row == 0 and col == cols - 1):
                        continue
                    if (row == rows - 1 and col == 0) or (row == rows - 1 and col == cols - 1):
                        continue
                    coords.append([col / (cols - 1), 1.0 - (row / (rows - 1))])
            return np.asarray(coords, dtype=np.float32)

        if point_count == 25:
            coords = []
            for row in range(5):
                for col in range(5):
                    coords.append([col / 4.0, row / 4.0])
            return np.asarray(coords, dtype=np.float32)

        cols = int(np.ceil(np.sqrt(point_count)))
        rows = int(np.ceil(point_count / max(cols, 1)))
        coords = []
        idx = 0
        for row in range(rows):
            for col in range(cols):
                if idx >= point_count:
                    break
                coords.append([col / max(cols - 1, 1), row / max(rows - 1, 1)])
                idx += 1
        return np.asarray(coords, dtype=np.float32)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate_channel(
        self,
        coords: Optional[np.ndarray],
        data: Optional[np.ndarray],
        channel: int,
    ) -> np.ndarray:
        """Interpolate one channel of distributed data onto the output grid."""
        if data is None or coords is None:
            return np.zeros((self.output_size, self.output_size), dtype=np.float32)

        values = data[:, channel].astype(np.float32)
        grid_x, grid_y = np.mgrid[
            0 : 1 : complex(self.output_size),
            0 : 1 : complex(self.output_size),
        ]
        try:
            grid = griddata(coords, values, (grid_x, grid_y), method="cubic", fill_value=0)
        except Exception:
            grid = griddata(coords, values, (grid_x, grid_y), method="linear", fill_value=0)
        return np.nan_to_num(grid, nan=0.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def normalize_scalar_map(
        self,
        values: np.ndarray,
        vmin: float,
        vmax: float,
        gamma: float = 1.0,
    ) -> np.ndarray:
        dynamic_range = max(vmax - vmin, 1e-6)
        normalized = np.clip((values - vmin) / dynamic_range, 0.0, 1.0)
        normalized = np.power(normalized, max(gamma, 1e-6))
        return (normalized * 255.0).astype(np.uint8)

    def make_fz_heatmap(
        self,
        coords: Optional[np.ndarray],
        data: Optional[np.ndarray],
    ) -> np.ndarray:
        """Create an Fz heatmap for one sensor."""
        if data is None:
            image = np.zeros((self.output_size, self.output_size), dtype=np.uint8)
            return cv2.applyColorMap(image, self.COLORMAPS[self.heatmap_colormap])

        fz = self.interpolate_channel(coords, data, 2)
        fz_norm = self.normalize_scalar_map(
            fz, vmin=self.heatmap_vmin, vmax=self.heatmap_vmax, gamma=self.heatmap_gamma
        )
        return cv2.applyColorMap(fz_norm, self.COLORMAPS[self.heatmap_colormap])

    def make_rgb_map(
        self,
        coords: Optional[np.ndarray],
        data: Optional[np.ndarray],
    ) -> np.ndarray:
        """Create an RGB force-vector map for one sensor."""
        if data is None:
            return np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)

        fx = self.interpolate_channel(coords, data, 0)
        fy = self.interpolate_channel(coords, data, 1)
        fz = self.interpolate_channel(coords, data, 2)

        shear_vmax = max(self.rgb_vmax_shear, 1e-6)
        fz_vmax = max(self.rgb_vmax_fz, 1e-6)

        red = np.clip((fx / shear_vmax * 127.0) + 128.0, 0, 255).astype(np.uint8)
        green = np.clip((fy / shear_vmax * 127.0) + 128.0, 0, 255).astype(np.uint8)
        blue = np.clip((fz / fz_vmax * 255.0), 0, 255).astype(np.uint8)
        return np.stack([blue, green, red], axis=-1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_snapshot(self, snapshot: TactileSnapshot, calibrated: bool = True) -> np.ndarray:
        """Render a calibrated snapshot into a 2×2 panel image."""
        force_override = snapshot.calibrated_force if calibrated else None
        distributed_override = snapshot.calibrated_distributed if calibrated else None
        return self.render_frame(
            snapshot.frame,
            force_override=force_override,
            distributed_override=distributed_override,
        )

    def render_frame(
        self,
        frame: TactileFrame,
        force_override: Optional[Dict[str, np.ndarray]] = None,
        distributed_override: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ) -> np.ndarray:
        """Render a single ``TactileFrame`` into a 2×2 panel image.

        Returns a BGR image of shape ``(2*output_size, 2*output_size, 3)``.
        """
        index_dist = self._resolve_distributed("index_middle", frame, distributed_override)
        middle_dist = self._resolve_distributed("middle_middle", frame, distributed_override)

        index_coords = self.gen_coords(index_dist.shape[0]) if index_dist is not None else None
        middle_coords = self.gen_coords(middle_dist.shape[0]) if middle_dist is not None else None

        index_hm = self.make_fz_heatmap(index_coords, index_dist)
        middle_hm = self.make_fz_heatmap(middle_coords, middle_dist)
        index_rgb = self.make_rgb_map(index_coords, index_dist)
        middle_rgb = self.make_rgb_map(middle_coords, middle_dist)

        row1 = np.hstack([index_hm, middle_hm])
        row2 = np.hstack([index_rgb, middle_rgb])
        canvas = np.vstack([row1, row2])

        ps = self.output_size
        index_force = self._resolve_force("index_middle", frame, force_override)
        middle_force = self._resolve_force("middle_middle", frame, force_override)

        self._overlay_text(canvas, "Index - Fz Heatmap", index_force, 10, 25)
        self._overlay_text(canvas, "Middle - Fz Heatmap", middle_force, ps + 10, 25)
        cv2.putText(
            canvas,
            "Index - RGB (Fx,Fy,Fz)",
            (10, ps + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "Middle - RGB (Fx,Fy,Fz)",
            (ps + 10, ps + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        if frame.error_code is not None:
            cv2.putText(
                canvas,
                f"Err:0x{frame.error_code:02X}",
                (ps * 2 - 140, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        footer = (
            f"Heatmap {self.heatmap_colormap} "
            f"[{self.heatmap_vmin:.2f},{self.heatmap_vmax:.2f}] "
            f"gamma={self.heatmap_gamma:.2f}"
        )
        cv2.putText(
            canvas,
            footer,
            (10, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return canvas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_force(
        self,
        sensor_name: str,
        frame: TactileFrame,
        force_override: Optional[Dict[str, np.ndarray]],
    ) -> Optional[np.ndarray]:
        if force_override is not None and sensor_name in force_override:
            return force_override[sensor_name]
        snapshot = frame.sensors.get(sensor_name)
        return None if snapshot is None else snapshot.force

    def _resolve_distributed(
        self,
        sensor_name: str,
        frame: TactileFrame,
        distributed_override: Optional[Dict[str, Optional[np.ndarray]]],
    ) -> Optional[np.ndarray]:
        if distributed_override is not None and sensor_name in distributed_override:
            return distributed_override[sensor_name]
        snapshot = frame.sensors.get(sensor_name)
        return None if snapshot is None else snapshot.distributed

    def _overlay_text(
        self,
        image: np.ndarray,
        title: str,
        force: Optional[np.ndarray],
        x: int,
        y: int,
    ) -> None:
        cv2.putText(image, title, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if force is None:
            return
        fx, fy, fz = force.tolist()
        text = f"F=({fx:.2f},{fy:.2f},{fz:.2f})"
        cv2.putText(image, text, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
