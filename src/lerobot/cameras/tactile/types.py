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

"""Data types for tactile sensor frames and snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

SENSOR_NAMES = ("index_middle", "middle_middle")


def _zero_force() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


@dataclass
class SensorSnapshot:
    """Raw sensor data for one tactile sensing element."""

    force: np.ndarray = field(default_factory=_zero_force)
    distributed: Optional[np.ndarray] = None
    point_count: int = 0

    def copy(self) -> "SensorSnapshot":
        return SensorSnapshot(
            force=self.force.copy(),
            distributed=None if self.distributed is None else self.distributed.copy(),
            point_count=self.point_count,
        )


@dataclass
class TactileFrame:
    """A single frame of tactile data from the sensor controller."""

    timestamp: float
    error_code: Optional[int]
    raw_payload_hex: Optional[str]
    sensors: Dict[str, SensorSnapshot]

    def copy(self) -> "TactileFrame":
        return TactileFrame(
            timestamp=self.timestamp,
            error_code=self.error_code,
            raw_payload_hex=self.raw_payload_hex,
            sensors={name: snapshot.copy() for name, snapshot in self.sensors.items()},
        )

    def get_force(self, sensor_name: str) -> np.ndarray:
        """Return a copy of the force vector for a given sensor."""
        snapshot = self.sensors.get(sensor_name)
        if snapshot is None:
            return _zero_force()
        return snapshot.force.copy()


@dataclass
class TactileSnapshot:
    """A calibrated tactile snapshot with offsets applied."""

    frame: TactileFrame
    calibrated_force: Dict[str, np.ndarray]
    offsets: Dict[str, np.ndarray]
    calibrated_distributed: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    distributed_offsets: Dict[str, Optional[np.ndarray]] = field(default_factory=dict)

    def copy(self) -> "TactileSnapshot":
        return TactileSnapshot(
            frame=self.frame.copy(),
            calibrated_force={name: value.copy() for name, value in self.calibrated_force.items()},
            offsets={name: value.copy() for name, value in self.offsets.items()},
            calibrated_distributed={
                name: None if value is None else value.copy()
                for name, value in self.calibrated_distributed.items()
            },
            distributed_offsets={
                name: None if value is None else value.copy()
                for name, value in self.distributed_offsets.items()
            },
        )
