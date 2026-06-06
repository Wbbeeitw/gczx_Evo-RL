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

"""Low-level serial protocol for tactile sensor communication."""

from __future__ import annotations

import logging
import struct
import time
from typing import Optional, Sequence

import serial


class TactileSerialProtocol:
    """RS-485 serial protocol layer for tactile sensor controllers.

    Handles frame construction, CRC validation, and send/receive over a
    ``serial.Serial`` port.  Subclass this and add sensor-specific parsing.
    """

    # --- Protocol constants ---
    HEADER_MAGIC = b"\x5A\xA5"
    RESP_HEAD_GENERAL = b"\x5A\xA5"
    RESP_HEAD_AUTO_PUSH = b"\xA5\x5A"

    FUNC_READ = 0x01
    FUNC_WRITE = 0x02

    DATA_TYPE_REG = 0x0030
    AUTO_PUSH_REG = 0x0031

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        timeout: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)
        self._serial: Optional[serial.Serial] = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the serial port."""
        if self._serial is not None and self._serial.is_open:
            return
        self._serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )
        self.logger.info("Opened tactile serial port %s @ %d baud", self.port, self.baudrate)

    def close(self) -> None:
        """Close the serial port."""
        if self._serial is not None and self._serial.is_open:
            self._serial.close()
            self.logger.info("Closed tactile serial port %s", self.port)
        self._serial = None

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    @staticmethod
    def compute_crc(data: bytes) -> int:
        """Compute 16-bit Modbus-style CRC over *data*."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    def build_request_frame(
        self,
        func_code: int,
        reg_addr: int,
        length: int,
        payload: bytes = b"",
    ) -> bytes:
        """Build a request frame: header + func + addr(2) + len(2) + payload + crc(2)."""
        if length > 0xFFFF:
            raise ValueError(f"Request length overflow: {length}")

        header_and_func = struct.pack("<4sB", self.HEADER_MAGIC, func_code)
        addr_bytes = struct.pack("<H", reg_addr & 0xFFFF)
        len_bytes = struct.pack("<H", length & 0xFFFF)

        crc_input = header_and_func + addr_bytes + len_bytes + payload
        crc = self.compute_crc(crc_input)
        return crc_input + struct.pack("<H", crc)

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------

    def send_command(self, frame: bytes) -> bool:
        """Send a raw command frame over the serial port."""
        if not self.is_open:
            self.logger.error("Serial port %s is not open", self.port)
            return False
        try:
            self._serial.write(frame)  # type: ignore[union-attr]
            return True
        except serial.SerialException as exc:
            self.logger.error("Failed to send tactile command: %s", exc)
            return False

    def read_response(
        self,
        timeout: float,
        expected_header: bytes,
    ) -> Optional[bytes]:
        """Read a response from the serial port, matching *expected_header*.

        Returns the raw bytes on success or *None* on timeout / mismatch.
        """
        if not self.is_open:
            return None

        header_len = len(expected_header)
        original_timeout = self._serial.timeout  # type: ignore[union-attr]

        try:
            self._serial.timeout = timeout  # type: ignore[union-attr]
            # Wait for header
            buffer = self._serial.read(header_len)  # type: ignore[union-attr]
            if len(buffer) < header_len:
                return None
            if buffer != expected_header:
                # Discard leading noise
                self._serial.reset_input_buffer()  # type: ignore[union-attr]
                return None

            # Read addr (2) + length (2)
            meta = self._serial.read(4)  # type: ignore[union-attr]
            if len(meta) < 4:
                return None

            payload_len = struct.unpack("<H", meta[2:4])[0]
            if payload_len > 4096:
                self.logger.warning("Tactile response payload too large: %d", payload_len)
                return None

            payload = self._serial.read(payload_len + 1)  # type: ignore[union-attr]  # +1 for the data byte after length
            crc_bytes = self._serial.read(2)  # type: ignore[union-attr]
            if len(payload) < payload_len + 1 or len(crc_bytes) < 2:
                return None

            return expected_header + meta + payload + crc_bytes
        except serial.SerialException as exc:
            self.logger.warning("Tactile serial read error: %s", exc)
            return None
        finally:
            self._serial.timeout = original_timeout  # type: ignore[union-attr]

    def flush_input(self) -> None:
        """Discard any pending input bytes."""
        if self.is_open:
            self._serial.reset_input_buffer()  # type: ignore[union-attr]
