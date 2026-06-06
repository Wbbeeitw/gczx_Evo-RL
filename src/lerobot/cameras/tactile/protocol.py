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

"""Low-level serial protocol for tactile sensor communication (Paxini adapter)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import serial


class TactileSerialProtocol:
    """Serial protocol layer for Paxini tactile sensor controllers.

    Frame format::

        REQ_HEAD(2) + RESERVED(1) + FUNC(1) + ADDR(2,LE) + LEN(2,LE) + DATA + LRC(1)

    Uses LRC (longitudinal redundancy check) for error detection.
    """

    REQ_HEAD = b"\x55\xAA"
    RESP_HEAD_GENERAL = b"\xAA\x55"
    RESP_HEAD_AUTO_PUSH = b"\xAA\x56"
    RESERVED = b"\x00"

    FUNC_READ = 0x03
    FUNC_WRITE = 0x10

    DATA_TYPE_REG = 0x0016
    AUTO_PUSH_REG = 0x0017

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        timeout: float = 1.0,
        write_timeout: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.logger = logger or logging.getLogger(__name__)
        self.serial_lock = threading.Lock()
        self.ser: Optional[serial.Serial] = None

    @property
    def is_open(self) -> bool:
        return self.ser is not None and self.ser.is_open

    def open(self) -> None:
        """Open the serial port."""
        if self.is_open:
            return
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )
        if not self.ser.is_open:
            self.ser.open()
        time.sleep(0.2)
        self.logger.info("Connected tactile serial port %s @ %s", self.port, self.baudrate)

    def close(self) -> None:
        """Close the serial port."""
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        self.ser = None

    # ------------------------------------------------------------------
    # LRC checksum
    # ------------------------------------------------------------------

    @staticmethod
    def calc_lrc(data: bytes) -> int:
        """Compute Longitudinal Redundancy Check (LRC-8)."""
        total = 0
        for value in data:
            total = (total + value) & 0xFF
        return ((~total) + 1) & 0xFF

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------

    def build_request_frame(
        self,
        func_code: int,
        reg_addr: int,
        data_len: int,
        write_data: bytes = b"",
    ) -> bytes:
        """Build a request frame.

        Format: REQ_HEAD + RESERVED + FUNC(1) + ADDR(2,LE) + LEN(2,LE) + DATA + LRC(1)
        """
        reg_addr_bytes = reg_addr.to_bytes(2, "little")
        data_len_bytes = data_len.to_bytes(2, "little")
        frame_wo_lrc = (
            self.REQ_HEAD
            + self.RESERVED
            + func_code.to_bytes(1, "big")
            + reg_addr_bytes
            + data_len_bytes
            + write_data
        )
        lrc = self.calc_lrc(frame_wo_lrc).to_bytes(1, "big")
        return frame_wo_lrc + lrc

    # ------------------------------------------------------------------
    # Send / receive
    # ------------------------------------------------------------------

    def send_command(self, frame: bytes) -> bool:
        """Send a raw command frame over the serial port."""
        if not self.is_open:
            raise RuntimeError("Serial port is not open")
        try:
            with self.serial_lock:
                assert self.ser is not None
                self.ser.flushInput()
                self.ser.flushOutput()
                self.ser.write(frame)
            return True
        except Exception as exc:
            self.logger.warning("Failed to send tactile command: %s", exc)
            return False

    def read_response(
        self,
        timeout: float,
        expected_head: Optional[bytes],
    ) -> Optional[bytes]:
        """Read and accumulate bytes until the expected header and frame length are matched.

        Returns the complete frame bytes, or None on timeout.
        """
        if not self.is_open:
            raise RuntimeError("Serial port is not open")
        try:
            start = time.time()
            data = b""
            with self.serial_lock:
                assert self.ser is not None
                while time.time() - start < timeout:
                    if self.ser.in_waiting > 0:
                        chunk = self.ser.read(self.ser.in_waiting)
                        data += chunk
                        if expected_head and expected_head in data:
                            pos = data.find(expected_head)
                            data = data[pos:]

                            expected_len = None
                            if expected_head == self.RESP_HEAD_AUTO_PUSH and len(data) >= 5:
                                valid_frame_len = int.from_bytes(data[3:5], "little")
                                expected_len = 6 + valid_frame_len
                            elif expected_head == self.RESP_HEAD_GENERAL and len(data) >= 8:
                                payload_len = int.from_bytes(data[6:8], "little")
                                expected_len = 8 + payload_len + 1

                            if expected_len is not None and len(data) >= expected_len:
                                data = data[:expected_len]
                                break
                            start = time.time()  # only reset when header found
                    time.sleep(0.001)
            return data if data else None
        except Exception as exc:
            self.logger.warning("Failed to read tactile response: %s", exc)
            return None

    def flush_input(self) -> None:
        """Discard any pending input bytes."""
        if self.is_open:
            with self.serial_lock:
                assert self.ser is not None
                self.ser.flushInput()
