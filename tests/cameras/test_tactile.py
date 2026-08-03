import time
from types import SimpleNamespace

import numpy as np
import pytest

import lerobot.cameras.tactile.camera_tactile as tactile_camera_module
from lerobot.cameras.tactile import (
    TactileCamera,
    TactileCameraConfig,
    TactileDropoutError,
)
from lerobot.cameras.tactile.driver import TactileSensorDriver
from lerobot.cameras.tactile.protocol import TactileSerialProtocol
from lerobot.cameras.tactile.runtime import TactileRuntime


class _AliveThread:
    @staticmethod
    def is_alive() -> bool:
        return True


class _NoFrameEvent:
    @staticmethod
    def clear() -> None:
        pass

    @staticmethod
    def wait(timeout: float) -> bool:
        del timeout
        return False


def _connected_camera() -> TactileCamera:
    camera = TactileCamera(
        TactileCameraConfig(port="unused", calibrate_on_connect=False)
    )
    camera._runtime = SimpleNamespace(_running=True)
    camera._read_thread = _AliveThread()
    return camera


def test_read_latest_returns_fresh_frame(monkeypatch) -> None:
    camera = _connected_camera()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    camera._latest_frame = frame
    camera._latest_timestamp = 10.0
    monkeypatch.setattr(tactile_camera_module.time, "perf_counter", lambda: 10.25)

    assert camera.read_latest(max_age_ms=500) is frame


def test_connected_camera_can_be_recalibrated() -> None:
    camera = _connected_camera()
    expected_offsets = {"index_middle": np.array([0.1, 0.2, 0.3], dtype=np.float32)}
    calls = []

    def calibrate(**kwargs):
        calls.append(kwargs)
        return expected_offsets

    camera._runtime.calibrate = calibrate

    offsets = camera.calibrate(
        sample_count=10,
        sample_interval=0.01,
        warmup_frames=5,
        reducer="median",
    )

    assert offsets is expected_offsets
    assert calls == [
        {
            "sample_count": 10,
            "sample_interval": 0.01,
            "warmup_frames": 5,
            "reducer": "median",
        }
    ]


def test_read_latest_rejects_stale_frame(monkeypatch) -> None:
    camera = _connected_camera()
    camera._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    camera._latest_timestamp = 10.0
    monkeypatch.setattr(tactile_camera_module.time, "perf_counter", lambda: 11.5)

    with pytest.raises(TimeoutError, match="latest frame is too old"):
        camera.read_latest(max_age_ms=1000)


def test_async_read_rejects_stale_cached_frame(monkeypatch) -> None:
    camera = _connected_camera()
    camera._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    camera._latest_timestamp = 10.0
    camera._new_frame_event = _NoFrameEvent()
    monkeypatch.setattr(tactile_camera_module.time, "perf_counter", lambda: 12.0)

    with pytest.raises(TimeoutError, match="fresh frame"):
        camera.async_read(timeout_ms=1000)


def test_async_read_holds_last_frame_during_short_reconnect() -> None:
    camera = _connected_camera()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    camera._latest_frame = frame
    camera._runtime.is_recovering = True
    camera._runtime.dropout_duration_s = 0.25

    assert camera.async_read(timeout_ms=1000) is frame


def test_async_read_rejects_reconnect_beyond_hold_limit() -> None:
    camera = _connected_camera()
    camera._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    camera._runtime.is_recovering = True
    camera._runtime.dropout_duration_s = 0.75

    with pytest.raises(TactileDropoutError, match="exceeding hold-last limit"):
        camera.async_read(timeout_ms=1000)


def test_read_latest_requires_a_capture_timestamp() -> None:
    camera = _connected_camera()
    camera._latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError, match="no frames captured"):
        camera.read_latest()


def test_connect_cleans_up_runtime_when_calibration_fails(monkeypatch) -> None:
    class FakeDriver:
        def __init__(self, **kwargs):
            del kwargs

    class FakeRuntime:
        instance = None

        def __init__(self, **kwargs):
            del kwargs
            self._running = False
            self.stop_calls = 0
            FakeRuntime.instance = self

        def start(self):
            self._running = True

        def wait_for_frame(self, timeout: float):
            del timeout
            return object()

        def calibrate(self):
            raise RuntimeError("calibration failed")

        def stop(self):
            self.stop_calls += 1
            self._running = False

    monkeypatch.setattr(tactile_camera_module, "TactileSensorDriver", FakeDriver)
    monkeypatch.setattr(tactile_camera_module, "TactileRuntime", FakeRuntime)
    monkeypatch.setattr(
        tactile_camera_module, "TactileVisualizer", lambda **kwargs: object()
    )
    camera = TactileCamera(
        TactileCameraConfig(port="unused", calibrate_on_connect=True)
    )

    with pytest.raises(RuntimeError, match="calibration failed"):
        camera.connect()

    assert FakeRuntime.instance.stop_calls == 1
    assert camera._runtime is None
    assert camera._driver is None
    assert not camera.is_connected


def test_serial_read_errors_are_not_treated_as_timeouts() -> None:
    class BrokenSerial:
        is_open = True

        @property
        def in_waiting(self):
            raise OSError("device removed")

    protocol = TactileSerialProtocol(port="unused")
    protocol.ser = BrokenSerial()

    with pytest.raises(ConnectionError, match="device removed"):
        protocol.read_response(timeout=0.1, expected_head=protocol.RESP_HEAD_AUTO_PUSH)


def test_runtime_stops_after_terminal_serial_error() -> None:
    class BrokenDriver:
        def __init__(self) -> None:
            self.close_calls = 0

        def initialize(self) -> None:
            pass

        def read_frame(self, timeout: float):
            del timeout
            raise OSError("device removed")

        def close(self) -> None:
            self.close_calls += 1

    driver = BrokenDriver()
    runtime = TactileRuntime(driver=driver, read_timeout=0.01)
    runtime.start()
    runtime._thread.join(timeout=1.0)

    assert not runtime._running
    assert isinstance(runtime._last_error, OSError)
    runtime.stop()
    assert driver.close_calls == 1


def test_runtime_reconnects_and_preserves_calibration_offsets() -> None:
    class RecoveringDriver:
        def __init__(self) -> None:
            self.initialize_calls = 0
            self.force_close_calls = 0
            self.close_calls = 0
            self.read_calls = 0

        def initialize(self) -> None:
            self.initialize_calls += 1

        def read_frame(self, timeout: float):
            del timeout
            self.read_calls += 1
            if self.read_calls == 1:
                raise OSError("device removed")
            if self.read_calls == 2:
                return SimpleNamespace(timestamp=1.0, copy=lambda: None)
            return None

        def force_close(self) -> None:
            self.force_close_calls += 1

        def close(self) -> None:
            self.close_calls += 1

    driver = RecoveringDriver()
    runtime = TactileRuntime(
        driver=driver,
        read_timeout=0.01,
        reconnect_on_error=True,
        reconnect_interval=0.001,
    )
    expected_offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    runtime.offsets["index_middle"] = expected_offset.copy()

    runtime.start()
    assert runtime.wait_until_recovered(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while runtime.reconnect_count == 0 and time.monotonic() < deadline:
        time.sleep(0.001)

    assert runtime.reconnect_count == 1
    assert driver.initialize_calls == 2
    assert driver.force_close_calls == 1
    np.testing.assert_array_equal(runtime.offsets["index_middle"], expected_offset)
    runtime.stop()


def test_driver_force_close_skips_graceful_disable() -> None:
    class FakeSerial:
        is_open = True

        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            self.is_open = False

    driver = TactileSensorDriver(port="unused")
    serial_port = FakeSerial()
    driver.ser = serial_port
    driver.auto_push_mode = True
    driver.disable_auto_push_mode = lambda: pytest.fail(
        "graceful disable must be skipped"
    )

    driver.force_close()

    assert serial_port.close_calls == 1
    assert driver.ser is None
    assert not driver.auto_push_mode


def test_runtime_force_closes_driver_when_read_thread_stays_blocked() -> None:
    class StuckDriver:
        def __init__(self) -> None:
            self.force_close_calls = 0
            self.close_calls = 0
            self.force_closed = False

        def force_close(self) -> None:
            self.force_close_calls += 1
            self.force_closed = True

        def close(self) -> None:
            self.close_calls += 1

    class StuckThread:
        def __init__(self, driver: StuckDriver) -> None:
            self.driver = driver
            self.join_calls = 0

        def is_alive(self) -> bool:
            return not self.driver.force_closed

        def join(self, timeout: float) -> None:
            assert timeout == 2.0
            self.join_calls += 1

    driver = StuckDriver()
    thread = StuckThread(driver)
    runtime = TactileRuntime(driver=driver)
    runtime._running = True
    runtime._thread = thread

    runtime.stop()

    assert driver.force_close_calls == 1
    assert driver.close_calls == 0
    assert thread.join_calls == 2
    assert runtime._thread is None


def test_runtime_preserves_force_close_error_when_thread_cannot_stop() -> None:
    class FailingDriver:
        def force_close(self) -> None:
            raise OSError("force close failed")

        def close(self) -> None:
            pytest.fail("graceful close must be skipped")

    class StuckThread:
        @staticmethod
        def is_alive() -> bool:
            return True

        @staticmethod
        def join(timeout: float) -> None:
            assert timeout == 2.0

    thread = StuckThread()
    runtime = TactileRuntime(driver=FailingDriver())
    runtime._running = True
    runtime._thread = thread

    with pytest.raises(RuntimeError, match="thread did not stop") as exc_info:
        runtime.stop()

    assert isinstance(exc_info.value.__cause__, OSError)
    assert runtime._thread is thread


def test_disconnect_can_retry_after_runtime_stop_failure() -> None:
    class FlakyRuntime:
        _running = False

        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1
            if self.stop_calls == 1:
                raise OSError("close failed")

    camera = TactileCamera(
        TactileCameraConfig(port="unused", calibrate_on_connect=False)
    )
    runtime = FlakyRuntime()
    camera._runtime = runtime
    camera._driver = object()
    camera._visualizer = object()

    with pytest.raises(RuntimeError, match="failed to stop cleanly"):
        camera.disconnect()

    assert camera._runtime is runtime
    camera.disconnect()
    assert camera._runtime is None
    assert camera._driver is None
