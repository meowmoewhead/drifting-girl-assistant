import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from frame_runtime import FrameRuntime


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class Stop:
    def __init__(self, clock):
        self.clock, self.stopped, self.waits = clock, False, []
    def is_set(self): return self.stopped
    def wait(self, seconds):
        self.waits.append(seconds)
        self.clock.now += seconds
        return self.stopped


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stop = Stop(self.clock)
        self.reader = SimpleNamespace(begin_frame=Mock(), end_frame=Mock())
        self.captures = []
        def capture():
            self.clock.now += .08
            frame = np.full((8, 8, 3), len(self.captures), np.uint8)
            self.captures.append(frame)
            return frame
        self.device = SimpleNamespace(screenshot_fast=capture, screenshot=Mock(),
                                      screenshot_transport='native-rgba')
        self.runtime = FrameRuntime(self.device, self.stop, lambda:self.reader, self.clock)

    def test_every_guard_gets_new_pixels_and_a_new_ocr_scope(self):
        first = self.runtime.capture_loop()
        second = self.runtime.capture()
        self.assertIsNot(first, second)
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(self.reader.end_frame.call_count, 2)
        self.assertEqual(self.reader.begin_frame.call_count, 2)
        self.device.screenshot.assert_not_called()

    def test_configured_period_includes_capture_and_processing(self):
        self.runtime.capture_loop()
        self.clock.now += .17
        self.runtime.wait(.6)
        self.assertAlmostEqual(self.stop.waits[-1], .35)
        self.assertAlmostEqual(self.clock.now, .6)
        self.assertAlmostEqual(self.runtime.snapshot()['timings']['cycle']['last_ms'], 600.)

    def test_slow_controller_adds_no_extra_poll_delay(self):
        self.runtime.capture_loop()
        self.clock.now += .7
        self.runtime.wait(.6)
        self.assertEqual(self.stop.waits[-1], 0.)
        self.assertAlmostEqual(self.clock.now, .78)

    def test_stop_during_capture_discards_frame_and_never_starts_scope(self):
        def stopped_capture():
            self.stop.stopped = True
            return np.zeros((8, 8, 3), np.uint8)
        self.device.screenshot_fast = stopped_capture
        with self.assertRaises(InterruptedError): self.runtime.capture()
        self.reader.end_frame.assert_called_once()
        self.reader.begin_frame.assert_not_called()

    def test_capture_failure_clears_scope_and_is_not_replayed(self):
        self.device.screenshot_fast = Mock(side_effect=RuntimeError('fresh capture failed'))
        with self.assertRaisesRegex(RuntimeError, 'fresh capture failed'):
            self.runtime.capture()
        self.reader.end_frame.assert_called_once()
        self.reader.begin_frame.assert_not_called()
        self.device.screenshot.assert_not_called()

    def test_devices_without_fast_transport_still_capture_fresh(self):
        self.runtime.device = SimpleNamespace(screenshot=Mock(side_effect=[object(), object()]))
        self.assertIsNot(self.runtime.capture(), self.runtime.capture())

    def test_stopped_wait_is_interruptible_and_close_ends_scope(self):
        stop = threading.Event()
        runtime = FrameRuntime(self.device, stop, lambda:self.reader)
        runtime.capture_loop()
        stop.set()
        self.assertTrue(runtime.wait(60))
        runtime.close()
        self.assertEqual(self.reader.end_frame.call_count, 2)

    def test_telemetry_is_bounded_and_does_not_change_controller_result(self):
        for _ in range(150): self.runtime.record('guard', .012)
        self.assertIs(self.runtime.call('guard', lambda:False), False)
        row = self.runtime.snapshot()['timings']['guard']
        self.assertEqual(row['count'], 151)
        self.assertEqual(row['window'], 128)


if __name__ == '__main__': unittest.main()
