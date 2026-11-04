import json
import struct
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import AdbDevice, Clicker, CroppedTemplateButton, png_bytes
from alas_core.base import ModuleBase
from digit_ocr import parse_counter


class FakeDevice:
    source = None

    def __init__(self, config, stop):
        self.config = config
        self.stop = stop
        self.image = None
        self.taps = []

    def screenshot(self):
        self.image = self.source.copy()
        return self.image

    def click(self, button):
        if self.stop.is_set():
            raise InterruptedError()
        self.taps.append(tuple(int(n) for n in button.button))

    def stuck_record_add(self, button):
        pass


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        config = {"adb_path": "fake", "serial": "test", "resolution": [64, 96],
                  "poll_interval": 0.1, "max_consecutive_clicks": 2, "rules": []}
        (self.root / "config.json").write_text(json.dumps(config))
        FakeDevice.source = np.random.default_rng(143).integers(0, 256, (96, 64, 3), dtype=np.uint8)
        self.app = Clicker(self.root, FakeDevice)
        self.frame = self.app.capture()

    def tearDown(self):
        self.app.stop()
        if self.app.thread:
            self.app.thread.join(3)
        for handler in list(self.app.log.handlers):
            handler.close()
            self.app.log.removeHandler(handler)
        self.temp.cleanup()

    def rule(self):
        return self.app.add_rule({"frame_id": self.frame["id"], "area": [10, 20, 26, 38],
                                  "name": "test", "threshold": 0.95, "interval": 0.2,
                                  "stable_frames": 2})

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("Runner did not reach expected state")

    def test_upstream_button_finds_translated_target(self):
        rule, button, offset = self.app.load_rule(self.rule())
        moved = np.zeros_like(FakeDevice.source)
        moved[52:70, 34:50] = FakeDevice.source[20:38, 10:26]
        self.assertTrue(button.match(moved, offset=offset, similarity=0.95))
        self.assertEqual(tuple(button.button), (34, 52, 50, 70))
        self.assertGreater(button.last_similarity, 0.99)

    def test_target_outside_search_area_is_not_matched(self):
        rule = self.rule()
        rule["search_area"] = [0, 0, 30, 40]
        _, button, offset = self.app.load_rule(rule)
        moved = np.zeros_like(FakeDevice.source)
        moved[52:70, 34:50] = FakeDevice.source[20:38, 10:26]
        self.assertFalse(button.match(moved, offset=offset, similarity=0.95))

    def test_upstream_click_cooldown(self):
        _, button, offset = self.app.load_rule(self.rule())
        module = ModuleBase(SimpleNamespace(), self.app.device)
        self.assertTrue(module.appear_then_click(button, offset=offset, interval=60, similarity=0.95))
        self.assertFalse(module.appear_then_click(button, offset=offset, interval=60, similarity=0.95))
        self.assertEqual(len(self.app.device.taps), 1)

    def test_preview_never_sends_clicks_and_stops(self):
        self.rule()
        self.app.start(preview=True)
        self.wait_for(lambda: self.app.status()["matches"] >= 1)
        self.app.stop()
        self.app.thread.join(3)
        self.assertFalse(self.app.running())
        self.assertEqual(self.app.device.taps, [])
        self.assertEqual(self.app.status()["clicks"], 0)

    def test_repeated_click_limit_stops_runner(self):
        self.rule()
        self.app.start(preview=False)
        self.wait_for(lambda: not self.app.running())
        self.assertEqual(len(self.app.device.taps), 2)
        self.assertIn("连续点击", self.app.status()["last_error"])
        self.assertEqual(len(list((self.root / "logs").glob("error-*.png"))), 1)

    def test_blank_template_and_empty_click_run_are_rejected(self):
        identifier = self.app.remember(np.zeros((96, 64, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            self.app.add_rule({"frame_id": identifier, "area": [0, 0, 20, 20], "name": "blank"})
        with self.assertRaises(ValueError):
            self.app.start(preview=False)

    def test_resolution_change_prevents_control(self):
        device = AdbDevice(self.app.config)
        with patch.object(device, 'command', return_value=png_bytes(np.zeros((48, 32, 3), dtype=np.uint8))) as command:
            with self.assertRaisesRegex(RuntimeError, "尺寸"):
                device.screenshot()
        self.assertEqual(command.call_count, 1)
        self.assertIsNone(device.image)

    def test_raw_boss_capture_preserves_rgb_with_both_native_headers(self):
        rgba=np.concatenate((FakeDevice.source,np.full((96,64,1),231,dtype=np.uint8)),axis=2)
        for header in (struct.pack('<III',64,96,1),struct.pack('<IIII',64,96,2,1)):
            with self.subTest(header=len(header)):
                device=AdbDevice(self.app.config)
                with patch.object(device,'command',return_value=header+rgba.tobytes()) as command:
                    np.testing.assert_array_equal(device.screenshot_fast(),FakeDevice.source)
                command.assert_called_once_with('exec-out','screencap')

    def test_raw_boss_capture_invalid_buffer_falls_back_to_a_new_png(self):
        for raw in (b'',b'broken',struct.pack('<IIII',64,96,1,1)+b'x',
                    struct.pack('<III',64,96,4)+b'x'):
            with self.subTest(size=len(raw)):
                device=AdbDevice(self.app.config)
                device.image=np.zeros_like(FakeDevice.source)
                with patch.object(device,'command',side_effect=[raw,png_bytes(FakeDevice.source)]) as command:
                    np.testing.assert_array_equal(device.screenshot_fast(),FakeDevice.source)
                self.assertEqual([c.args for c in command.call_args_list],
                    [('exec-out','screencap'),('exec-out','screencap','-p')])

    def test_raw_boss_capture_wrong_resolution_or_stop_never_returns_stale_pixels(self):
        device=AdbDevice(self.app.config)
        previous=FakeDevice.source.copy();device.image=previous
        with patch.object(device,'command',return_value=struct.pack('<IIII',32,48,1,1)+bytes(32*48*4)) as command:
            with self.assertRaisesRegex(RuntimeError,'尺寸'):device.screenshot_fast()
        self.assertEqual(command.call_count,1)
        self.assertIs(device.image,previous)
        def stopped_capture(*_):
            device.stop.set()
            return b'broken'
        with patch.object(device,'command',side_effect=stopped_capture) as command:
            with self.assertRaises(InterruptedError):device.screenshot_fast()
        self.assertEqual(command.call_count,1)
        self.assertIs(device.image,previous)

    def test_state_loop_uses_selected_fresh_capture_transport(self):
        next_frame=np.full_like(FakeDevice.source,173)
        module=ModuleBase(SimpleNamespace(),self.app.device)
        with patch.object(self.app.device,'screenshot',side_effect=AssertionError('Wrong transport')):
            loop=module.loop(skip_first=False,capture=lambda:next_frame)
            np.testing.assert_array_equal(next(loop),next_frame)
            self.assertIs(self.app.device.image,next_frame)
            loop.close()

    def test_transient_empty_and_corrupt_screenshots_retry_without_input(self):
        device = AdbDevice(self.app.config)
        with patch.object(device, 'command', side_effect=[b'', b'broken png', png_bytes(FakeDevice.source)]) as command:
            np.testing.assert_array_equal(device.screenshot(), FakeDevice.source)
        self.assertEqual(command.call_count, 3)
        self.assertTrue(all(call.args == ('exec-out', 'screencap', '-p') for call in command.call_args_list))

    def test_disconnected_screenshot_stops_without_reusing_last_image(self):
        device = AdbDevice(self.app.config)
        previous = FakeDevice.source.copy()
        device.image = previous
        with patch.object(device, 'command', return_value=b'') as command:
            with self.assertRaisesRegex(RuntimeError, '连续.*截图.*模拟器'):
                device.screenshot()
        self.assertEqual(command.call_count, 3)
        self.assertIs(device.image, previous)

    def test_stop_during_capture_discards_frame_and_prevents_retry(self):
        device = AdbDevice(self.app.config)
        def stopping(*args):
            device.stop.set()
            return png_bytes(FakeDevice.source)
        with patch.object(device, 'command', side_effect=stopping) as command:
            with self.assertRaises(InterruptedError):
                device.screenshot()
        self.assertEqual(command.call_count, 1)
        self.assertIsNone(device.image)

    def test_stopped_worker_still_allows_read_only_ui_capture(self):
        device = AdbDevice(self.app.config)
        device.stop.set()
        with patch.object(device, 'command', return_value=png_bytes(FakeDevice.source)):
            np.testing.assert_array_equal(device.screenshot(), FakeDevice.source)

    def test_exit_zero_adb_error_still_blocks_control(self):
        device = AdbDevice(self.app.config)
        responses = [SimpleNamespace(returncode=0, stdout=b'', stderr=b'error: device test not found\n'),
                     SimpleNamespace(returncode=0, stdout=b'device\n', stderr=b'* daemon started successfully *\n')]
        with patch('app.subprocess.run', side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, 'device test not found'):
                device.command('get-state')
            self.assertEqual(device.command('get-state'), b'device\n')

    def test_empty_hold_feedback_does_not_repeat_a_gesture(self):
        device = AdbDevice(self.app.config)
        with patch.object(device, 'command', return_value=b'') as command:
            with self.assertRaisesRegex(RuntimeError, '长按.*截图无效'):
                device.hold_feedback(360, 1120, 1500)
        self.assertEqual(command.call_count, 2)
        self.assertEqual(sum(call.args[0] == 'shell' for call in command.call_args_list), 1)

    def test_stop_prevents_queued_adb_click(self):
        stop = threading.Event()
        device = AdbDevice(self.app.config, stop)
        commands = []
        device.command = lambda *args: commands.append(args)
        stop.set()
        with self.assertRaises(InterruptedError):
            device.click(SimpleNamespace(button=(10, 20, 26, 38)))
        self.assertEqual(commands, [])

    def test_color_outline_ignores_background_but_requires_the_shape(self):
        silhouette = np.zeros((24, 30), dtype=bool)
        for y in range(4, 17):
            silhouette[y, 15 - (y - 4):16 + (y - 4)] = True
        silhouette[17:23, 11:20] = True
        template = np.full((24, 30, 3), (25, 40, 60), dtype=np.uint8)
        template[silhouette] = (120, 185, 70)
        button = CroppedTemplateButton(template, (10, 10, 64, 90), [1], "arrow",
                                       match_hsv=[[35, 80, 70], [85, 255, 255]])
        for background in ((190, 90, 60), (10, 65, 130)):
            frame = np.full((96, 64, 3), background, dtype=np.uint8)
            frame[45:69, 28:58][silhouette] = (120, 185, 70)
            frame[37:42, 45:60] = (255, 255, 255)  # Changing counter above the icon.
            self.assertTrue(button.match(frame, similarity=0.94))
            self.assertEqual(tuple(button.button), (28, 45, 58, 69))
        frame[:] = (10, 65, 130)
        self.assertFalse(button.match(frame, similarity=0.94))
        frame[40:60, 30:50] = (120, 185, 70)  # A green rectangle is not an arrow.
        self.assertFalse(button.match(frame, similarity=0.94))

    def test_single_action_stops_even_when_icon_remains_visible(self):
        self.rule()["stop_after"] = True
        self.app.start(preview=False)
        self.wait_for(lambda: not self.app.running())
        self.assertEqual(len(self.app.device.taps), 1)
        self.assertIsNone(self.app.status()["last_error"])

    def counter_rule(self, total=10):
        rule = self.rule()
        rule["source_crop"] = rule.pop("area")
        rule["stable_frames"] = 3
        rule["stop_after"] = True
        rule["counter_condition"] = {"source_area": [28, 8, 55, 18], "mode": "used",
                                     "expected_total": total, "confidence": 0.93}
        return rule

    def bounded_counter_run(self, readings):
        calls = 0

        def screenshot():
            nonlocal calls
            calls += 1
            self.app.device.image = FakeDevice.source.copy()
            if calls > len(readings):
                self.app.stop_event.set()
            return self.app.device.image

        self.app.device.screenshot = screenshot
        self.app.digit_reader = SimpleNamespace(read_counter=lambda _: readings[min(calls - 1, len(readings) - 1)])
        self.app.device.taps.clear()
        self.app.start(preview=False)
        self.wait_for(lambda: not self.app.running())
        self.assertIsNone(self.app.status()["last_error"])
        return len(self.app.device.taps)

    def test_daily_counter_blocks_full_unclear_and_wrong_limits(self):
        rule = self.counter_rule()
        cases = [
            ("0/10", 10, 0.99, 1), ("9/10", 10, 0.99, 1), ("10/10", 10, 0.99, 0),
            ("39/40", 40, 0.99, 1), ("40/40", 40, 0.99, 0),
            ("1/100", 10, 0.99, 0), ("1/40", 10, 0.99, 0),
            ("", 10, 0.99, 0), ("1/10", 10, 0.90, 0), ("11/10", 10, 0.99, 0),
        ]
        for text, total, confidence, expected_taps in cases:
            with self.subTest(text=text, total=total, confidence=confidence):
                rule["counter_condition"]["expected_total"] = total
                self.assertEqual(self.bounded_counter_run([parse_counter(text, confidence)] * 4), expected_taps)

    def test_daily_counter_requires_three_equal_readings(self):
        self.counter_rule()
        self.assertEqual(self.bounded_counter_run([parse_counter(text, 0.99) for text in
                                                  ("7/10", "8/10", "9/10", "10/10")]), 0)

    def test_daily_counter_uses_game_reset_instead_of_local_run_count(self):
        self.counter_rule()
        readings = [parse_counter(text, 0.99) for text in
                    ("10/10", "10/10", "10/10", "0/10", "0/10", "0/10")]
        self.assertEqual(self.bounded_counter_run(readings), 1)

    def test_counter_parser_requires_complete_valid_fraction(self):
        for text, values in [("0/10", (0, 10)), ("10/10", (10, 10)), ("39 / 40", (39, 40)),
                             ("９/１０", (9, 10))]:
            reading = parse_counter(text, 0.99)
            self.assertEqual((reading.current, reading.total), values)
        for text in ("", "10", "/10", "1/", "1/0", "11/10", "1/10/2", "I/10", "1O/10", "-1/10"):
            reading = parse_counter(text, 0.99)
            self.assertIsNone(reading.current, text)
            self.assertIsNone(reading.total, text)

    def test_scoped_run_preserves_rules_and_stops_at_click_limit(self):
        selected = self.rule()
        self.rule()
        original = json.dumps(self.app.config)
        self.app.start(preview=False, rule_ids=[selected["id"]], max_clicks=1, timeout=3)
        self.wait_for(lambda: not self.app.running())
        state = self.app.status()
        self.assertEqual(state["active_rule_ids"], [selected["id"]])
        self.assertEqual(len(self.app.device.taps), 1)
        self.assertEqual(state["stop_reason"], "click_limit")
        self.assertEqual(json.dumps(self.app.config), original)

    def test_scoped_cycle_stops_after_returning_to_main(self):
        entry = self.rule()
        entry["next_stage"] = "return"
        followup = self.rule()
        followup.update(stage="return", action="observe", next_stage="main")
        self.app.start(preview=False, rule_ids=[entry["id"], followup["id"]], stop_on_main=True, timeout=3)
        self.wait_for(lambda: not self.app.running())
        state = self.app.status()
        self.assertEqual(state["stop_reason"], "cycle_complete")
        self.assertEqual(state["stage"], "main")
        self.assertEqual(len(self.app.device.taps), 1)


if __name__ == "__main__":
    unittest.main()
