import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from drift.device import discovery, profiles, selection


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(adb_path="adb.exe", serial="emulator-5554", resolution=[720, 1280])

    def discover(self, listing, sizes=None):
        sizes = sizes or {}
        def run(argv, **kwargs):
            if argv[1:] == ["devices", "-l"]:
                return listing.encode()
            return sizes.get(argv[2], b"Physical size: 720x1280")
        return discovery.discover(self.config, runner=run, finder=lambda value: value)

    def test_lists_online_offline_and_unauthorized_without_input(self):
        result = self.discover("List of devices attached\nemulator-5554 device model:MuMu_Player\n"
                               "emulator-5556 offline\nphone unauthorized\n")
        self.assertEqual([d["state"] for d in result["devices"]], ["device", "offline", "unauthorized"])
        self.assertTrue(result["automatic_available"])
        self.assertEqual(result["devices"][0]["name"], "MuMu Player")

    def test_auto_only_chooses_one_compatible_device(self):
        result = self.discover("emulator-5554 device\nemulator-5556 device")
        with self.assertRaisesRegex(ValueError, "多个"):
            discovery.choose(result, "auto")
        result = self.discover("emulator-5554 device\nemulator-5556 offline")
        self.assertEqual(discovery.choose(result, "auto")["serial"], "emulator-5554")

    def test_auto_with_no_ready_device_refuses(self):
        with self.assertRaisesRegex(ValueError, "没有"):
            discovery.choose(self.discover("emulator-5554 offline"), "auto")

    def test_override_dimensions_and_wrong_resolution_are_reported(self):
        result = self.discover("emulator-5554 device\nemulator-5556 device",
                              {"emulator-5554": b"Physical size: 1080x1920\nOverride size: 720x1280",
                               "emulator-5556": b"Physical size: 1280x720"})
        self.assertTrue(result["devices"][0]["compatible"])
        self.assertFalse(result["devices"][1]["compatible"])
        with self.assertRaisesRegex(ValueError, "720"):
            discovery.choose(result, "emulator-5556")

    def test_invalid_and_repeated_serial_lines_are_ignored(self):
        result = self.discover("* daemon started successfully\n-s device\nemulator-5554 device\nemulator-5554 device")
        self.assertEqual(len(result["devices"]), 1)
        with self.assertRaises(ValueError):
            discovery.choose(result, "-s")

    def test_read_timeout_does_not_claim_device_is_ready(self):
        def run(argv, **kwargs):
            if "devices" in argv:
                return b"emulator-5554 device"
            raise ValueError("检查超时")
        result = discovery.discover(self.config, runner=run, finder=lambda v: v)
        self.assertFalse(result["devices"][0]["compatible"])
        self.assertIn("超时", result["devices"][0]["reason"])

    def test_new_screenshot_is_required_and_resolution_rechecked(self):
        out = BytesIO(); Image.new("RGB", (720, 1280)).save(out, format="PNG")
        runner = Mock(return_value=out.getvalue())
        frame = discovery.verify_frame("adb", "emulator-5554", [720, 1280], runner)
        self.assertEqual(frame.shape, (1280, 720, 3))
        self.assertEqual(runner.call_args.args[0], ["adb", "-s", "emulator-5554", "exec-out", "screencap", "-p"])
        with self.assertRaisesRegex(ValueError, "尺寸"):
            discovery.verify_frame("adb", "emulator-5554", [1280, 720], runner)
        with self.assertRaisesRegex(ValueError, "有效"):
            discovery.verify_frame("adb", "emulator-5554", [720, 1280], Mock(return_value=b"broken"))


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text('{"serial":"one","rules":[]}', encoding="utf-8")
        self.old = b'{"pending":true,"deadline":100,"attempts":{"start":1}}\r\n'
        (self.root / "meter_calibration_state.json").write_bytes(self.old)
        (self.root / "stone_farm_retry.json").write_bytes(b'{"expires_at":90}')

    def test_switch_back_preserves_deadlines_and_spent_inputs_byte_for_byte(self):
        original = (self.root / "config.json").read_bytes()
        with profiles.switch_records(self.root, "one", "two"):
            self.assertFalse((self.root / "meter_calibration_state.json").exists())
        second = b'{"pending":true,"deadline":150,"attempts":{"start":2}}'
        (self.root / "meter_calibration_state.json").write_bytes(second)
        with profiles.switch_records(self.root, "two", "one"):
            self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), self.old)
            self.assertEqual((self.root / "stone_farm_retry.json").read_bytes(), b'{"expires_at":90}')
        with profiles.switch_records(self.root, "one", "two"):
            self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), second)
        self.assertEqual((self.root / "config.json").read_bytes(), original)

    def test_failure_before_config_commit_rolls_back(self):
        with self.assertRaisesRegex(OSError, "disk"):
            with profiles.switch_records(self.root, "one", "two"):
                raise OSError("disk full")
        self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), self.old)
        self.assertFalse((self.root / "device-profiles/pending-switch.json").exists())

    def test_restart_rolls_back_interrupted_switch_if_config_still_old(self):
        marker = self.root / "device-profiles/pending-switch.json"
        marker.parent.mkdir()
        marker.write_text(json.dumps(dict(version=1, old_serial="one", new_serial="two",
                                         before={"meter_calibration_state.json": base64.b64encode(self.old).decode()})))
        (self.root / "meter_calibration_state.json").unlink()
        profiles.recover_switch(self.root)
        self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), self.old)
        self.assertFalse(marker.exists())

    def test_committed_config_keeps_destination_records_after_restart(self):
        marker = self.root / "device-profiles/pending-switch.json"; marker.parent.mkdir()
        marker.write_text(json.dumps(dict(version=1, old_serial="two", new_serial="one", before={})))
        profiles.recover_switch(self.root)
        self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), self.old)

    def test_same_device_does_not_write_or_retire_any_record(self):
        with profiles.switch_records(self.root, "one", "one"):
            pass
        self.assertFalse((self.root / "device-profiles").exists())
        self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), self.old)

    def test_profile_cannot_write_other_files_or_escape_root(self):
        for name in ("../config.json", "config.json", "assets/meter_state.json"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                profiles._restore(self.root, {name: base64.b64encode(b"x").decode()})
        self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), self.old)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = dict(serial="one", adb_path="adb.exe", resolution=[720, 1280], rules=[])
        (self.root / "config.json").write_text(json.dumps(config))
        self.app = SimpleNamespace(root=self.root, config=config, lock=threading.RLock(),
                                   idle_only=Mock(), device=SimpleNamespace(config=config, io_lock=threading.RLock()),
                                   stop_event=threading.Event(), device_factory=Mock(return_value=SimpleNamespace()),
                                   frames={}, state={}, runtime=None, remember=Mock(return_value="new-frame"),
                                   frame_info=Mock(return_value=dict(id="new-frame", width=720, height=1280)), event=Mock())
        def commit(updated):
            (self.root / "config.json").write_text(json.dumps(updated))
            self.app.config = updated
        self.app.commit_config = Mock(side_effect=commit)
        self.scan = dict(adb_path="adb.exe", devices=[dict(serial="two", name="second", compatible=True)],
                         automatic_available=True)

    def select(self, serial="two"):
        with patch.object(discovery, "discover", return_value=self.scan), \
             patch.object(discovery, "verify_frame", return_value=np.zeros((1280, 720, 3), dtype=np.uint8)):
            return selection.select_device(self.app, {"serial": serial})

    def test_selected_device_commits_only_connection_and_never_starts_worker(self):
        before = deepcopy(self.app.config)
        self.app.config["rules"] = [{"enabled": True, "id": "custom"}]
        selected = self.select()
        self.assertEqual(self.app.config["serial"], "two")
        self.assertEqual(self.app.config["rules"], [{"enabled": True, "id": "custom"}])
        self.assertEqual(self.app.config["resolution"], before["resolution"])
        self.assertEqual(selected["frame"]["id"], "new-frame")
        self.assertEqual(self.app.state["mode"], "idle")
        self.assertEqual(self.app.daily_schedule.serial, "two")

    def test_running_worker_cannot_be_retargeted(self):
        self.app.idle_only.side_effect = ValueError("请先停止")
        with patch.object(discovery, "discover") as scan, self.assertRaisesRegex(ValueError, "停止"):
            selection.select_device(self.app, {"serial": "two"})
        scan.assert_not_called(); self.app.commit_config.assert_not_called()

    def test_reselect_current_instance_keeps_records_and_config(self):
        record = b'{"attempts":{"start":1},"deadline":123}'
        (self.root / "meter_calibration_state.json").write_bytes(record)
        self.scan["devices"][0]["serial"] = "one"
        self.assertFalse(self.select("one")["changed"])
        self.app.commit_config.assert_not_called()
        self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), record)

    def test_invalid_screenshot_cannot_save_choice_or_move_records(self):
        with patch.object(discovery, "discover", return_value=self.scan), \
             patch.object(discovery, "verify_frame", side_effect=ValueError("bad frame")), \
             self.assertRaisesRegex(ValueError, "bad frame"):
            selection.select_device(self.app, {"serial": "two"})
        self.app.commit_config.assert_not_called()
        self.assertFalse((self.root / "device-profiles").exists())

    def test_config_write_failure_preserves_all_old_records(self):
        record = b'{"attempts":{"start":1}}'
        (self.root / "meter_calibration_state.json").write_bytes(record)
        self.app.commit_config.side_effect = OSError("disk full")
        with self.assertRaises(OSError):
            self.select()
        self.assertEqual((self.root / "meter_calibration_state.json").read_bytes(), record)
        self.assertEqual(self.app.config["serial"], "one")


if __name__ == "__main__":
    unittest.main()
