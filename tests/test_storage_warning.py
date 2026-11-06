"""Storage-full detection is observation only, including in the real loop."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from drift.runtime import Clicker
from drift.tasks.frame_runtime import FrameRuntime
from drift.web.server import make_handler

ROOT = Path(__file__).resolve().parents[1]
ASSET = 'assets/storage/storage-full-banner-source.png'


class StorageWarningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        (root/ASSET).parent.mkdir(parents=True)
        shutil.copyfile(ROOT/ASSET, root/ASSET)
        config = dict(serial='fake', adb_path='never-used', resolution=[720,1280],
                      poll_interval=.1, max_consecutive_clicks=3, rules=[])
        (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
        with Image.open(ROOT/ASSET) as source:
            self.banner = np.asarray(source.convert('RGB'))
        self.absent = self.banner.copy()
        self.absent[225:302,171:549] = 0
        frame = self.banner
        class Device:
            def __init__(self, config, stop):
                self.stop, self.image, self.reads = stop, None, 0
                self.source = frame
                self.taps = []
            def screenshot(self):
                self.reads += 1
                if self.reads > 4:
                    raise InterruptedError('bounded offline replay')
                self.image = self.source.copy()
                return self.image
            def click(self, button):
                self.taps.append(button)
                raise AssertionError('Warning detection must never click')
        self.app = Clicker(root, Device)
        self.addCleanup(self.close)

    def close(self):
        for handler in list(self.app.log.handlers):
            handler.close()
            self.app.log.removeHandler(handler)

    def warnings(self):
        handler_type = make_handler(self.app)
        handler = object.__new__(handler_type)
        handler.path = '/api/status'
        handler.respond = Mock()
        handler.do_GET()
        return json.loads(json.dumps(handler.respond.call_args.args[0]))['warnings']

    def warning_events(self):
        return [e['message'] for e in self.app.events if '保管箱已满' in e['message']]

    def test_present_banner_sets_status_and_one_event_without_action(self):
        before = deepcopy(self.app.config)
        for _ in range(3):
            self.app.capture()
        self.assertEqual(self.warnings(), ['storage_full'])
        self.assertEqual(len(self.warning_events()), 1)
        self.assertEqual(self.app.device.taps, [])
        self.assertEqual(self.app.state['clicks'], 0)
        self.assertEqual(self.app.config, before)
        self.assertIsNone(self.app.state['last_error'])

    def test_absent_banner_never_emits_warning(self):
        self.app.device.source = self.absent
        self.app.capture()
        self.assertEqual(self.warnings(), [])
        self.assertEqual(self.warning_events(), [])
        self.assertEqual(self.app.device.taps, [])

    def test_partial_text_or_other_modal_does_not_match(self):
        for box in ((294,238,416,262), (205,264,517,294)):
            frame = self.banner.copy()
            x1,y1,x2,y2 = box
            frame[y1:y2,x1:x2] = 0
            self.app.remember(frame)
            self.assertEqual(self.warnings(), [])
        with Image.open(ROOT/'assets/potion-ad-confirm-source.png') as source:
            self.app.remember(np.asarray(source.convert('RGB')))
        self.assertEqual(self.warnings(), [])
        self.assertEqual(self.warning_events(), [])

    def test_warning_clears_when_absent_and_reappears_without_erasing_others(self):
        self.app.state['warnings'] = ['other_warning']
        self.app.remember(self.banner)
        self.assertEqual(self.warnings(), ['other_warning', 'storage_full'])
        self.app.remember(self.absent)
        self.assertEqual(self.warnings(), ['other_warning'])
        self.app.remember(self.banner)
        self.assertEqual(len(self.warning_events()), 2)

    def test_runner_continues_to_scan_without_input(self):
        with patch.object(FrameRuntime, 'start_engine'), \
             patch.object(FrameRuntime, 'wait', return_value=False):
            self.app.run([], False)
        self.assertEqual(self.app.state['frames'], 4)
        self.assertEqual(self.warnings(), ['storage_full'])
        self.assertIsNone(self.app.state['last_error'])
        self.assertEqual(self.app.device.taps, [])
        self.assertEqual(len(self.warning_events()), 1)


if __name__ == '__main__':
    unittest.main()
