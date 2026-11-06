"""A new due run can enter a known leftover container without reviving it."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2

from drift.tasks.restaurant_upgrade import RestaurantUpgrade

ROOT = Path(__file__).resolve().parents[1]


class RestaurantPanelEntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1000.0
        self.path = Path(self.tmp.name) / 'restaurant.json'
        self.old = dict(version=1, serial='emulator-5554', pending=False,
                        phase='idle', run_id='failed-original', deadline=1,
                        attempts={'restaurant-open': 1}, success=False)
        self.raw = json.dumps(self.old).encode()
        self.path.write_bytes(self.raw)
        self.action = Mock()
        self.task = RestaurantUpgrade(ROOT, None, self.action, Mock(),
                                      'emulator-5554', state_path=self.path,
                                      clock=lambda: self.now)
        self.task.root = Path(self.tmp.name)
        self.frame = cv2.cvtColor(cv2.imread(str(ROOT /
            'tests/fixtures/restaurant-list/storage-tab.png')), cv2.COLOR_BGR2RGB)

    def test_real_storage_panel_is_a_stopped_restaurant_start(self):
        result = self.task.panel_start_screen(self.frame)
        self.assertIsNotNone(result)
        self.assertTrue(result['parallel'])
        self.assertIs(result['route'], False)
        self.action.assert_not_called()

    def test_new_round_archives_failed_record_after_three_fresh_frames(self):
        for elapsed in (0, 1.5, 3):
            self.now = 1000 + elapsed
            self.task.step(self.frame.copy(), due=True)
        self.assertTrue(self.task.active)
        self.assertEqual(self.task.phase, 'open')
        self.assertTrue(self.task.journal['origin']['at_restaurant'])
        self.assertFalse(self.task.journal['resume_route'])
        self.assertNotEqual(self.task.journal['run_id'], self.old['run_id'])
        archives = list((self.task.root / 'logs/restaurant-history').glob('*.json'))
        self.assertEqual([p.read_bytes() for p in archives], [self.raw])
        self.action.assert_not_called()

    def test_cached_frame_cannot_start_a_round(self):
        for elapsed in (0, 2, 4):
            self.now = 1000 + elapsed
            self.task.step(self.frame, due=True)
        self.assertFalse(self.task.active)
        self.assertEqual(self.path.read_bytes(), self.raw)

    def test_not_due_keeps_record_and_container_untouched(self):
        for elapsed in (0, 2, 4):
            self.now = 1000 + elapsed
            self.task.step(self.frame.copy(), due=False)
        self.assertFalse(self.task.active)
        self.assertEqual(self.path.read_bytes(), self.raw)
        self.action.assert_not_called()

    def test_unknown_header_location_or_container_is_rejected(self):
        for area in [(35, 911, 165, 947), (285, 12, 434, 36), (592, 145, 683, 220)]:
            frame = self.frame.copy()
            x, y, X, Y = area
            frame[y:Y, x:X] = 0
            self.assertIsNone(self.task.panel_start_screen(frame), area)
        self.assertIsNone(self.task.panel_start_screen(self.frame[:1000]))

    def test_running_battle_motion_or_detail_is_rejected(self):
        for owner, name, value in [
            (self.task.sailing, 'running', True),
            (self.task.sailing, 'menu', True),
            (self.task.view, '_off_icon', False),
            (self.task.view, '_battle', True),
            (self.task.bag, 'moving', True),
            (self.task.visual, 'detail', True),
        ]:
            with patch.object(owner, name, return_value=value):
                self.assertIsNone(self.task.panel_start_screen(self.frame), name)


if __name__ == '__main__':
    unittest.main()
