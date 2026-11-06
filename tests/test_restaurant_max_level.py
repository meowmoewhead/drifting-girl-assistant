"""Real Lv19->20 layout regression; no ingredient input is valid on max detail."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import cv2
import numpy as np

from drift.perception.restaurant import RestaurantVisual
from drift.perception.digit_ocr import DigitReader
from drift.tasks.restaurant_upgrade import pack
import tests.test_restaurant_upgrade as existing

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'diagnostics/2026-10-29/restaurant-run-2307'


class RestaurantMaxLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.visual = RestaurantVisual(ROOT, DigitReader(ROOT))
        cls.maximum = cls.read('finish-native-f35dd1a1a2ef45c68df9bdbae84fec19.png')
        cls.previous = cls.read('at-restaurant/observe-7ccfdf4cbdf042819d8aa2e785939c35.png')

    @staticmethod
    def read(name):
        frame = cv2.imread(str(EVIDENCE / name))
        if frame is None:
            raise unittest.SkipTest('Local live evidence unavailable')
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def controller(self, directory):
        c = existing.RestaurantJournalTests().controller(directory, 'upgrade_wait')
        c.visual = self.visual
        c.same_world = lambda _: True
        c.sailing = Mock()
        c.sailing.menu.return_value = False
        c.journal.update(
            awaiting=dict(identity='the-dish', level=19, counters=[(12, 12)] * 4),
            awaiting_name=pack(self.previous[480:510, 209:530]),
            dish_identity='the-dish', card=dict(level=19, identity='the-dish'),
            attempts={'upgrade:the-dish:19': 1})
        c.action = Mock(return_value=True)
        c._save()
        return c

    def test_max_layout_reads_twenty_and_foreground_close(self):
        f = self.maximum.copy()
        self.assertTrue(self.visual.max_level(f))
        self.assertTrue(self.visual.detail(f))
        self.assertFalse(self.visual.atlas(f))
        self.assertEqual(self.visual.point(f, 'detail-close'), (658, 517))
        self.assertIsNone(self.visual.detail_values(f))
        for i in range(4):
            self.assertFalse(self.visual.ingredient_ready(f, i, None))
        self.assertFalse(self.visual.upgrade_ready(f, None))

    def test_ordinary_detail_and_crafting_are_not_max(self):
        self.assertFalse(self.visual.max_detail(self.previous.copy()))
        craft = self.read('after-avatar/observe-430c12cca0b14487a1694735d3ca759a.png')
        self.assertFalse(self.visual.max_detail(craft))
        self.assertFalse(self.visual.detail(craft))

    def test_dark_missing_book_and_missing_full_progress_refused(self):
        self.assertFalse(self.visual.max_detail((self.maximum * .4).astype('uint8')))
        for y, Y, x, X in ((475, 561, 37, 109), (668, 702, 330, 405)):
            f = self.maximum.copy()
            f[y:Y, x:X] = 0
            self.assertFalse(self.visual.max_detail(f))

    def test_name_must_match_complete_pre_upgrade_title(self):
        name = self.previous[480:510, 209:530]
        self.assertTrue(self.visual.max_name_matches(self.maximum.copy(), name))
        self.assertFalse(self.visual.max_name_matches(self.maximum.copy(), np.roll(name, 40, axis=1)))
        self.assertFalse(self.visual.max_name_matches(self.maximum.copy(), np.full_like(name, 255)))

    def test_final_upgrade_requires_three_fresh_frames_and_three_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            for now, expected in ((100, 0), (101, 0), (103, 1)):
                c.clock = lambda now=now: now
                c.step(self.maximum.copy(), due=False)
                self.assertEqual(c.journal['upgraded'], expected)
            self.assertEqual(c.phase, 'detail_close')
            self.assertEqual(c.journal['card']['level'], 20)
            self.assertEqual(c.journal['attempts'], {'upgrade:the-dish:19': 1})
            c.action.assert_not_called()
            self.assertFalse(c.validate(dict(kind='cleanup-panel-close', phase='detail_close', x=637, y=182), self.maximum.copy()))
            c.clock = lambda: 104
            c.step(self.maximum.copy(), due=False)
            self.assertEqual((c.action.call_args[0][0]['x'], c.action.call_args[0][0]['y']), (658, 517))

    def test_no_upgrade_count_without_matching_name_and_prior_attempt(self):
        for missing in ('awaiting_name', 'attempts'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as d:
                c = self.controller(d)
                c.journal[missing] = {} if missing == 'attempts' else None
                for now in (100, 101, 104):
                    c.clock = lambda now=now: now
                    c.step(self.maximum.copy(), due=False)
                self.assertEqual(c.journal['upgraded'], 0)
                c.action.assert_not_called()

    def test_expired_window_cannot_count_late_maximum_as_success(self):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(d)
            c.clock = lambda: 201
            c.step(self.maximum.copy(), due=False)
            self.assertEqual(c.journal['upgraded'], 0)
            self.assertEqual(c.journal['failure']['phase'], 'upgrade_wait')
            self.assertEqual(c.journal['deadline'], 1000)


if __name__ == '__main__':
    unittest.main()
