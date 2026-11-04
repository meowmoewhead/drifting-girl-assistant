"""Offline regression coverage for the 2026-10-28 restaurant map failure."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2

from drift.perception.restaurant import RestaurantVisual
from drift.tasks import restaurant_upgrade as module
from drift.tasks.restaurant_upgrade import RestaurantUpgrade


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/restaurant-map'


def frame(name):
    image = cv2.imread(str(FIXTURES / name))
    if image is None:
        raise AssertionError('Required restaurant map fixture missing: ' + name)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class RestaurantMapVisualTests(unittest.TestCase):
    def setUp(self):
        self.visual = RestaurantVisual(ROOT, None)

    def test_live_icon_with_score_08880_is_accepted(self):
        hit = self.visual.find(frame('s02.png'), 'map-restaurant')
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit[2], .8880, delta=.0001)
        self.assertEqual(hit[0], (576, 292))
        self.assertEqual(hit[1], (551, 271, 51, 43))

    def test_strongest_negative_with_score_03989_is_rejected(self):
        image = frame('negative-03989.png')
        self.assertIsNone(self.visual.find(image, 'map-restaurant'))
        # Confirm this is the measured hard negative using the production
        # grayscale matcher, with only its acceptance threshold disabled.
        probe = RestaurantVisual(ROOT, None)
        probe.spec['map-restaurant']['threshold'] = -1
        hit = probe.find(image, 'map-restaurant')
        self.assertAlmostEqual(hit[2], .3989, delta=.0001)

    def test_left_edge_real_icons_are_accepted(self):
        for name, score in (('s03.png', .8751), ('s04.png', .8773)):
            with self.subTest(frame=name):
                image = frame(name)
                hit = self.visual.find(image, 'map-restaurant')
                self.assertIsNotNone(hit)
                self.assertEqual(hit[0], (43, 292))
                self.assertEqual(hit[1], (18, 271, 51, 43))
                self.assertAlmostEqual(hit[2], score, delta=.0001)
                self.assertIsNone(self.visual.find(
                    image, 'map-restaurant', area=[20, 170, 700, 1115]))

    def test_only_map_restaurant_uses_relaxed_threshold(self):
        self.assertEqual(self.visual.spec['map-restaurant']['threshold'], .85)
        others = {name: spec['threshold'] for name, spec in self.visual.spec.items()
                  if name != 'map-restaurant'}
        self.assertEqual(len(others), 38)
        self.assertEqual(set(others.values()), {.94})


class RestaurantMapNavigationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.now = 100
        self.task = RestaurantUpgrade(
            ROOT, None, Mock(), Mock(), 'offline',
            state_path=Path(folder.name) / 'state.json', clock=lambda: self.now)
        self.task.root = Path(folder.name)
        self.task.same_world = Mock(return_value=True)
        self.task.map_ready = Mock(return_value=True)
        self.task.view._off_icon = Mock(return_value=True)
        self.task.phase = 'map'
        self.task.journal.update(
            pending=True, phase='map', destination='restaurant', scans=0,
            attempts={}, deadline=1000, phase_deadline=220,
            origin=dict(parallel=False, route=True), resume_route=True)
        self.task._save()

        def issue(action):
            # Exercise validation, reservations and receipts without device I/O.
            if not self.task.validate(action, self.current.copy()):
                return False
            if not self.task.reserve(action):
                return False
            self.task.input_issued(action, dict(input_id='offline-map-input'))
            return True

        self.task.action = Mock(side_effect=issue)

    def confirm(self, name, start):
        image = frame(name)
        for now in (start, start + 1, start + 2):
            self.now = now
            self.current = image.copy()
            self.assertTrue(self.task.step(self.current))

    def test_east_scan_uses_slow_follow_drag(self):
        self.confirm('negative-03989.png', 100)
        begin, end = module.SCAN_DRAGS['east']
        self.task.action.assert_called_once_with(dict(
            kind='scan:restaurant', x=begin[0], y=begin[1], phase='map',
            swipe=[end[0], end[1], module.SCAN_DRAG_MS]))
        self.assertEqual(self.task.journal['scans'], 1)
        self.assertEqual(self.task.journal['attempts'], {'scan:restaurant': 1})

    def test_west_scan_uses_slow_follow_drag_one_row_later(self):
        first = module.SCAN_STEPS_PER_ROW + 1
        self.task.journal.update(scans=first, attempts={'scan:restaurant': first})
        self.confirm('negative-03989.png', 100)
        begin, end = module.SCAN_DRAGS['west']
        self.task.action.assert_called_once_with(dict(
            kind='scan:restaurant', x=begin[0], y=begin[1], phase='map',
            swipe=[end[0], end[1], module.SCAN_DRAG_MS]))
        self.assertEqual(self.task.journal['scans'], first + 1)
        self.assertEqual(self.task.journal['attempts'],
                         {'scan:restaurant': first + 1})

    def test_scan_then_live_icon_selects_restaurant_and_closes_map(self):
        self.confirm('negative-03989.png', 100)
        # 拖后帧一旦命中就立即选择，不再等三轮确认。
        self.now = 104
        self.current = frame('s02.png')
        self.assertTrue(self.task.step(self.current))
        self.assertEqual(self.task.action.call_count, 2)
        self.assertEqual(self.task.action.call_args.args[0], dict(
            kind='map-select:restaurant', x=576, y=292, phase='map'))
        self.assertEqual(self.task.phase, 'map_close')
        self.assertEqual(self.task.journal['scans'], 1)
        self.assertEqual(self.task.journal['attempts'],
                         {'scan:restaurant': 1, 'map-select:restaurant': 1})
        self.assertNotIn('failure', self.task.journal)

    def test_forty_unsuccessful_scans_enter_cleanup(self):
        self.task.journal.update(scans=module.SCAN_LIMIT,
                                 attempts={'scan:restaurant': module.SCAN_LIMIT})
        self.task._save()
        self.confirm('negative-03989.png', 100)
        self.task.action.assert_not_called()
        self.assertEqual(self.task.phase, 'cleanup')
        self.assertEqual(self.task.journal['failure']['reason'], '餐馆地图搜索已达上限')


if __name__ == '__main__':
    unittest.main()
