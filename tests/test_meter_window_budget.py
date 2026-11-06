"""A short stop opportunity must not be lost to OCR or used too late."""
from copy import deepcopy
from unittest.mock import patch
import unittest

from tests import test_meter_calibration as base


class MeterWindowBudgetTests(unittest.TestCase):
    setUpClass = classmethod(base.MeterCalibrationTests.setUpClass.__func__)
    setUp = base.MeterCalibrationTests.setUp
    tearDown = base.MeterCalibrationTests.tearDown
    pending = base.MeterCalibrationTests.pending

    def current(self):
        c = self.pending()
        c.journal.update(route_policy='preserve', calibration_context='place')
        c.route_verified = True
        c._save()
        return c

    def test_54_seconds_elapsed_cannot_cancel_or_consume_an_attempt(self):
        c = self.current()
        before = deepcopy(c.journal)
        with patch('meter_calibration.time.time', return_value=1054):
            for action in ('cancel', 'scan_cancel'):
                self.assertFalse(c.reserve_meter_input(action))
        self.assertEqual(c.journal, before)
        self.assertEqual(self.inputs, [])

    def test_cancel_budget_boundary_keeps_existing_deadline(self):
        c = self.current()
        with patch('meter_calibration.time.time', return_value=1030):
            self.assertTrue(c.reserve_meter_input('scan_cancel'))
        self.assertEqual(c.journal['attempts'], {'cancel': 1})
        self.assertEqual(c.journal['deadline'], 1090)
        with patch('meter_calibration.time.time', return_value=1030.001):
            self.assertFalse(c.reserve_meter_input('cancel'))
        self.assertEqual(c.journal['attempts'], {'cancel': 1})

    def test_late_untouched_window_archives_without_stopping_or_marking_success(self):
        c = self.current()
        original = c.path.read_bytes()
        with patch('meter_calibration.time.time', return_value=1054), \
             patch.object(c, 'screen', side_effect=AssertionError('no input probe')):
            self.assertFalse(c.step(base.frame('sleep')))
        self.assertFalse(c.active)
        self.assertTrue(c.awaiting_recovery)
        self.assertFalse(c.journal['retirement']['had_inputs'])
        self.assertEqual(c.journal['attempts'], {})
        self.assertEqual(c.journal['deadline'], 1090)
        self.assertEqual(c.cycles, 0)
        from pathlib import Path
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(), original)
        self.assertEqual(self.inputs, [])

    def test_existing_cancel_continues_recovery_in_original_window(self):
        c = self.current()
        c.journal['attempts'] = {'cancel': 1}
        with patch('meter_calibration.time.time', return_value=1061), \
             patch.object(c.world.training, 'boss_mode', return_value=False), \
             patch.object(c, '_restore_fishing', return_value=True) as restore:
            self.assertTrue(c.step(base.frame('sleep')))
        restore.assert_called_once()
        self.assertEqual(c.phase, 'restore_fishing')
        self.assertEqual(c.journal['deadline'], 1090)
        self.assertEqual(c.journal['attempts'], {'cancel': 1})

    def test_known_prepare_does_not_repeat_rank_ocr_before_fresh_scan(self):
        c = self.current()
        picture = base.frame('sleep')
        with patch('meter_calibration.time.time', return_value=1000), \
             patch.object(c, 'meter_frame_allowed', return_value=True), \
             patch.object(c, 'screen', side_effect=AssertionError('redundant OCR')), \
             patch.object(c, 'step', return_value=True) as step:
            self.assertTrue(c.pre_battle_step(picture))
        step.assert_called_once_with(picture)

    def test_changed_prepare_still_requires_full_context_reconfirmation(self):
        c = self.current()
        with patch('meter_calibration.time.time', return_value=1000), \
             patch.object(c, 'meter_frame_allowed', return_value=False), \
             patch.object(c, 'screen', return_value=None) as screen, \
             patch.object(c, 'step', side_effect=AssertionError('invalid context')):
            self.assertFalse(c.pre_battle_step(base.frame('sleep')))
        screen.assert_called_once()
        self.assertEqual(self.inputs, [])

    def test_confirmed_candidate_requests_fresh_scan_in_same_iteration(self):
        c = self.controller
        data = dict(player=1609, value=1515, high=1604, context='place', adaptive=True)
        view = dict(rank=38, stage=10, menu=False, route=False)
        with patch.object(c, 'screen', return_value=view), \
             patch.object(c.fishing, 'measure', return_value=data), \
             patch.object(c.world, '_stage_strokes', return_value=b'stage'), \
             patch.object(c, '_candidate_ready', return_value=True), \
             patch.object(c, 'meter_frame_allowed', return_value=True), \
             patch.object(c, '_observe_cancel_window', return_value=True) as scan, \
             patch('meter_calibration.time.time', return_value=1000):
            for now in (100, 101.5, 103):
                with patch('meter_calibration.time.monotonic', return_value=now):
                    c.step(base.frame('sleep').copy())
        scan.assert_called_once()
        self.assertEqual(c.journal['deadline'], 1090)
        self.assertEqual(c.journal['attempts'], {})
        self.assertEqual(self.inputs, [])

    def test_expired_prebattle_still_refuses_all_input(self):
        c = self.current()
        with patch('meter_calibration.time.time', return_value=1091), \
             patch.object(c, 'meter_frame_allowed', side_effect=AssertionError('expired')):
            self.assertFalse(c.pre_battle_step(base.frame('sleep')))
        self.assertEqual(self.inputs, [])


if __name__ == '__main__':
    unittest.main()
