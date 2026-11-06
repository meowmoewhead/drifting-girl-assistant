"""A late meter check must leave the native route alone, not half-stop it."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_meter_preflight as base


class StopBudgetTests(unittest.TestCase):
    setUpClass = classmethod(base.MeterPreflightTests.setUpClass.__func__)
    setUp = base.MeterPreflightTests.setUp
    tearDown = base.MeterPreflightTests.tearDown

    def test_4_66_seconds_remaining_rejects_fresh_stop_without_using_an_attempt(self):
        c = self.c
        c.journal['deadline'] = 1004.66
        c._save()
        raw = c.path.read_bytes()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            self.assertTrue(c.input_allowed())
            self.assertFalse(c.validate_sailing('stop', self.frames['timeout-result']))
            self.assertFalse(c.reserve_sailing_input('stop'))
            c.pre_battle_step(self.frames['timeout-result'].copy())
        self.assertEqual(c.path.read_bytes(), raw)
        self.assertEqual(self.inputs, [])
        self.assertFalse(c.off_verified)

    def test_full_stop_budget_is_required_even_for_another_attempt(self):
        c = self.c
        c.journal['sailing_attempts'] = {'stop': 1}
        for remaining, allowed in ((15, True), (14.999, False), (4.66, False), (0, False)):
            with self.subTest(remaining=remaining):
                c.journal['deadline'] = 1000 + remaining
                with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
                    self.assertEqual(c.stop_input_allowed(), allowed)
        self.assertEqual(c.journal['sailing_attempts'], {'stop': 1})

    def test_ocr_cost_is_rechecked_before_stop_and_reservation(self):
        c = self.c
        c.journal['deadline'] = 1020
        clock = [1000]
        def slow_screen(_):
            clock[0] = 1006
            return dict(rank=39, stage=8, route=True, menu=False)
        with patch('drift.tasks.meter_calibration.time.time', side_effect=lambda: clock[0]), \
             patch.object(c, 'screen', side_effect=slow_screen):
            self.assertFalse(c.validate_sailing('stop', self.frames['timeout-result']))
            self.assertFalse(c.reserve_sailing_input('stop'))
        self.assertEqual(c.journal['sailing_attempts'], {})

    def test_healthy_untouched_route_is_deferred_with_original_bytes_and_deadline(self):
        c = self.c
        c.journal['deadline'] = 1014
        c._save()
        raw = c.path.read_bytes()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for moment in (100, 101.5, 103):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=moment):
                    c.step(self.frames['timeout-result'].copy())
        self.assertFalse(c.active)
        self.assertEqual(c.journal['resolution']['kind'], 'deferred_before_any_input')
        self.assertEqual(Path(c.journal['resolution']['archive']).read_bytes(), raw)
        self.assertEqual(c.journal['deadline'], 1014)
        self.assertEqual(c.cycles, 24)
        self.assertEqual(self.inputs, [])

    def test_used_stop_remains_unresolved_and_cannot_be_reclassified_as_untouched(self):
        c = self.c
        c.journal.update(deadline=1014, sailing_attempts={'stop': 1})
        c._save()
        saved = deepcopy(c.journal)
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for moment in (100, 101.5, 103):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=moment):
                    c.step(self.frames['timeout-result'].copy())
        self.assertTrue(c.active)
        self.assertEqual(c.journal, saved)
        self.assertEqual(self.inputs, [])

    def test_unknown_or_obscured_frames_cannot_prove_zero_input_deferral(self):
        c = self.c
        c.journal['deadline'] = 1014
        c._save()
        raw = c.path.read_bytes()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for moment in (100, 101.5, 103):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=moment):
                    c.step(np.zeros_like(self.frames['timeout-result']))
        self.assertEqual(c.path.read_bytes(), raw)
        self.assertTrue(c.active)
        self.assertEqual(self.inputs, [])

