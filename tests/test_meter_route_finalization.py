"""Expired sailing recovery may finish only from new read-only evidence."""
import hashlib
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import test_meter_rank_progress as rank_tests

ROOT = Path(__file__).resolve().parents[1]


class MeterRouteFinalizationTests(unittest.TestCase):
    setUpClass = classmethod(rank_tests.MeterRankProgressTests.setUpClass.__func__)
    setUp = rank_tests.MeterRankProgressTests.setUp
    tearDown = rank_tests.MeterRankProgressTests.tearDown

    def expired_restore(self):
        c = self.c
        c.journal.update(proof='post_release_fishing', resume_requested=True,
            attempts={'cancel': 1, 'drag': 1, 'start': 1},
            sailing_attempts={'stop': 1, 'open': 1, 'start': 1},
            restore_retry={'deadline': 999, 'from_stopped_main': True})
        c.phase = 'running'
        c._save()
        return c, c.path.read_bytes()

    def test_consumed_start_does_not_hide_new_rank_evidence(self):
        c, raw = self.expired_restore()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for now in (100, 101.5):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=now):
                    self.assertTrue(c.step(self.frame.copy()))
                self.assertEqual(c.path.read_bytes(), raw)
                self.assertFalse(c.input_allowed())
            with patch('drift.tasks.meter_calibration.time.monotonic', return_value=103):
                self.assertFalse(c.step(self.frame.copy()))
        resolution = c.journal['resolution']
        self.assertEqual(resolution['kind'], 'verified_route_after_rank_progress')
        self.assertFalse(resolution['calibration_completed'])
        self.assertEqual(Path(resolution['archive']).read_bytes(), raw)
        self.assertEqual(resolution['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(c.journal['cycles'], 10)
        self.assertEqual(c.journal['rank'], 37)
        self.assertEqual(c.journal['deadline'], 990)
        self.assertEqual(c.journal['restore_retry']['deadline'], 999)
        self.assertEqual(c.journal['attempts'], {'cancel': 1, 'drag': 1, 'start': 1})
        self.assertEqual(c.journal['sailing_attempts'], {'stop': 1, 'open': 1, 'start': 1})
        self.assertEqual(self.inputs, [])

    def test_readonly_restart_window_is_bounded_and_never_allows_inputs(self):
        c, raw = self.expired_restore()
        blank = np.zeros_like(self.frame)
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for now in (100, 110, 114.9):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=now):
                    self.assertTrue(c.step(blank))
                self.assertFalse(c.input_allowed())
                for action in ('cancel', 'drag', 'start', 'reveal'):
                    self.assertFalse(c.reserve_meter_input(action))
                for action in ('stop', 'open', 'start'):
                    self.assertFalse(c.reserve_sailing_input(action))
            with patch('drift.tasks.meter_calibration.time.monotonic', return_value=115):
                self.assertFalse(c.step(blank))
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(), raw)
        self.assertEqual(self.inputs, [])

    def test_unknown_frame_resets_new_rank_consensus(self):
        c, raw = self.expired_restore()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for now, image, pending in ((100, self.frame, True),
                    (101.5, np.zeros_like(self.frame), True),
                    (103, self.frame, True), (104.5, self.frame, True),
                    (106, self.frame, False)):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=now):
                    self.assertEqual(c.step(image.copy()), pending)
        self.assertEqual(Path(c.journal['resolution']['archive']).read_bytes(), raw)
        self.assertEqual(self.inputs, [])

    def test_full_battle_can_prove_fishing_when_low_fill_is_not_detected(self):
        c, raw = self.expired_restore()
        frame = np.asarray(Image.open(ROOT / 'tests/fixtures/stone-farm/lower-seagod-battle.png').convert('RGB'))
        c.journal['rank'] = 38
        self.assertTrue(c.fishing.real_battle(frame))
        fixed = c.sailing._fixed
        def running(f, name, *args):
            if name == 'active-arrow': return True
            if name == 'inactive-face': return False
            return fixed(f, name, *args)
        with patch.object(c.fishing, 'blue_fill', return_value=False), \
             patch.object(c.sailing, '_fixed', side_effect=running):
            self.assertIsNotNone(c._progressed_route(frame))
            self.assertIsNone(c.screen(frame))
            with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
                for action in ('stop', 'open', 'start'):
                    self.assertFalse(c.validate_sailing(action, frame))
        self.assertEqual(c.path.read_bytes(), raw)
        self.assertEqual(self.inputs, [])

    def test_no_fill_and_incomplete_battle_cannot_retire_old_state(self):
        c, raw = self.expired_restore()
        with patch.object(c.fishing, 'blue_fill', return_value=False), \
             patch.object(c.fishing, 'real_battle', return_value=False), \
             patch.object(c.fishing, 'cast_preparation', return_value=False):
            self.assertIsNone(c._progressed_route(self.frame)['fishing_evidence'])
            for now in (100, 102, 104):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=now):
                    self.assertFalse(c._resolve_expired(self.frame))
        self.assertEqual(c.path.read_bytes(), raw)
        self.assertEqual(self.inputs, [])

    def test_three_actual_catches_retire_only_obsolete_rank_and_preserve_failure(self):
        c, _ = self.expired_restore()
        c.journal['rank'] = 38
        c._save()
        raw = c.path.read_bytes()
        def image(name):
            return np.asarray(Image.open(ROOT/'tests/fixtures/meter-route-finalization'/name).convert('RGB'))
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for now, name, pending in ((100,'catch-2.png',True),
                    (101,'preparation.png',True), (102,'catch-3.png',True),
                    (103,'preparation.png',True), (104,'catch-2.png',False)):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=now):
                    self.assertEqual(c.step(image(name)), pending)
                if pending: self.assertEqual(c.path.read_bytes(), raw)
        self.assertFalse(c.journal['resolution']['calibration_completed'])
        self.assertEqual(c.journal['resolution']['fishing_evidence'], 'catch')
        self.assertEqual(Path(c.journal['resolution']['archive']).read_bytes(), raw)
        self.assertEqual(c.journal['cycles'], 10)
        self.assertEqual(c.journal['deadline'], 990)
        self.assertEqual(c.journal['restore_retry']['deadline'], 999)
        self.assertEqual(self.inputs, [])

    def test_catch_probe_rejects_preparation_sleep_hook_and_crossed_weapons(self):
        c, raw = self.expired_restore()
        paths = [ROOT/'tests/fixtures/meter-route-finalization/preparation.png']
        paths += [ROOT/'tests/fixtures/fishing-meter'/name for name in
                  ('sleep.png','fill-empty.png','fill-hook.png','filled-combat.png')]
        for path in paths:
            self.assertFalse(c._catch_result(np.asarray(Image.open(path).convert('RGB'))), path.name)
        self.assertEqual(c.path.read_bytes(), raw)
        self.assertEqual(self.inputs, [])

    def test_context_alone_never_finishes_and_final_frame_must_prove_fishing(self):
        c, _ = self.expired_restore()
        c.journal['rank'] = 38
        c._save()
        raw = c.path.read_bytes()
        prep = np.asarray(Image.open(ROOT/'tests/fixtures/meter-route-finalization/preparation.png').convert('RGB'))
        catch = np.asarray(Image.open(ROOT/'tests/fixtures/meter-route-finalization/catch-2.png').convert('RGB'))
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for now in (100,101.5,103,104):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=now):
                    self.assertTrue(c.step(prep.copy()))
                self.assertEqual(c.path.read_bytes(), raw)
            with patch('drift.tasks.meter_calibration.time.monotonic', return_value=104.5):
                self.assertFalse(c.step(catch))
        self.assertFalse(c.journal['resolution']['calibration_completed'])
        self.assertEqual(self.inputs, [])


if __name__ == '__main__':
    unittest.main()
