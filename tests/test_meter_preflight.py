"""Keep a never-issued calibration from failing the whole native route."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from drift.perception.digit_ocr import DigitReader
from drift.tasks.meter_calibration import MeterCalibration

ROOT = Path(__file__).resolve().parents[1]


class MeterPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.frames = {name: np.asarray(Image.open(ROOT/'tests/fixtures/meter-preflight'/
            (name+'.png')).convert('RGB')) for name in ('timeout-result','sapphire-r51','raid-r50','task-preparation')}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.inputs, self.events = [], []
        path = ROOT/'meter_calibration_state.json'
        exists = Path.is_file
        with patch.object(Path, 'is_file', lambda p: False if p == path else exists(p)):
            self.c = MeterCalibration(ROOT, self.reader, lambda *a: self.inputs.append(a) or False,
                lambda *a: self.inputs.append(a) or True, self.events.append)
        c = self.c
        c.path = Path(self.temp.name)/'meter_calibration_state.json'
        c.journal.update(pending=True, rank=39, stage=8, restore_running=True, deadline=1090,
            value=1705, high=1810, target=1761, trigger_player=1761, attempts={},
            sailing_attempts={}, proof=None, resume_requested=False, stopped_context=None,
            restore_retry=None, cycles=24,
            stage_strokes=c.world._stage_strokes(self.frames['timeout-result']).hex())
        c.world.training.state = dict(mode='farming', rank=39, sailing_enabled=True,
            sailing_sleep_recovery=True, sailing_meter_adjustment=True)
        c.phase, c.off_verified, c.cycles, c.world.rank = 'stop', False, 24, 39
        c._save()

    def tearDown(self):
        self.temp.cleanup()

    def test_real_untouched_timeout_archives_without_inputs_or_success(self):
        c = self.c
        original = c.path.read_bytes()
        with patch('drift.tasks.meter_calibration.time.time', return_value=1100):
            for now in (100,101.5,103):
                with patch('drift.tasks.meter_calibration.time.monotonic', return_value=now):
                    c.step(self.frames['timeout-result'].copy())
        self.assertFalse(c.active)
        result = c.journal['resolution']
        self.assertEqual(result['kind'], 'deferred_before_any_input')
        self.assertFalse(result['calibration_completed'])
        self.assertEqual(Path(result['archive']).read_bytes(), original)
        self.assertEqual((c.journal['deadline'],c.cycles,c.fishing.confirmed_starts),(1090,24,0))
        self.assertEqual((c.journal['attempts'],c.journal['sailing_attempts']), ({},{}))
        self.assertEqual(c.journal['next_check'],1100+c.interval)
        self.assertEqual(self.inputs,[])

    def test_pending_stop_can_run_before_optional_combat_and_only_stops_route(self):
        c = self.c
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            self.assertTrue(c.pre_battle_step(self.frames['timeout-result'].copy()))
        self.assertEqual([a[0] for a in self.inputs],['stop'])
        self.assertFalse(c.off_verified)
        self.assertEqual(c.phase,'stop')
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual(c.journal['attempts'],{})

    def test_priority_stop_never_takes_task_boss_or_unknown_frame(self):
        c = self.c
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            with patch.object(c.world.training,'boss_mode',return_value=True):
                self.assertFalse(c.pre_battle_step(self.frames['timeout-result']))
            self.assertFalse(c.pre_battle_step(np.zeros((1280,720,3),dtype='uint8')))
        self.assertEqual(self.inputs,[])

    def test_resource_r51_and_raid_r50_never_authorize_or_replace_parallel_values(self):
        c = self.c
        before = deepcopy(c.journal)
        with patch('drift.tasks.meter_calibration.time.time', return_value=1000):
            for name in ('sapphire-r51','raid-r50'):
                frame = self.frames[name]
                self.assertIsNone(c.screen(frame))
                self.assertFalse(c.pre_battle_step(frame))
                self.assertFalse(c.validate_sailing('stop',frame))
        self.assertEqual(c.journal,before)
        self.assertEqual(self.inputs,[])

    def test_used_input_or_saved_stop_proof_cannot_use_zero_input_deferral(self):
        c = self.c
        initial = deepcopy(c.journal)
        for field,value in (('attempts',{'cancel':1}),('attempts',{'drag':1}),
                ('attempts',{'start':1}),('sailing_attempts',{'stop':1}),
                ('sailing_attempts',{'open':1}),('resume_requested',True),
                ('proof','post_release_fishing'),('stopped_context','already-stopped')):
            c.journal = deepcopy(initial); c.phase = 'stop'
            with self.subTest(field=field,value=value):
                c.journal[field] = value
                c._save()
                original = c.path.read_bytes()
                self.assertFalse(c._untouched_route_wait())
                with patch('drift.tasks.meter_calibration.time.time',return_value=1100), \
                     patch.object(c,'_resolve_expired',return_value=False):
                    self.assertFalse(c.step(self.frames['timeout-result']))
                self.assertTrue(c.awaiting_recovery)
                self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),original)
                self.assertEqual(c.journal[field],value)
        self.assertEqual(self.inputs,[])

    def test_expired_observation_never_allows_stop_or_extends_input_deadline(self):
        c = self.c
        with patch('drift.tasks.meter_calibration.time.time', return_value=1100), \
             patch('drift.tasks.meter_calibration.time.monotonic', return_value=100):
            self.assertFalse(c.pre_battle_step(self.frames['timeout-result']))
            self.assertFalse(c.validate_sailing('stop',self.frames['timeout-result']))
            blank = np.zeros((1280,720,3),dtype='uint8')
            for now in (100,110):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    self.assertTrue(c.step(blank))
            with patch('drift.tasks.meter_calibration.time.monotonic',return_value=115):
                self.assertFalse(c.step(blank))
        self.assertEqual(c.journal['deadline'],1090)
        self.assertFalse(c.active)
        self.assertTrue(c.awaiting_recovery)
        self.assertEqual(self.inputs,[])

    def test_obscured_frame_resets_deferral_consensus(self):
        c = self.c
        frame = self.frames['timeout-result']
        with patch('drift.tasks.meter_calibration.time.time',return_value=1100):
            for now,image in ((100,frame),(101.5,np.zeros_like(frame)),(103,frame),(104.5,frame)):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    c.step(image.copy())
                self.assertTrue(c.active)
            with patch('drift.tasks.meter_calibration.time.monotonic',return_value=106):
                c.step(frame.copy())
        self.assertFalse(c.active)
        self.assertEqual(self.inputs,[])

    def test_priority_step_does_not_start_new_work_or_cancel_drag_hold(self):
        c = self.c
        with patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            for phase in ('idle','prepare','open','menu','running'):
                c.phase = phase
                self.assertFalse(c.pre_battle_step(self.frames['timeout-result']))
        self.assertEqual(self.inputs,[])

    def test_real_task_frame_can_only_retire_untouched_wait_before_combat(self):
        c = self.c
        frame = self.frames['task-preparation']
        original = c.path.read_bytes()
        self.assertTrue(c.world.training.boss_mode(frame))
        self.assertIsNone(c.screen(frame))
        self.assertIsNotNone(c.screen(frame,observe_untouched=True))
        with patch('drift.tasks.meter_calibration.time.time',return_value=1100):
            for now in (100,101.5,103):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    # False leaves the same frame available to combat.
                    self.assertFalse(c.pre_battle_step(frame.copy()))
        self.assertFalse(c.active)
        self.assertEqual(c.journal['resolution']['kind'],'deferred_before_any_input')
        self.assertEqual(Path(c.journal['resolution']['archive']).read_bytes(),original)
        self.assertEqual((c.journal['deadline'],c.cycles),(1090,24))
        self.assertEqual(self.inputs,[])

    def test_readonly_task_route_never_authorizes_any_input_even_before_expiry(self):
        c = self.c
        frame = self.frames['task-preparation']
        with patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            for action in ('stop','open','start'):
                self.assertFalse(c.validate_sailing(action,frame))
            self.assertFalse(c.meter_frame_allowed(frame))
            self.assertFalse(c.pre_battle_step(frame))
        self.assertEqual(self.inputs,[])

    def test_readonly_task_route_still_rejects_used_limits_changed_rank_and_resource_world(self):
        c = self.c
        frame = self.frames['task-preparation']
        c.journal['sailing_attempts']={'stop':1}
        self.assertIsNone(c.screen(frame,observe_untouched=True))
        c.journal['sailing_attempts']={}
        c.journal['rank']=38
        self.assertIsNone(c.screen(frame,observe_untouched=True))
        c.journal['rank']=39
        for name in ('sapphire-r51','raid-r50'):
            self.assertIsNone(c.screen(self.frames[name],observe_untouched=True))
        self.assertEqual(self.inputs,[])


if __name__ == '__main__':
    unittest.main()
