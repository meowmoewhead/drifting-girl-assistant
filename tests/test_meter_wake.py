"""Bounded wake-before-hold workflow and real failed-Zzz frame guards."""
from contextlib import ExitStack
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import cv2

from drift.tasks.meter_calibration import MeterCalibration
from drift.perception.digit_ocr import DigitReader
from drift.tasks.meter_limits import MeterWorkflowLimit
from drift.core.persistence import ReliablePath

ROOT = Path(__file__).resolve().parents[1]


class MeterWakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)
        cls.frame = cv2.cvtColor(cv2.imread(str(ROOT / 'tests/fixtures/meter-wake/zzz.png')), cv2.COLOR_BGR2RGB)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.actions = []
        target = ROOT / 'meter_calibration_state.json'
        exists = Path.is_file
        with patch.object(Path, 'is_file', lambda p: False if p == target else exists(p)):
            self.c = MeterCalibration(ROOT, self.reader,
                lambda *a: self.actions.append(a) or True, lambda *a: False, lambda *a: None)
        self.c.path = Path(self.tmp.name) / target.name
        self.m = self.c.fishing
        self.c.world.training.state.update(mode='farming', rank=37)
        self.c.world.rank = 37
        self.data = dict(value=1388, player=1390, goal=1390, low=1195, high=1421,
                         adaptive=True, context='same', x=245, y=958)
        self.signature = dict(value=1388, context='same', box=(210, 939, 280, 977),
                              digit_strokes='digits', world_strokes='world')

    def mocked_scene(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        for owner, name, value in (
            (self.m, 'world_matches', True), (self.m.training, 'boss_mode', False),
            (self.m, 'at_port', False), (self.m, '_main_visible', True),
            (self.m, 'busy', False), (self.m, 'blue_fill', False),
            (self.m, 'real_battle', False), (self.m, 'cancel_button', False),
            (self.m, 'post_start_evidence', False), (self.m, '_start_identity', ('same', b'world')),
            (self.m, 'read', self.data), (self.m, 'cast_data', self.signature),
            (self.m, 'observe', None)):
            stack.enter_context(patch.object(owner, name, return_value=value))
        self.icon = stack.enter_context(patch.object(self.m, 'idle_icon', return_value='sleep'))

    def tick(self, now):
        with patch('drift.tasks.fishing_meter.time.monotonic', return_value=now):
            return self.m.step(self.frame.copy())

    def wake(self):
        self.mocked_scene()
        for now in (100, 101, 104):
            self.tick(now)
        self.assertEqual([a[0] for a in self.actions], ['wake'])
        self.assertEqual((self.m.phase, self.m.wakes, self.m.starts), ('waking', 1, 0))

    def test_zzz_is_woken_once_before_fresh_hook_consensus_and_hold(self):
        self.wake()
        self.tick(105)
        self.icon.return_value = 'hook'
        self.tick(106)
        self.tick(107)
        self.assertEqual(len(self.actions), 1)
        self.tick(110)
        self.assertEqual([a[0] for a in self.actions], ['wake', 'start'])
        self.assertEqual(self.m.phase, 'starting')

    def test_direct_fishing_after_wake_is_verified_without_a_hold(self):
        self.wake()
        with patch.object(self.m, 'post_start_evidence', return_value=True):
            for now in (105, 106.5, 108):
                self.tick(now)
        self.assertEqual([a[0] for a in self.actions], ['wake'])
        self.assertEqual(self.m.start_proof, 'post_release_fishing')
        self.assertEqual(self.m.confirmed_starts, 1)

    def test_persistent_zzz_exhausts_wake_wait_without_a_second_input(self):
        self.wake()
        self.tick(110)
        with self.assertRaises(MeterWorkflowLimit):
            self.tick(120)
        self.assertEqual([a[0] for a in self.actions], ['wake'])

    def test_a_spent_hold_can_never_be_followed_by_a_wake_retry(self):
        self.mocked_scene()
        self.m.starts = 1
        with patch('drift.tasks.fishing_meter.time.monotonic', return_value=100):
            self.m.arm_start_observation(self.frame)
        self.tick(110)
        with self.assertRaises(MeterWorkflowLimit):
            self.tick(131)
        self.assertEqual(self.actions, [])

    def test_wake_protects_new_fill_from_prepare_cancellation(self):
        self.wake()
        with patch.object(self.m, 'blue_fill', return_value=True), \
             patch.object(self.m, '_saved_low_correction', return_value=True), \
             patch.object(self.m, 'step', return_value=True) as step:
            self.m.prepare(self.frame.copy())
        step.assert_called_once()
        self.assertEqual([a[0] for a in self.actions], ['wake'])

    def test_real_failed_frame_can_wake_without_repeating_numeric_ocr(self):
        self.m.current = 1388
        self.assertEqual(self.m.idle_icon(self.frame), 'sleep')
        self.assertFalse(self.m.blue_fill(self.frame))
        expected = self.m.cast_data(self.frame)
        self.assertIsNotNone(expected)
        with patch.object(self.m, 'read', side_effect=AssertionError('wake must use fixed strokes')):
            self.assertTrue(self.m.validate('wake', self.frame.copy(), expected))
        self.assertEqual(self.actions, [])

    def test_changed_strokes_or_battle_boss_port_and_used_inputs_reject_wake(self):
        self.m.current = 1388
        expected = self.m.cast_data(self.frame)
        for owner, name, value in (
            (self.m.training, 'boss_mode', True), (self.m, 'at_port', True),
            (self.m, 'busy', True), (self.m, 'real_battle', True),
            (self.m, 'idle_icon', 'hook')):
            with self.subTest(name=name), patch.object(owner, name, return_value=value):
                self.assertFalse(self.m.validate('wake', self.frame.copy(), expected))
        self.assertFalse(self.m.validate('wake', self.frame.copy(), {**expected, 'digit_strokes': 'changed'}))
        for name in ('starts', 'wakes'):
            with patch.object(self.m, name, 1):
                self.assertFalse(self.m.validate('wake', self.frame.copy(), expected))
        self.assertEqual(self.actions, [])

    def test_hook_to_zzz_change_before_hold_rejects_input_without_numeric_ocr(self):
        with patch.object(self.m, 'read', side_effect=AssertionError('state changed before input')):
            self.assertFalse(self.m.validate('start', self.frame.copy(), {**self.data, 'idle_icon': 'hook'}))
        self.assertIsNone(self.m.validated_start)

    def arm_journal(self):
        self.c.journal.update(pending=True, rank=37, stage=10, restore_running=False,
            deadline=1000, attempts={'drag': 1}, route_policy='preserve')
        self.c.phase = 'prepare'

    def test_wake_is_persisted_once_without_changing_deadline_or_drag_count(self):
        self.arm_journal()
        with patch('drift.tasks.meter_calibration.time.time', return_value=900):
            self.assertTrue(self.c.reserve_meter_input('wake'))
            with self.assertRaises(MeterWorkflowLimit):
                self.c.reserve_meter_input('wake')
        saved = json.loads(self.c.path.read_text(encoding='utf-8'))
        self.assertEqual(saved['attempts'], {'drag': 1, 'wake': 1})
        self.assertEqual(saved['deadline'], 1000)
        self.assertEqual(saved['wake_reserved_at'], 900)

    def test_wake_rejected_near_deadline_or_after_prior_start(self):
        self.arm_journal()
        with patch('drift.tasks.meter_calibration.time.time', return_value=971):
            self.assertFalse(self.c.reserve_meter_input('wake'))
        self.c.journal['attempts']['start'] = 1
        with patch('drift.tasks.meter_calibration.time.time', return_value=900):
            self.assertFalse(self.c.reserve_meter_input('wake'))
        self.assertNotIn('wake', self.c.journal['attempts'])
        self.assertEqual(self.c.journal['deadline'], 1000)

    def test_post_wake_fill_cannot_enter_the_cancel_scanner(self):
        self.arm_journal()
        self.c.journal['attempts']['wake'] = 1
        with patch.object(self.m, 'idle_icon', return_value=None), \
             patch.object(self.m, 'cast_data', side_effect=AssertionError('no cancel after wake')):
            self.assertFalse(self.c._observe_cancel_window(self.frame))

    def test_post_wake_restoration_cannot_issue_an_extra_short_cast(self):
        self.arm_journal()
        self.c.phase = 'restore_fishing'
        self.c.journal['attempts'].update(cancel=1, wake=1)
        with patch('drift.tasks.meter_calibration.time.time', return_value=900):
            self.assertFalse(self.c.reserve_meter_input('resume_cast'))
        self.assertNotIn('start', self.c.journal['attempts'])

    def test_restart_keeps_spent_wake_and_original_wake_timestamp(self):
        target = ROOT / 'meter_calibration_state.json'
        saved = dict(version=1, pending=True, rank=37, stage=10, restore_running=False,
                     deadline=1000, attempts={'drag': 1, 'wake': 1}, wake_reserved_at=895,
                     route_policy='preserve', training={'mode':'farming','rank':37})
        from drift.core import persistence
        exists, read = ReliablePath.is_file, persistence.read_bytes
        with patch.object(ReliablePath, 'is_file', lambda p: True if p == target else exists(p)), \
             patch.object(persistence, 'read_bytes', lambda p: json.dumps(saved).encode() if p == target else read(p)), \
             patch('drift.tasks.meter_calibration.time.time', return_value=900), \
             patch('drift.tasks.meter_calibration.time.monotonic', return_value=200):
            restarted = MeterCalibration(ROOT, self.reader, lambda *a: self.actions.append(a), lambda *a: False, lambda *a: None)
        self.assertEqual(restarted.fishing.wakes, 1)
        self.assertEqual(restarted.fishing.phase, 'waking')
        self.assertEqual(restarted.fishing.since, 195)
        self.assertEqual(restarted.journal['deadline'], 1000)
        self.assertEqual(self.actions, [])


if __name__ == '__main__':
    unittest.main()
