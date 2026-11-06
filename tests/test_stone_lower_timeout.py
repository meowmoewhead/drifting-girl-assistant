"""Actual lowered-gate idle frame and task-scoped timeout continuation."""
from copy import deepcopy
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from drift.runtime import Clicker
from drift.tasks.frame_runtime import FrameRuntime
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as base


class LoweredCastTimeoutTests(unittest.TestCase):
    setUpClass = classmethod(base.StoneFarmTests.setUpClass.__func__)
    setUp = base.StoneFarmTests.setUp
    tearDown = base.StoneFarmTests.tearDown
    act = base.StoneFarmTests.act

    def expired_cast(self):
        frame = base.image('tests/fixtures/stone-lower-timeout/after-cast.png')
        self.w._begin(dict(stage=9, rank=39, menu=False, completion=False, route=True))
        self.w.phase = 'first_verify'
        self.w.journal.update(
            phase='first_verify', phase_deadline=self.now-1,
            first_observation_deadline=self.now-1, first_deadline=self.now-30,
            verified_gate_stage=8, entered_verified_at=self.now-50,
            first_selected_rank=39, first_signature=self.w.signature(frame),
            attempts={'enter-confirm': 1, 'first-select': 1, 'first-cast': 1},
            issued_inputs=[dict(kind=k, input_id=k) for k in
                           ('enter-confirm', 'first-select', 'first-cast')])
        self.w._save()
        return frame

    def test_real_controller_exposes_the_configured_schedule_interval(self):
        self.assertEqual(self.w.interval, 2)
        other = StoneFarm(base.ROOT, self.reader, self.act, lambda _: None,
                          interval=7, state_path=self.path)
        self.assertEqual(other.interval, 7)
        self.assertEqual(self.actions, [])

    def test_real_expired_gray_hook_archives_without_credit_input_or_renewal(self):
        frame = self.expired_cast()
        old = deepcopy(self.w.journal)
        raw = self.path.read_bytes()
        self.assertIsNone(self.w._first_result_screen(frame))
        self.assertTrue(self.w.retire_retryable(frame))
        new = self.w.journal
        self.assertFalse(self.w.active)
        self.assertEqual(Path(new['retirement']['archive']).read_bytes(), raw)
        self.assertEqual(new['retirement']['previous_phase'], 'first_verify')
        for key in old.keys()-{'pending', 'phase'}:
            self.assertEqual(new[key], old[key], key)
        self.assertNotIn('first_cast_verified_at', new)
        self.assertNotIn('finished_at', new)
        self.assertEqual(self.actions, [])

    def test_missing_receipt_selection_or_verified_entry_keeps_ownership(self):
        frame = self.expired_cast()
        old = deepcopy(self.w.journal)
        for changes in (dict(issued_inputs=[]), dict(first_selected_rank=38),
                        dict(verified_gate_stage=9), dict(entered_verified_at=None),
                        dict(attempts={'first-cast': 0})):
            with self.subTest(changes=changes):
                self.w.journal = {**old, **changes}
                self.w._save()
                raw = self.path.read_bytes()
                self.assertFalse(self.w.retire_retryable(frame.copy()))
                self.assertEqual(self.path.read_bytes(), raw)

    def test_running_wrong_world_location_dark_and_unknown_frames_cannot_retire(self):
        frame = self.expired_cast()
        raw = self.path.read_bytes()
        wrong_world = frame.copy(); wrong_world[45:72, 13:68] = 0
        wrong_rank = frame.copy(); wrong_rank[9:60, 70:137] = 0
        wrong_location = frame.copy(); wrong_location[12:34, 286:434] = 255
        for bad in (frame//2, wrong_world, wrong_rank, wrong_location):
            with self.subTest():
                self.assertFalse(self.w.retire_retryable(bad))
                self.assertEqual(self.path.read_bytes(), raw)
        with patch.object(self.w, '_off_icon', return_value=False):
            self.assertFalse(self.w.retire_retryable(frame))
        self.assertEqual(self.actions, [])

    def test_unexpired_and_hash_changed_record_are_preserved(self):
        frame = self.expired_cast()
        self.w.journal['phase_deadline'] = self.now+10
        self.w.journal['first_observation_deadline'] = self.now+10
        self.w._save()
        self.assertFalse(self.w.retire_retryable(frame))
        self.w.journal['phase_deadline'] = self.now-1
        self.w._save()
        self.path.write_text(json.dumps(dict(self.w.journal, concurrent=True)), encoding='utf-8')
        raw = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, '已变化'):
            self.w.retire_retryable(frame)
        self.assertEqual(self.path.read_bytes(), raw)

    def test_real_controller_timeout_keeps_full_runner_and_other_tasks_running(self):
        frame = self.expired_cast()
        root = self.path.parent
        config = dict(serial='fake', resolution=[720, 1280], poll_interval=.1,
                      max_consecutive_clicks=3, adb_path='never-used', rules=[
                          dict(id='driftinggirl_stone_farm', interval=2, enabled=True),
                          dict(id='driftinggirl_equipment_upgrade', interval=900, enabled=True)])
        (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
        class Device:
            def __init__(self, config, stop):
                self.stop, self.reads = stop, 0
                self.io_lock = threading.RLock()
            def screenshot(self):
                self.reads += 1
                if self.reads > 5: raise InterruptedError('end of bounded replay')
                return frame.copy()
            def click(self, button): raise AssertionError('No input allowed')
            def touch_fast(self, *args, **kwargs): raise AssertionError('No input allowed')
            def stuck_record_add(self, button): pass
        app = Clicker(root, Device); app.digit_reader = self.reader
        other = Mock(return_value=False)
        equipment = SimpleNamespace(active=False, cycles=0, panel=lambda _: False,
                                    step=other, status=lambda: {})
        try:
            with patch('app.StoneFarm', return_value=self.w), \
                 patch('app.EquipmentUpgrade', return_value=equipment), \
                 patch.object(self.w, 'retire_expired_on_start', return_value=False), \
                 patch('drift.runtime.CruiseSetup.step', return_value=False), \
                 patch.object(FrameRuntime, 'start_engine'), \
                 patch.object(FrameRuntime, 'wait', return_value=False), \
                 patch('drift.runtime.scan_rules', return_value=(None, False, False)):
                app.run([(r, None, None) for r in config['rules']], False)
            self.assertIsNone(app.state['last_error'],
                              (root/'logs/clicker.log').read_text(encoding='utf-8')[-6500:])
            self.assertFalse(self.w.active)
            self.assertGreaterEqual(other.call_count, 2)
            self.assertEqual(app.task_schedule.entries['stone_farm']['reason'], 'retry')
            self.assertEqual(self.actions, [])
        finally:
            for handler in list(app.log.handlers):
                handler.close(); app.log.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
