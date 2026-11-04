"""Task-scoped failures must preserve journals and the real runner's loop."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from drift.core.exceptions import RetryableStage
from drift.runtime import Clicker
from drift.tasks.frame_runtime import FrameRuntime
from drift.tasks.schedule import TaskSchedule
from drift.tasks.stone_farm import StoneFarm


class RetryableMetadataTests(unittest.TestCase):
    def test_new_cast_window_wake_uses_fixed_cast_bound_without_renewing_selection(self):
        c=object.__new__(StoneFarm)
        c.phase='first_idle'
        c.journal=dict(deadline=300.,first_deadline=90.,first_cast_deadline=200.)
        c._first_idle=Mock(return_value='same-actual-task')
        c._confirmed=Mock(return_value=True)
        c.fishing=SimpleNamespace(idle_icon=lambda _: 'sleep')
        c._click=Mock(return_value=True);c._save=Mock()
        with patch('drift.tasks.stone_farm.time.time',return_value=100):
            self.assertTrue(c._first_step(object()))
        self.assertEqual(c.journal['phase_deadline'],115.)
        self.assertEqual(c.journal['first_deadline'],90.)
        self.assertEqual(c.journal['first_cast_deadline'],200.)
        c.journal.pop('first_cast_deadline')
        with patch('drift.tasks.stone_farm.time.time',return_value=100),self.assertRaises(RuntimeError):
            c._first_step(object())

    def test_untagged_exception_defaults_do_not_authorize_task_continuation(self):
        error = RetryableStage('unowned stage expired')
        self.assertIsNone(error.task)
        self.assertIsNone(error.retry_seconds)
        self.assertEqual(str(error), 'unowned stage expired')

    def test_tagged_exception_preserves_task_backoff_and_message(self):
        error = RetryableStage('stone stage expired', task='stone_farm',
                               retry_seconds=1800)
        self.assertEqual(error.task, 'stone_farm')
        self.assertEqual(error.retry_seconds, 1800)
        self.assertEqual(str(error), 'stone stage expired')

    def test_exception_hierarchy_does_not_classify_ordinary_runtime_error_as_retryable(self):
        self.assertIsInstance(RetryableStage('expired'), RuntimeError)
        self.assertNotIsInstance(RuntimeError('fault'), RetryableStage)

    def test_expired_stone_deadline_identifies_task_and_preserves_journal(self):
        from drift.tasks.stone_farm import StoneFarm

        controller = object.__new__(StoneFarm)
        controller.journal = dict(pending=True, phase_deadline=90., deadline=520.,
                                  attempts={'route-start': 1})
        controller.phase = 'route_running'
        controller.retry = None
        controller.input_allowed = Mock(return_value=False)
        controller._first_close_diagnostic = Mock(return_value='')
        controller._save = Mock()

        with self.assertRaises(RetryableStage) as raised:
            controller.check_deadline()

        self.assertEqual(raised.exception.task, 'stone_farm')
        self.assertEqual(raised.exception.retry_seconds, 1800)
        self.assertEqual(controller.journal,
                         dict(pending=True, phase_deadline=90., deadline=520.,
                              attempts={'route-start': 1}))
        controller._save.assert_not_called()


class RetryableRunnerTests(unittest.TestCase):
    def replay(self, error, *, safe=True, origin='deadline'):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = dict(serial='fake', resolution=[64, 96], poll_interval=.1,
                          max_consecutive_clicks=3, adb_path='never-used', rules=[
                              dict(id='driftinggirl_stone_farm', interval=2, enabled=True),
                              dict(id='driftinggirl_equipment_upgrade', interval=900, enabled=True)])
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            frame = np.zeros((96, 64, 3), dtype=np.uint8)
            class Device:
                def __init__(self, config, stop):
                    self.image = None
                    self.stop = stop
                    self.reads = 0
                def screenshot(self):
                    self.reads += 1
                    if self.reads > 5:
                        raise InterruptedError('end of offline replay')
                    self.image = frame.copy() + self.reads
                    return self.image
                def click(self, button):
                    raise AssertionError('No device input is allowed')
                def stuck_record_add(self, button):
                    pass
            stone = SimpleNamespace(active=True, phase='route_running', interval=2,
                                    defense=SimpleNamespace(scan=False))
            stone.status = lambda: dict(pending=stone.active, phase=stone.phase)
            def fail():
                raise error
            stone.check_deadline = fail if origin == 'deadline' else lambda: None
            stone.step = Mock(side_effect=lambda _: fail())
            def retire(_):
                if not safe:
                    return False
                stone.active, stone.phase = False, 'idle'
                return True
            stone.retire_retryable = Mock(side_effect=retire)
            stone.defense.encounter_rank = lambda _: None
            stone.defense.owns = lambda *_: False
            stone.defense.step = lambda *_: False
            stone.defense.status = lambda: {}
            app = Clicker(root, Device)
            app.digit_reader = SimpleNamespace()
            other_task = Mock(return_value=False)
            equipment = SimpleNamespace(active=False, cycles=0, panel=lambda _: False,
                                        step=other_task, status=lambda: {})
            try:
                with patch('app.StoneFarm', return_value=stone), \
                     patch('app.EquipmentUpgrade', return_value=equipment), \
                     patch.object(FrameRuntime, 'start_engine'), \
                     patch.object(FrameRuntime, 'wait', return_value=False), \
                     patch('drift.runtime.scan_rules', return_value=(None, False, False)), \
                     patch.object(app.task_schedule, 'finish', wraps=app.task_schedule.finish) as finish:
                    app.run([(r, None, None) for r in config['rules']], False)
                if other_task.called:
                    # A release capture must also be the frame handed to the
                    # next task, never the now-stale outer-loop frame.
                    self.assertEqual(int(other_task.call_args_list[0].args[0][0,0,0]), 2)
                return (deepcopy(app.state), other_task.call_count, stone,
                        finish.call_args_list, deepcopy(app.task_schedule.entries),
                        app.task_schedule.ready('stone_farm'),
                        list((root/'logs').glob('error-*.json')))
            finally:
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_tagged_deadline_and_step_keep_loop_and_other_tasks_running(self):
        for origin in ('deadline', 'step'):
            with self.subTest(origin=origin):
                state, calls, stone, finishes, entries, ready, errors = self.replay(
                    RetryableStage('expired', task='stone_farm', retry_seconds=1800), origin=origin)
                self.assertIsNone(state['last_error'])
                self.assertGreaterEqual(calls, 2)
                stone.retire_retryable.assert_called_once()
                self.assertFalse(stone.active)
                self.assertEqual(len(finishes), 1)
                self.assertEqual(finishes[0].kwargs, dict(success=False, retry_seconds=1800))
                self.assertEqual(entries['stone_farm']['reason'], 'retry')
                self.assertFalse(ready)
                self.assertEqual(errors, [])

    def test_untagged_unknown_and_unsafe_keep_clean_stop(self):
        cases = [(RetryableStage('untagged'), True),
                 (RetryableStage('unknown', task='unknown', retry_seconds=1800), True),
                 (RetryableStage('unsafe', task='stone_farm', retry_seconds=1800), False),
                 (RetryableStage('missing backoff', task='stone_farm'), True),
                 (RetryableStage('invalid backoff', task='stone_farm', retry_seconds=float('nan')), True)]
        for error, safe in cases:
            with self.subTest(error=str(error)):
                state, calls, _, finishes, _, _, errors = self.replay(error, safe=safe)
                self.assertEqual(state['stop_reason'], 'retryable')
                self.assertEqual(calls, 0)
                self.assertEqual(finishes, [])
                self.assertEqual(errors, [])

    def test_ordinary_runtime_error_uses_failure_path(self):
        state, calls, stone, finishes, _, _, errors = self.replay(RuntimeError('real fault'))
        self.assertEqual(state['last_error'], 'real fault')
        self.assertNotEqual(state.get('stop_reason'), 'retryable')
        self.assertEqual(calls, 0)
        stone.retire_retryable.assert_not_called()
        self.assertEqual(finishes, [])
        self.assertEqual(len(errors), 1)


class RetryableJournalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.controller = object.__new__(StoneFarm)
        self.controller.path = self.root/'stone_farm_state.json'
        self.controller.journal = dict(version=1, pending=True, phase='route_running',
            deadline=520., phase_deadline=90., from_stage=9, to_stage=8, rank=39,
            entry_rank=39, attempts={'route-start': 1}, issued_inputs=[{'kind': 'route-start'}],
            cycles=3, handoffs=2)
        self.controller.path.write_text(json.dumps(self.controller.journal), encoding='utf-8')
        self.raw = self.controller.path.read_bytes()
        self.controller.phase, self.controller.retry = 'route_running', None
        self.controller.screen = Mock(return_value=dict(stage=8, rank=39, menu=False,
                                                        completion=False, route=True))
        self.controller.fishing = SimpleNamespace(_main_visible=Mock(return_value=True))

    def test_retirement_preserves_failed_bytes_budgets_and_counts(self):
        old = deepcopy(self.controller.journal)
        with patch('drift.tasks.stone_farm.time.time', return_value=100):
            self.assertTrue(self.controller.retire_retryable(object()))
        record = self.controller.journal
        self.assertEqual(Path(record['retirement']['archive']).read_bytes(), self.raw)
        self.assertEqual(record['retirement']['source'], 'retryable_stage')
        self.assertFalse(record['pending'])
        self.assertFalse(record['retirement']['released'])
        for key in old.keys()-{'pending', 'phase'}:
            self.assertEqual(record[key], old[key], key)
        self.assertNotIn('finished_at', record)

    def test_unknown_modal_phase_and_approved_retry_do_not_release(self):
        for changes in (dict(phase='return_confirm'), dict(retry={'expires_at': 200})):
            with self.subTest(changes=changes), patch.multiple(self.controller, **changes):
                self.assertFalse(self.controller.retire_retryable(object()))
        for screen in (None, dict(stage=8, rank=39, menu=True, completion=False, route=None),
                       dict(stage=9, rank=39, menu=False, completion=False, route=True)):
            with patch.object(self.controller, 'screen', return_value=screen):
                self.assertFalse(self.controller.retire_retryable(object()))
        self.assertEqual(self.controller.path.read_bytes(), self.raw)

    def test_archive_conflict_does_not_replace_pending_record(self):
        digest = hashlib.sha256(self.raw).hexdigest()
        archive = self.root/'logs/stone-farm-retired'/f'{digest}.json'
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b'conflict')
        with patch('drift.tasks.stone_farm.time.time', return_value=100), self.assertRaises(RuntimeError):
            self.controller.retire_retryable(object())
        self.assertEqual(self.controller.path.read_bytes(), self.raw)
        self.assertTrue(self.controller.active)

    def test_return_deadline_retires_each_stopped_return_phase_without_renewing_inputs(self):
        for phase in ('stop_fishing','return','return_confirm'):
            with self.subTest(phase=phase):
                c=self.controller
                c.phase=phase
                c.journal.update(pending=True,phase=phase,return_deadline=90.,phase_deadline=280.,
                                 attempts={'stop':1,'handoff-cancel':1})
                c.path.write_text(json.dumps(c.journal),encoding='utf-8')
                raw=c.path.read_bytes();old=deepcopy(c.journal)
                c.screen.return_value=dict(stage=9,rank=39,menu=False,completion=False,route=False)
                c._off_icon=Mock(return_value=True)
                with patch('drift.tasks.stone_farm.time.time',return_value=100):
                    self.assertTrue(c.retire_retryable(object()))
                self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),raw)
                for key in old.keys()-{'pending','phase','retirement'}:
                    self.assertEqual(c.journal[key],old[key],key)
                self.assertFalse(c.active)
                self.assertNotIn('finished_at',c.journal)

    def test_expired_return_keeps_ownership_on_running_wrong_rank_or_unknown_main(self):
        c=self.controller
        c.phase='stop_fishing'
        c.journal.update(phase='stop_fishing',return_deadline=90.,phase_deadline=280.)
        c.path.write_text(json.dumps(c.journal),encoding='utf-8');raw=c.path.read_bytes()
        c._off_icon=Mock(return_value=True)
        for screen in (None,dict(stage=9,rank=39,menu=False,completion=False,route=True),
                       dict(stage=9,rank=38,menu=False,completion=False,route=False),
                       dict(stage=9,rank=39,menu=True,completion=False,route=False)):
            with self.subTest(screen=screen),patch.object(c,'screen',return_value=screen),\
                 patch('drift.tasks.stone_farm.time.time',return_value=100):
                self.assertFalse(c.retire_retryable(object()))
        c.screen.return_value=dict(stage=9,rank=39,menu=False,completion=False,route=False)
        c.fishing._main_visible.return_value=False
        with patch('drift.tasks.stone_farm.time.time',return_value=100):
            self.assertFalse(c.retire_retryable(object()))
        self.assertEqual(c.path.read_bytes(),raw)

    def test_realtime_failure_backoff_survives_reload_and_expires(self):
        now = datetime(2026, 10, 27, tzinfo=timezone.utc)
        schedule = TaskSchedule(self.root, 'fake', clock=lambda: now)
        self.assertTrue(schedule.ready('stone_farm'))
        schedule.finish('stone_farm', 2, success=False, retry_seconds=1800)
        restored = TaskSchedule(self.root, 'fake', clock=lambda: now)
        self.assertFalse(restored.ready('stone_farm'))
        self.assertTrue(restored.ready('parallel_world'))
        now += timedelta(seconds=1799)
        self.assertFalse(restored.ready('stone_farm'))
        now += timedelta(seconds=1)
        self.assertTrue(restored.ready('stone_farm'))


if __name__ == '__main__':
    unittest.main()
