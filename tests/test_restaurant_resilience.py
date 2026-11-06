"""Offline regressions for expired restaurant journals and task isolation."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from drift.tasks.restaurant_upgrade import RestaurantUpgrade
from drift.tasks.schedule import TaskSchedule


ROOT = Path(__file__).resolve().parents[1]


class RestaurantResilienceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root/'restaurant_upgrade_state.json'
        self.now = 1000
        self.issued = []

    def controller(self, **changes):
        record = dict(version=1, serial='offline', pending=True, phase='restore_verify',
                      deadline=2000, phase_deadline=1100, cleanup_deadline=1800,
                      failure=dict(reason='执行窗口到期', phase='stop', at=900,
                                   phase_deadline=899),
                      attempts={'stop':1, 'restore-open':1, 'restore-start':1},
                      origin={'route':True}, settings={'sale':True},
                      visited=[], categories=[], skipped=[], upgraded=0, submitted=0)
        record.update(changes)
        self.path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        self.raw = self.path.read_bytes()
        with patch('drift.tasks.restaurant_upgrade.RestaurantVisual'), \
             patch('drift.tasks.restaurant_upgrade.GuildVisual'), \
             patch('drift.tasks.restaurant_upgrade.StoneFarm'), \
             patch('drift.tasks.restaurant_upgrade.BackpackSale'):
            task = RestaurantUpgrade(self.root, None, Mock(), Mock(), 'offline', clock=lambda:self.now)
        task.same_world = Mock(return_value=True)
        task.map_ready = Mock(return_value=False)
        task.settings = Mock(return_value=None)
        task.visual.detail.return_value = False
        task.visual.panel.return_value = False
        task.sailing.menu.return_value = False
        task.sailing.running.return_value = None
        task.fishing._main_visible.return_value = False
        task.view._off_icon.return_value = True
        task._confirmed = Mock(return_value=True)

        def action(request):
            # Match the real input gate. No device or network is constructed.
            if not task.validate(request, object()) or not task.reserve(request):
                return False
            self.issued.append(request)
            task.input_issued(request, {'input_id':'offline'})
            return True

        task.action = Mock(side_effect=action)
        return task

    def assert_retired(self, task, raw, failure=None):
        saved = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertFalse(task.active)
        self.assertEqual(task.phase, 'idle')
        self.assertFalse(saved['pending'])
        self.assertEqual(saved['phase'], 'idle')
        self.assertIs(task.done, False)
        self.assertIn('last_failure', saved)
        self.assertNotIn('failure', saved)
        if failure is not None:
            self.assertEqual(saved['last_failure'], failure)
        archive = self.root/'logs/restaurant-history'/(hashlib.sha256(raw).hexdigest()+'.json')
        self.assertEqual(archive.read_bytes(), raw)
        self.assertEqual(saved['attempts'], json.loads(raw)['attempts'])

    def test_expired_saved_cleanup_is_retired_on_first_step_without_input(self):
        task = self.controller(phase_deadline=999, cleanup_deadline=999)
        failure = deepcopy(task.journal['failure'])
        self.assertFalse(task.step(object()))
        self.assert_retired(task, self.raw, failure)
        task.event.assert_called_once_with('餐品强化：上次执行已过期，已归档并重新排队')
        task.action.assert_not_called()
        self.assertEqual(self.issued, [])
        self.assertFalse(task.retire_expired_on_start())

    def test_startup_expiration_uses_journal_deadline_or_phase_deadline(self):
        for deadlines in (dict(deadline=999, phase_deadline=1100),
                          dict(deadline=2000, phase_deadline=999)):
            with self.subTest(deadlines=deadlines):
                task = self.controller(phase='stop', **deadlines)
                task.journal.pop('cleanup_deadline')
                task.journal.pop('failure')
                task._save()
                raw = self.path.read_bytes()
                self.assertTrue(task.retire_expired_on_start())
                self.assert_retired(task, raw)
                task.action.assert_not_called()

    def test_cleanup_deadline_does_not_restart_or_raise(self):
        task = self.controller()
        task.retire_expired_on_start()
        self.now = 1801
        self.assertFalse(task._begin_cleanup('执行窗口到期'))
        self.assert_retired(task, self.raw)
        self.assertEqual(task.journal['cleanup_deadline'], 1800)
        self.assertTrue(any('安全返回未完成，放弃本轮并保留记录' in c.args[0]
                            and '拒绝未知画面输入' in c.args[0] for c in task.event.call_args_list))

    def test_exhausted_cleanup_and_restore_inputs_abandon_without_replaying(self):
        cases = [('cleanup', 'cleanup-detail-close'), ('cleanup', 'cleanup-panel-close'),
                 ('cleanup', 'cleanup-map-close'), ('cleanup', 'cleanup-settings-close'),
                 ('restore', 'restore-open'), ('restore_verify', 'restore-open'),
                 ('restore_menu', 'restore-start')]
        for phase, kind in cases:
            with self.subTest(phase=phase, kind=kind):
                task = self.controller(phase=phase, attempts={kind:RestaurantUpgrade.input_limit(kind)})
                if kind == 'cleanup-detail-close':
                    task.visual.detail.return_value = True
                    task.visual.point.return_value = (650, 100)
                elif kind == 'cleanup-panel-close':
                    task.visual.panel.return_value = True
                    task.visual.point.return_value = (650, 100)
                elif kind == 'cleanup-map-close':
                    task.map_ready.return_value = True
                elif kind in ('cleanup-settings-close', 'restore-start'):
                    task.sailing.menu.return_value = True
                    task.settings.return_value = task.journal['settings']
                else:
                    task.fishing._main_visible.return_value = True
                    task.sailing.running.return_value = False
                self.assertFalse(task.step(object(), due=False))
                self.assert_retired(task, self.raw)
                self.assertEqual(self.issued, [])
                self.assertTrue(any('安全返回未完成，放弃本轮并保留记录' in c.args[0]
                                    for c in task.event.call_args_list))

    def test_live_restore_timeout_abandons_even_without_cleanup_window(self):
        for phase in ('restore', 'restore_menu', 'restore_verify'):
            with self.subTest(phase=phase):
                task = self.controller(phase=phase)
                task.journal.pop('cleanup_deadline')
                task.journal.pop('failure')
                task._save()
                raw = self.path.read_bytes()
                task.retire_expired_on_start()
                self.now = 1101
                self.assertFalse(task.step(object(), due=False))
                self.assert_retired(task, raw)
                task.action.assert_not_called()
                self.now = 1000

    def test_unexpired_saved_journal_resumes_with_original_budget(self):
        task = self.controller()
        before = deepcopy(task.journal)
        self.assertFalse(task.retire_expired_on_start())
        task.fishing._main_visible.return_value = True
        task.sailing.running.return_value = False
        self.assertTrue(task.step(object(), due=False))
        self.assertTrue(task.active)
        self.assertEqual(task.phase, 'restore_menu')
        self.assertEqual(task.journal['attempts']['restore-open'], 2)
        self.assertEqual(task.journal['attempts']['restore-start'], 1)
        self.assertEqual(task.journal['deadline'], before['deadline'])
        self.assertEqual(task.journal['cleanup_deadline'], before['cleanup_deadline'])
        self.assertEqual([a['kind'] for a in self.issued], ['restore-open'])

    def test_unknown_screens_never_issue_cleanup_or_restore_inputs(self):
        for phase in ('cleanup', 'restore', 'restore_menu', 'restore_verify'):
            for same_world in (False, True):
                with self.subTest(phase=phase, same_world=same_world):
                    task = self.controller(phase=phase)
                    task.same_world.return_value = same_world
                    self.assertTrue(task.step(object(), due=False))
                    self.assertTrue(task.active)
                    task.action.assert_not_called()
                    self.assertEqual(self.issued, [])

    def test_recognized_restore_still_rejects_unregistered_coordinates(self):
        task = self.controller(phase='restore')
        task.fishing._main_visible.return_value = True
        task.sailing.running.return_value = False
        self.assertFalse(task.validate(dict(kind='restore-open', phase='restore', x=999, y=999), object()))
        self.assertTrue(task.validate(dict(kind='restore-open', phase='restore', x=233, y=45), object()))
        self.assertEqual(self.issued, [])

    def test_last_restore_attempt_can_still_be_confirmed_successfully(self):
        task = self.controller(attempts={'restore-open':3, 'restore-start':3})
        task.fishing._main_visible.return_value = True
        task.sailing.running.return_value = True
        task.fishing.blue_fill.return_value = True
        task.step(object(), due=False)
        self.assertFalse(task.active)
        self.assertEqual(task.phase, 'idle')
        task.action.assert_not_called()

    def test_restore_without_confirmation_abandons_within_bounded_window(self):
        # 2026-10-29 线上：航巡已停但恢复确认一直不成功，restore_verify 只受
        # 三小时执行窗口约束，整轮长期占住 owned 阶段，连标尺校准都被挡住。
        task = self.controller()
        task.fishing._main_visible.return_value = True
        task.sailing.running.return_value = None
        task.view._off_icon.return_value = False
        self.assertTrue(task.step(object(), due=False))
        self.assertTrue(task.active)
        self.assertEqual(task.phase, 'restore_verify')
        self.assertEqual(task.journal['restore_deadline'], 1060)
        self.assertEqual(self.issued, [])
        self.now = 1061
        self.assertFalse(task.step(object(), due=False))
        self.assert_retired(task, self.raw)
        self.assertEqual(task.journal['retirement']['reason'], '航巡恢复确认超时')
        self.assertEqual(self.issued, [])
        task.action.assert_not_called()
        self.now = 1000

    def test_archive_failure_preserves_original_pending_record(self):
        task = self.controller()
        with patch('drift.tasks.restaurant_upgrade.archive_bytes', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                task.abandon('task fault')
        self.assertEqual(self.path.read_bytes(), self.raw)
        self.assertTrue(task.active)
        task.action.assert_not_called()

    def test_abandon_before_first_journal_save_preserves_in_memory_evidence(self):
        task = self.controller(pending=False, phase='idle')
        self.path.unlink()
        before = deepcopy(task.journal)
        self.assertTrue(task.abandon('first frame failure'))
        self.assertFalse(task.active)
        self.assertEqual(task.phase, 'idle')
        saved = json.loads(self.path.read_text(encoding='utf-8'))
        archived = json.loads(Path(saved['retirement']['archive']).read_text(encoding='utf-8'))
        self.assertEqual(archived, before)
        task.action.assert_not_called()


class RestaurantRunnerResilienceTests(unittest.TestCase):
    def setUp(self):
        # Execute the actual nested function with fake collaborators. run(), ADB,
        # frame capture and the engine are never started by this harness.
        import drift.runtime as runtime_module
        self.app = SimpleNamespace(lock=threading.RLock(), state={'events':[]},
                                   task_schedule=Mock(), stop_event=Mock(), restaurant_failures=0)
        self.app.event = lambda message:self.app.state['events'].append({'message':message})
        self.task = SimpleNamespace(active=True, phase='restore_verify', interval=259200, done=None)
        self.task.step = Mock(side_effect=RuntimeError('offline failure'))
        self.task.status = lambda:dict(pending=self.task.active, phase=self.task.phase)

        def abandon(*args):
            self.task.active, self.task.phase, self.task.done = False, 'idle', False

        self.task.abandon = Mock(side_effect=abandon)
        self.st = SimpleNamespace(stable={'old':3}, number_values={'old':1}, target_keys={'old':1})
        self.scope = dict(vars(runtime_module), self=self.app, restaurant=self.task,
                          runtime=SimpleNamespace(call=lambda label, fn, *a, **kw:fn(*a, **kw)),
                          st=self.st, stop_on_main=False)
        from drift.runtime_tasks import build_services
        self.step = build_services(lambda:self.scope,vars(runtime_module))['step_restaurant']

    def test_runtime_error_retires_schedules_backoff_and_keeps_runner_alive(self):
        self.assertIs(self.step(object()), False)
        self.task.abandon.assert_called_once()
        self.app.task_schedule.finish.assert_called_once_with(
            'restaurant_upgrade', 259200, success=False, retry_seconds=1800,controller=self.task)
        self.assertEqual(self.app.state['restaurant_upgrade'], {'pending':False, 'phase':'idle'})
        self.assertIn('餐品强化：本轮中止，30 分钟后重试，运行继续（offline failure）',
                      [e['message'] for e in self.app.state['events']])
        self.app.stop_event.set.assert_not_called()
        self.assertEqual(self.st.stable, {'old':0})
        self.assertEqual(self.st.number_values, {})
        self.assertEqual(self.st.target_keys, {})

    def test_third_consecutive_failure_propagates_original_runtime_error(self):
        error = RuntimeError('persistent failure')
        self.task.step.side_effect = error
        for _ in range(2):
            self.task.active = True
            self.assertIs(self.step(object()), False)
        self.task.active = True
        with self.assertRaises(RuntimeError) as caught:
            self.step(object())
        self.assertIs(caught.exception, error)
        self.assertEqual(self.task.abandon.call_count, 3)
        self.assertFalse(self.task.active)

    def test_waiting_frames_and_controller_recreation_do_not_clear_failure_streak(self):
        for _ in range(2):
            self.task.step.side_effect = RuntimeError('repeated failure')
            self.assertIs(self.step(object()), False)
            self.task.step.side_effect = None
            self.task.step.return_value = False
            self.task.done = None
            self.assertIs(self.step(object()), False)
            # Configuration refresh may reconstruct the task, within one runner.
            replacement = SimpleNamespace(**vars(self.task))
            self.task = replacement
            self.scope['restaurant'] = replacement
        self.task.step.side_effect = RuntimeError('third failure')
        with self.assertRaisesRegex(RuntimeError, 'third failure'):
            self.step(object())

    def test_completed_success_resets_failure_streak(self):
        for _ in range(2):
            self.assertIs(self.step(object()), False)
        self.task.step.side_effect = None
        self.task.step.return_value = False
        self.task.done = True
        self.step(object())
        self.task.done = None
        self.task.step.side_effect = RuntimeError('new streak')
        for _ in range(2):
            self.assertIs(self.step(object()), False)

    def test_abandoned_run_is_rescheduled_without_waiting_the_full_interval(self):
        # 退休的一轮必须短期回来；否则要等三天，料理背包继续爆满。
        self.task.step.side_effect = None
        self.task.step.return_value = False
        self.task.journal = dict(retirement=dict(reason='航巡恢复确认超时'))
        self.task.done = False
        self.step(object())
        self.app.task_schedule.finish.assert_called_once_with(
            'restaurant_upgrade', 259200, success=False, retry_seconds=1800,controller=self.task)
        self.task.done = None

    def test_incomplete_cleanup_retries_without_waiting_the_full_interval(self):
        self.task.step.side_effect = None
        self.task.step.return_value = False
        self.task.journal = dict(failure=dict(reason='feed_wait 阶段确认超时'))
        self.task.done = False
        self.step(object())
        self.app.task_schedule.finish.assert_called_once_with(
            'restaurant_upgrade', 259200, success=False, retry_seconds=1800,controller=self.task)
        self.task.done = None

    def test_finished_run_keeps_the_configured_interval(self):
        self.task.step.side_effect = None
        self.task.step.return_value = False
        self.task.done = True
        self.step(object())
        self.app.task_schedule.finish.assert_called_once_with(
            'restaurant_upgrade', 259200, success=True,controller=self.task)
        self.task.done = None

    def test_non_runtime_exception_is_not_swallowed(self):
        self.task.step.side_effect = ValueError('invalid data')
        with self.assertRaisesRegex(ValueError, 'invalid data'):
            self.step(object())
        self.task.abandon.assert_not_called()
        self.app.task_schedule.finish.assert_not_called()

    def test_retry_is_persisted_and_does_not_block_other_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            now = datetime(2026, 10, 28, tzinfo=timezone.utc)
            self.app.task_schedule = TaskSchedule(directory, 'offline', clock=lambda:now)
            self.assertIs(self.step(object()), False)
            restored = TaskSchedule(directory, 'offline', clock=lambda:now)
            self.assertFalse(restored.ready('restaurant_upgrade'))
            self.assertTrue(restored.ready('guild_daily'))
            self.assertEqual(restored.entries['restaurant_upgrade']['reason'], 'retry')
            now += timedelta(seconds=1799)
            self.assertFalse(restored.ready('restaurant_upgrade'))
            now += timedelta(seconds=1)
            self.assertTrue(restored.ready('restaurant_upgrade'))

    def test_real_runner_loop_continues_other_tasks_after_restaurant_failure(self):
        import numpy as np
        from drift.runtime import Clicker
        from drift.tasks.frame_runtime import FrameRuntime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(serial='offline', resolution=[64, 96], poll_interval=.1,
                          max_consecutive_clicks=3, adb_path='never-used', rules=[
                              dict(id='driftinggirl_restaurant_upgrade', interval=259200, enabled=True),
                              dict(id='driftinggirl_equipment_upgrade', interval=900, enabled=True)])
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')

            class Device:
                def __init__(self, config, stop):
                    self.image = None
                    self.reads = 0

                def screenshot(self):
                    self.reads += 1
                    if self.reads > 5:
                        raise InterruptedError('end of offline replay')
                    self.image = np.zeros((96, 64, 3), dtype=np.uint8)
                    return self.image

                def click(self, button):
                    raise AssertionError('No device input is allowed')

                def stuck_record_add(self, button):
                    pass

            task = self.task
            task.retire_expired_on_start = Mock(return_value=False)

            def step(image, *, due=True):
                task.done = None
                if task.active:
                    raise RuntimeError('offline failure')
                return False

            task.step = Mock(side_effect=step)
            app = Clicker(root, Device)
            app.digit_reader = SimpleNamespace()
            other_task = Mock(return_value=False)
            equipment = SimpleNamespace(active=False, cycles=0, panel=lambda _:False,
                                        step=other_task, status=lambda:{})
            try:
                with patch('drift.runtime.RestaurantUpgrade', return_value=task), \
                     patch('app.EquipmentUpgrade', return_value=equipment), \
                     patch.object(FrameRuntime, 'start_engine'), \
                     patch.object(FrameRuntime, 'wait', return_value=False), \
                     patch('drift.runtime.scan_rules', return_value=(None, False, False)):
                    app.run([(r, None, None) for r in config['rules']], False)
                self.assertIsNone(app.state['last_error'])
                self.assertGreaterEqual(other_task.call_count, 2)
                self.assertFalse(task.active)
                self.assertFalse(app.task_schedule.ready('restaurant_upgrade'))
                self.assertEqual(app.task_schedule.entries['restaurant_upgrade']['reason'], 'retry')
                self.assertTrue(any('30 分钟后重试，运行继续' in e['message'] for e in app.events))
                self.assertEqual(list((root/'logs').glob('error-*.json')), [])
            finally:
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)


if __name__ == '__main__':
    unittest.main()
