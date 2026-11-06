"""Persisted task timing must not delay live triggers or unfinished returns."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_clicker as helpers
from daily_schedule import DailySchedule, BEIJING
from features import FEATURES, TIMED_FEATURES, DAILY_FEATURES, REALTIME_CONDITIONS
from task_schedule import TaskSchedule


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime(2026,10,10,12,tzinfo=BEIJING)
        self.schedule = TaskSchedule(self.root, 'test', lambda:self.now)
        self.daily = DailySchedule(self.root, 'test', lambda:self.now)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_features_have_one_explicit_execution_mode(self):
        self.assertFalse(TIMED_FEATURES & DAILY_FEATURES)
        self.assertFalse((TIMED_FEATURES | DAILY_FEATURES) & REALTIME_CONDITIONS.keys())
        self.assertEqual(TIMED_FEATURES | DAILY_FEATURES | REALTIME_CONDITIONS.keys(),
                         {f['id'] for f in FEATURES})

    def test_realtime_conditions_never_gain_a_next_run_or_timing_file(self):
        for key in REALTIME_CONDITIONS:
            self.assertTrue(self.schedule.ready(key))
            self.assertIsNone(self.schedule.next_run(key))
            with self.assertRaises(ValueError):
                self.schedule.set_next(key, self.now.isoformat())
        self.assertFalse(self.schedule.path.exists())

    def test_completed_cycle_delays_from_finish_not_start_and_survives_restart(self):
        self.schedule.begin('equipment_upgrade',900)
        self.now += timedelta(seconds=100)
        self.schedule.finish('equipment_upgrade',900)
        expected = self.now+timedelta(seconds=900)
        restarted = TaskSchedule(self.root,'test',lambda:self.now)
        self.assertEqual(restarted.next_run('equipment_upgrade'),expected)
        self.assertFalse(restarted.ready('equipment_upgrade'))
        self.now = expected
        self.assertTrue(restarted.ready('equipment_upgrade'))

    def test_stopping_unfinished_attempt_reserves_retry_without_completing(self):
        self.schedule.begin('parallel_skills',300)
        self.schedule.release()
        restarted = TaskSchedule(self.root,'test',lambda:self.now)
        self.assertFalse(restarted.ready('parallel_skills'))
        self.assertEqual(restarted.entries['parallel_skills']['reason'],'attempt')
        self.assertNotIn('last_finished',restarted.entries['parallel_skills'])

    def test_next_time_cannot_override_daily_completion_and_reset_releases_old_retry(self):
        self.daily.mark_complete('guild_shop',self.daily.period())
        daily = self.daily.status()['guild_shop']
        self.schedule.set_next('guild_shop',None,daily)
        self.assertEqual(self.schedule.next_run('guild_shop',daily),
                         datetime(2026,10,11,8,tzinfo=BEIJING))
        self.assertFalse(self.schedule.ready('guild_shop',daily))
        self.now = datetime(2026,10,11,8,tzinfo=BEIJING)
        self.assertTrue(self.schedule.ready('guild_shop',self.daily.status()['guild_shop']))

    def test_unfinished_daily_retries_at_saved_time_but_missed_refresh_is_due(self):
        self.now = datetime(2026,10,10,7,59,tzinfo=BEIJING)
        self.schedule.begin('daily_reward',3600,self.daily.status()['daily_reward'])
        self.assertFalse(self.schedule.ready('daily_reward',self.daily.status()['daily_reward']))
        self.now += timedelta(minutes=1)
        self.assertTrue(self.schedule.ready('daily_reward',self.daily.status()['daily_reward']))

    def test_manual_daily_postponement_survives_refresh_but_never_bypasses_completion(self):
        self.daily.mark_complete('guild_shop',self.daily.period())
        chosen = datetime(2026,10,11,10,tzinfo=BEIJING)
        self.schedule.set_next('guild_shop',chosen.isoformat(),self.daily.status()['guild_shop'])
        self.assertEqual(self.schedule.next_run('guild_shop',self.daily.status()['guild_shop']),chosen)
        self.now = datetime(2026,10,11,8,tzinfo=BEIJING)
        restarted = TaskSchedule(self.root,'test',lambda:self.now)
        self.assertFalse(restarted.ready('guild_shop',self.daily.status()['guild_shop']))
        self.now = chosen
        self.assertTrue(restarted.ready('guild_shop',self.daily.status()['guild_shop']))

    def test_legacy_free_completion_releases_bonus_but_preserves_manual_time_and_retry(self):
        self.daily.mark_complete('daily_reward',self.daily.period())
        self.schedule.finish('daily_reward',3600,self.daily.status()['daily_reward'])
        self.daily.require_bonus=True
        pending=self.daily.status()['daily_reward']
        before=self.schedule.path.read_bytes()
        self.assertTrue(self.schedule.ready('daily_reward',pending))
        self.assertEqual(self.schedule.path.read_bytes(),before)
        self.schedule.begin('daily_reward',3600,pending)
        self.assertFalse(self.schedule.ready('daily_reward',pending))
        chosen=self.now+timedelta(hours=2)
        self.schedule.release();self.schedule.set_next('daily_reward',chosen.isoformat(),pending)
        self.assertEqual(self.schedule.next_run('daily_reward',pending),chosen)

    def test_bad_or_other_serial_state_does_not_suppress_tasks(self):
        self.schedule.path.write_text('{broken')
        broken = TaskSchedule(self.root,'test',lambda:self.now)
        self.assertIsNotNone(broken.warning)
        self.assertTrue(broken.ready('equipment_upgrade'))
        self.schedule.set_next('equipment_upgrade',(self.now+timedelta(days=1)).isoformat())
        self.assertTrue(TaskSchedule(self.root,'other',lambda:self.now).ready('equipment_upgrade'))

    def test_failed_atomic_write_does_not_change_saved_time(self):
        self.schedule.set_next('equipment_upgrade',self.now.isoformat())
        saved = self.schedule.path.read_bytes()
        with patch('drift.core.persistence.os.replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.schedule.set_next('equipment_upgrade',(self.now+timedelta(hours=1)).isoformat())
        self.assertEqual(self.schedule.path.read_bytes(),saved)
        self.assertEqual(self.schedule.next_run('equipment_upgrade'),self.now)

    def test_timezone_is_explicit_and_normalized_to_beijing(self):
        for invalid in ('2026-10-10T12:00:00',True,15,'bad'):
            with self.assertRaises(ValueError):
                self.schedule.set_next('equipment_upgrade',invalid)
        self.schedule.set_next('equipment_upgrade','2026-10-10T04:00:00Z')
        self.assertEqual(self.schedule.next_run('equipment_upgrade'),self.now)

    def test_disabled_task_keeps_saved_time_and_real_events_have_no_countdown(self):
        self.schedule.set_next('equipment_upgrade',(self.now+timedelta(hours=1)).isoformat())
        config = dict(rules=[dict(id=r,enabled=False) for f in FEATURES for r in f['rules']])
        state = self.schedule.status(config,self.daily.status())
        self.assertEqual(state['equipment_upgrade']['status'],'disabled')
        self.assertEqual(state['equipment_upgrade']['seconds_until'],3600)
        self.assertIsNone(state['raid']['next_run'])
        self.assertIsNone(state['raid']['seconds_until'])
        config['rules'][0]['enabled']=True
        self.assertEqual(self.schedule.next_run('equipment_upgrade'),self.now+timedelta(hours=1))

    def controller(self):
        class Controller:
            active=False
            cycles=0
            next_check_at=0
            interval=900
            calls=0
            shown=False
            def panel(self, frame): return self.shown
            def step(self, frame):
                self.calls+=1
                self.active=not self.active
                if not self.active:
                    self.cycles+=1
                return self.active
        return Controller()

    def test_due_gate_blocks_start_but_never_blocks_in_progress_return(self):
        controller=self.controller()
        self.schedule.set_next('equipment_upgrade',(self.now+timedelta(seconds=50)).isoformat())
        invoke=lambda f, frame:f(frame)
        self.assertFalse(self.schedule.step_controller('equipment_upgrade',controller,None,invoke))
        self.assertEqual(controller.calls,0)
        self.now += timedelta(seconds=50)
        self.assertTrue(self.schedule.step_controller('equipment_upgrade',controller,None,invoke))
        self.assertFalse(self.schedule.ready('equipment_upgrade'))
        self.assertFalse(self.schedule.step_controller('equipment_upgrade',controller,None,invoke))
        self.assertEqual(controller.calls,2)
        self.assertEqual(self.schedule.next_run('equipment_upgrade'),self.now+timedelta(seconds=900))

    def test_existing_panel_can_be_finished_even_before_next_time(self):
        controller=self.controller()
        controller.shown=True
        self.schedule.set_next('equipment_upgrade',(self.now+timedelta(hours=1)).isoformat())
        self.assertTrue(self.schedule.step_controller('equipment_upgrade',controller,None,lambda f,x:f(x)))
        self.assertEqual(controller.calls,1)


class RunnerScheduleTests(unittest.TestCase):
    tearDown=helpers.FrameworkTests.tearDown
    rule=helpers.FrameworkTests.rule
    wait_for=helpers.FrameworkTests.wait_for

    def setUp(self):
        helpers.FrameworkTests.setUp(self)
        self.now=datetime(2026,10,10,12,tzinfo=BEIJING)
        self.app.task_schedule.clock=lambda:self.now
        self.app.daily_schedule.clock=lambda:self.now

    def daily_workflow(self):
        for suffix,stage,following,action in (
                ('open','main','daily_tab','click'),
                ('premium','daily_tab','daily_reward','click'),
                ('claim','daily_reward','daily_bonus','click'),
                # 附赠奖励步骤(2026-10-11 新增)
                ('bonus','daily_bonus','daily_after_claim','click'),
                ('unavailable','daily_reward','daily_return','click'),
                ('close','daily_after_claim','daily_return','click'),
                ('return','daily_return','main','observe')):
            self.rule().update(id='driftinggirl_daily_'+suffix,stage=stage,
                               next_stage=following,action=action,next_delay=0)
        self.app.persist()

    def test_future_daily_does_not_block_realtime_relief(self):
        self.daily_workflow()
        self.app.change_task_time(dict(id='daily_reward',next_run=(self.now+timedelta(hours=1)).isoformat()))
        self.rule().update(id='driftinggirl_relief_parachute')
        self.app.start(preview=False,max_clicks=1)
        self.wait_for(lambda:not self.app.running())
        self.assertEqual(len(self.app.device.taps),1)
        self.assertIn('下次开启时间',str(self.app.events))
        self.assertNotIn('daily_reward',self.app.task_schedule.active)
        self.assertFalse(self.app.daily_schedule.is_complete('daily_reward'))

    def test_due_time_releases_whole_daily_workflow_and_saves_reset(self):
        self.daily_workflow()
        self.app.change_task_time(dict(id='daily_reward',next_run=(self.now+timedelta(seconds=30)).isoformat()))
        self.app.start(preview=False,stop_on_main=True)
        self.wait_for(lambda:self.app.state['frames']>=3)
        self.assertFalse(self.app.device.taps)
        self.now+=timedelta(seconds=30)
        self.wait_for(lambda:not self.app.running())
        self.assertEqual(len(self.app.device.taps),5)
        self.assertTrue(self.app.daily_schedule.is_complete('daily_reward'))
        self.assertEqual(self.app.task_schedule.next_run('daily_reward',self.app.daily_schedule.status()['daily_reward']),
                         datetime(2026,10,11,8,tzinfo=BEIJING))

    def test_preview_and_scoped_diagnostics_do_not_write_formal_next_time(self):
        self.daily_workflow()
        self.app.start(preview=True)
        self.wait_for(lambda:self.app.state['frames']>=3)
        self.app.stop();self.app.thread.join(3)
        self.assertFalse(self.app.task_schedule.path.exists())
        self.app.start(preview=False,rule_ids=[r['id'] for r in self.app.config['rules']],stop_on_main=True)
        self.wait_for(lambda:not self.app.running())
        self.assertFalse(self.app.task_schedule.path.exists())

    def test_schedule_api_rejects_realtime_unknown_and_active_changes(self):
        self.daily_workflow()
        original=deepcopy(self.app.config)
        for data in (dict(id='raid',next_run=None),dict(id='bad',next_run=None),
                     dict(id='daily_reward',next_run=True),dict(id='daily_reward',next_run=None,enabled=True)):
            with self.assertRaises(ValueError):self.app.change_task_time(data)
        self.assertEqual(self.app.config,original)
        self.assertFalse(self.app.task_schedule.path.exists())
        self.app.start(preview=True)
        self.app.task_schedule.active.add('daily_reward')
        with self.assertRaises(ValueError):self.app.change_task_time(dict(id='daily_reward',next_run=None))


if __name__=='__main__':
    unittest.main()
