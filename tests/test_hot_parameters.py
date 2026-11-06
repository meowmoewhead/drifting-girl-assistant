from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import tests.test_features as feature_helpers
from drift.tasks.schedule import TaskSchedule
from drift.tasks.features import FEATURE_BY_ID


class HotParameterTests(unittest.TestCase):
    setUp = feature_helpers.FeatureTests.setUp
    tearDown = feature_helpers.FeatureTests.tearDown
    rule = feature_helpers.FeatureTests.rule
    feature = feature_helpers.FeatureTests.feature
    wait_for = feature_helpers.FeatureTests.wait_for

    def test_runtime_values_are_hot_without_reconstructing_tasks_or_touching_journals(self):
        record = b'{"deadline":50,"attempts":{"start":1},"pending":true}'
        path = self.root/'meter_calibration_state.json'; path.write_bytes(record)
        with patch.object(self.app, 'running', return_value=True):
            self.app.change_settings(dict(poll_interval=.3, max_consecutive_clicks=17))
            self.assertFalse(self.app.status()['settings_pending'])
        self.assertEqual(self.app.config_revision, 0)
        self.assertEqual(self.app.config['poll_interval'], .3)
        self.assertEqual(self.app.config['max_consecutive_clicks'], 17)
        self.assertEqual(path.read_bytes(), record)

    def test_invalid_hot_values_do_not_partially_save(self):
        original = deepcopy(self.app.config)
        for values in (dict(poll_interval=.05), dict(max_consecutive_clicks=True),
                       dict(poll_interval=.3, max_consecutive_clicks=0), dict(serial='other')):
            with self.subTest(values=values), patch.object(self.app,'running',return_value=True), self.assertRaises(ValueError):
                self.app.change_settings(values)
            self.assertEqual(self.app.config, original)

    def test_feature_interval_and_threshold_queue_while_another_task_is_open(self):
        rules = self.feature('mana'); rules[0]['number_condition'] = {'value':25}
        self.app.state.update(stage='equipment_owned', applied_revision=0)
        record = b'{"deadline":100,"attempts":{"hold":1},"growth_until":500}'
        path=self.root/'parallel_skill_state.json'; path.write_bytes(record)
        with patch.object(self.app,'running',return_value=True):
            self.app.change_feature(dict(id='mana',interval=120,mana_threshold=30))
            self.assertTrue(self.app.status()['settings_pending'])
        self.assertEqual(self.app.state['stage'],'equipment_owned')
        self.assertEqual(path.read_bytes(), record)
        self.assertEqual(self.app.config['rules'][0]['number_condition']['value'],30)
        self.assertEqual(self.app.config['rules'][0]['interval'],120)

    def test_scoped_task_still_rejects_unrelated_parameter_change(self):
        self.feature('mana'); self.app.state['scoped']=True
        before=deepcopy(self.app.config)
        with patch.object(self.app,'running',return_value=True), self.assertRaisesRegex(ValueError,'单项'):
            self.app.change_feature(dict(id='mana',interval=120))
        self.assertEqual(self.app.config,before)

    def test_hot_runtime_limit_keeps_spent_consecutive_clicks_and_same_thread(self):
        self.rule(); self.app.config['max_consecutive_clicks']=8
        original = self.app.device.click
        saved_threads=[]
        def click(button):
            original(button)
            if len(self.app.device.taps)==1:
                saved_threads.append(self.app.thread)
                self.app.change_settings(dict(poll_interval=.1,max_consecutive_clicks=2))
        self.app.device.click=click
        self.app.start(preview=False)
        self.wait_for(lambda:not self.app.running())
        self.assertEqual(len(self.app.device.taps),2)
        self.assertIs(saved_threads[0], self.app.thread)
        self.assertIn('连续点击',self.app.state['last_error'])

    def test_open_workflow_finishes_with_old_steps_before_interval_reload(self):
        rules=self.feature('mailbox')
        for rule,stage,following in zip(rules,['main','claim','close','return'],['claim','close','return','main']):
            rule.update(stage=stage,next_stage=following,next_delay=.05)
        rules[3]['action']='observe'
        for rule in rules[4:]:rule.update(stage='claim',action='observe',threshold=1)
        original=self.app.device.click;pending=[]
        def click(button):
            original(button)
            if len(self.app.device.taps)==1:
                self.app.change_feature(dict(id='mailbox',interval=120))
                pending.append(self.app.status()['settings_pending'])
        self.app.device.click=click
        self.app.start(preview=False)
        self.wait_for(lambda:self.app.state['applied_revision']==1)
        self.assertEqual(pending,[True])
        self.assertEqual(len(self.app.device.taps),3)
        self.assertEqual(self.app.state['stage'],'main')
        self.assertIsNone(self.app.state['last_error'])

    def test_unrelated_schedule_can_change_while_running_but_active_task_is_reserved(self):
        self.feature('equipment_upgrade')
        with patch.object(self.app,'running',return_value=True):
            result=self.app.change_task_time(dict(id='equipment_upgrade',next_run='2026-11-05T18:00:00+08:00'))
        self.assertEqual(result['id'],'equipment_upgrade')
        self.app.task_schedule.active.add('equipment_upgrade')
        before=(self.root/'task_schedule.json').read_bytes()
        with patch.object(self.app,'running',return_value=True), self.assertRaisesRegex(ValueError,'正在进行'):
            self.app.change_task_time(dict(id='equipment_upgrade',next_run='2026-11-05T19:00:00+08:00'))
        self.assertEqual((self.root/'task_schedule.json').read_bytes(),before)


class HotCooldownTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.now=datetime(2026,11,5,12,tzinfo=timezone.utc)
        self.schedule=TaskSchedule(Path(self.temp.name),'test',clock=lambda:self.now)
        self.identifier=FEATURE_BY_ID['equipment_upgrade']['rules'][0]
        self.old=[dict(id=self.identifier,interval=900)]
        self.new=[dict(id=self.identifier,interval=300)]

    def test_completed_cooldown_uses_original_finish_instead_of_change_time(self):
        self.schedule.finish('equipment_upgrade',900)
        finished=self.schedule.entries['equipment_upgrade']['last_finished']
        self.now+=timedelta(seconds=100)
        self.schedule.refresh_intervals(self.old,self.new)
        self.assertEqual(self.schedule.next_run('equipment_upgrade'),datetime.fromisoformat(finished)+timedelta(seconds=300))
        self.assertEqual(self.schedule.entries['equipment_upgrade']['last_finished'],finished)

    def test_failed_reserved_manual_daily_and_active_windows_are_not_renewed(self):
        for reason in ('retry','attempt','manual'):
            with self.subTest(reason=reason):
                self.schedule.finish('equipment_upgrade',900)
                self.schedule.entries['equipment_upgrade']['reason']=reason
                before=deepcopy(self.schedule.entries)
                self.schedule.refresh_intervals(self.old,self.new)
                self.assertEqual(self.schedule.entries,before)
        self.schedule.finish('equipment_upgrade',900)
        self.schedule.active.add('equipment_upgrade')
        before=deepcopy(self.schedule.entries)
        self.schedule.refresh_intervals(self.old,self.new)
        self.assertEqual(self.schedule.entries,before)

    def test_reenabled_task_applies_saved_interval_without_reopening_manual_dates(self):
        self.schedule.finish('equipment_upgrade',900)
        self.schedule.refresh_intervals([],self.new)
        self.assertEqual((self.schedule.next_run('equipment_upgrade')-self.now).total_seconds(),300)


if __name__=='__main__':unittest.main()
