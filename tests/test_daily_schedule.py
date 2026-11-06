"""08:00 boundaries and real completion must survive restarts without extra taps."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_clicker as helpers
from daily_schedule import DailySchedule, BEIJING


class DailyScheduleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime(2026,10,3,9,tzinfo=BEIJING)
        self.schedule = DailySchedule(self.root, 'test', lambda: self.now)

    def tearDown(self):
        self.temp.cleanup()

    def test_reset_is_eight_beijing_not_midnight_or_host_timezone(self):
        for moment, period in [
                (datetime(2026,10,3,0,0,tzinfo=BEIJING), '2026-10-02'),
                (datetime(2026,10,3,7,59,59,tzinfo=BEIJING), '2026-10-02'),
                (datetime(2026,10,3,8,tzinfo=BEIJING), '2026-10-03'),
                (datetime(2026,10,3,0,tzinfo=timezone.utc), '2026-10-03')]:
            self.now = moment
            self.assertEqual(self.schedule.period(), period)
        self.assertEqual(self.schedule.status()['guild_shop']['next_reset'], '2026-10-04T08:00:00+08:00')

    def test_completion_survives_restart_and_missed_reset_becomes_due(self):
        self.assertTrue(self.schedule.mark_complete('daily_reward', '2026-10-03'))
        self.assertFalse(self.schedule.is_complete('guild_shop'))
        self.now = datetime(2026,10,4,7,59,59,tzinfo=BEIJING)
        restarted = DailySchedule(self.root, 'test', lambda: self.now)
        self.assertTrue(restarted.is_complete('daily_reward'))
        self.now = datetime(2026,10,4,8,tzinfo=BEIJING)
        self.assertFalse(restarted.is_complete('daily_reward'))
        self.now = datetime(2026,10,8,15,tzinfo=BEIJING)
        self.assertFalse(DailySchedule(self.root,'test',lambda:self.now).is_complete('daily_reward'))

    def test_bonus_check_preserves_free_timestamp_and_survives_restart_and_reset(self):
        self.schedule.mark_complete('daily_reward',self.schedule.period())
        original=dict(self.schedule.completed['daily_reward'])
        self.schedule.require_bonus=True
        self.assertFalse(self.schedule.is_complete('daily_reward'))
        self.assertTrue(self.schedule.status()['daily_reward']['bonus_pending'])
        self.schedule.mark_complete('daily_reward',self.schedule.period())
        self.assertEqual(self.schedule.completed['daily_reward'],original)
        restarted=DailySchedule(self.root,'test',lambda:self.now);restarted.require_bonus=True
        self.assertTrue(restarted.is_complete('daily_reward'))
        self.now=datetime(2026,10,4,8,tzinfo=BEIJING)
        self.assertFalse(restarted.is_complete('daily_reward'))

    def test_workflow_crossing_reset_cannot_complete_the_new_period(self):
        self.assertFalse(self.schedule.mark_complete('guild_shop', '2026-10-02'))
        self.assertFalse(self.schedule.is_complete('guild_shop'))
        self.assertFalse(self.schedule.path.exists())

    def test_invalid_state_or_other_device_does_not_suppress_game_check(self):
        self.schedule.path.write_text('{broken', encoding='utf-8')
        broken = DailySchedule(self.root,'test',lambda:self.now)
        self.assertIsNotNone(broken.warning)
        self.assertFalse(broken.is_complete('daily_reward'))
        self.schedule.mark_complete('daily_reward','2026-10-03')
        self.assertFalse(DailySchedule(self.root,'another-device',lambda:self.now).is_complete('daily_reward'))

    def test_failed_atomic_write_keeps_previous_completion(self):
        self.schedule.mark_complete('daily_reward','2026-10-03')
        previous = self.schedule.path.read_bytes()
        with patch('drift.core.persistence.os.replace',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.schedule.mark_complete('guild_shop','2026-10-03')
        self.assertTrue(self.schedule.is_complete('daily_reward'))
        self.assertFalse(self.schedule.is_complete('guild_shop'))
        self.assertEqual(self.schedule.path.read_bytes(),previous)


class DailyRunnerTests(unittest.TestCase):
    tearDown = helpers.FrameworkTests.tearDown
    rule = helpers.FrameworkTests.rule
    wait_for = helpers.FrameworkTests.wait_for

    def setUp(self):
        helpers.FrameworkTests.setUp(self)
        self.now = datetime(2026,10,3,9,tzinfo=BEIJING)
        self.app.daily_schedule.clock = lambda:self.now

    def workflow(self):
        for suffix, stage, following, action in (
                ('open','main','daily_tab','click'),
                ('premium','daily_tab','daily_reward','click'),
                ('claim','daily_reward','daily_bonus','click'),
                # 附赠奖励步骤(2026-10-11 新增):免费领完后顺手点一次,领不上就跳过
                ('bonus','daily_bonus','daily_after_claim','click'),
                ('unavailable','daily_reward','daily_return','click'),
                ('close','daily_after_claim','daily_return','click'),
                ('return','daily_return','main','observe')):
            self.rule().update(id='driftinggirl_daily_'+suffix, stage=stage,
                               next_stage=following, action=action, next_delay=0)
        self.app.persist()

    def test_completed_day_blocks_taps_after_restart_and_switch_toggle(self):
        self.workflow()
        self.app.start(preview=False,stop_on_main=True)
        self.wait_for(lambda:not self.app.running())
        self.assertEqual(len(self.app.device.taps),5)
        self.assertTrue(self.app.daily_schedule.is_complete('daily_reward'))
        self.app.daily_schedule = DailySchedule(self.root,'test',lambda:self.now)
        self.app.change_feature({'id':'daily_reward','enabled':False})
        self.app.change_feature({'id':'daily_reward','enabled':True})
        self.app.start(preview=False)
        self.wait_for(lambda:self.app.state['frames']>=4)
        self.assertEqual(len(self.app.device.taps),5)
        self.assertTrue(next(f for f in self.app.status()['features'] if f['id']=='daily_reward')['schedule']['completed'])

    def test_preview_never_creates_completion_and_stop_before_return_remains_due(self):
        self.workflow()
        self.app.start(preview=True)
        self.wait_for(lambda:self.app.state['matches']>0)
        self.app.stop(); self.app.thread.join(3)
        self.assertFalse(self.app.daily_schedule.path.exists())
        original = self.app.device.click
        def stop_at_close(button):
            original(button)
            if button.name=='driftinggirl_daily_close':self.app.stop()
        self.app.device.click=stop_at_close
        self.app.start(preview=False)
        self.wait_for(lambda:not self.app.running())
        self.assertFalse(self.app.daily_schedule.is_complete('daily_reward'))
        self.assertFalse(self.app.daily_schedule.path.exists())

    def test_reset_clears_old_retry_interval_for_unfinished_work(self):
        self.now=datetime(2026,10,3,7,59,59,tzinfo=BEIJING)
        self.rule().update(id='driftinggirl_daily_open',stage='main',next_stage='daily_return',interval=3600)
        self.rule().update(id='driftinggirl_daily_return',stage='daily_return',next_stage='main',action='observe')
        self.app.start(preview=False,max_clicks=2)
        self.wait_for(lambda:self.app.state['stage']=='main' and self.app.state['matches']>=2)
        frames=self.app.state['frames']
        self.wait_for(lambda:self.app.state['frames']>=frames+3)
        self.assertEqual(len(self.app.device.taps),1)
        self.assertFalse(self.app.daily_schedule.is_complete('daily_reward'))
        self.now=datetime(2026,10,3,8,tzinfo=BEIJING)
        self.wait_for(lambda:not self.app.running())
        self.assertEqual(len(self.app.device.taps),2)
        self.assertEqual(self.app.state['stop_reason'],'click_limit')

    def test_entry_frame_before_reset_cannot_mark_new_day_even_if_tap_crosses_it(self):
        self.workflow()
        self.now=datetime(2026,10,3,7,59,59,tzinfo=BEIJING)
        original=self.app.device.click
        def cross_reset(button):
            original(button)
            if button.name=='driftinggirl_daily_open':
                self.now=datetime(2026,10,3,8,tzinfo=BEIJING)
        self.app.device.click=cross_reset
        self.app.start(preview=False,stop_on_main=True)
        self.wait_for(lambda:not self.app.running())
        self.assertEqual(len(self.app.device.taps),5)
        self.assertFalse(self.app.daily_schedule.is_complete('daily_reward'))


if __name__=='__main__':
    unittest.main()
