from copy import deepcopy
from datetime import datetime
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np

from drift.tasks.daily import DailySchedule, BEIJING
from drift.tasks.schedule import TaskSchedule
from drift.tasks.guild_daily import GuildDaily, GuildVisual
import cv2
from tools.configure_guild_daily import add, RULE


class GuildScheduleTests(unittest.TestCase):
    def test_18_local_daily_boundary_and_restart(self):
        with tempfile.TemporaryDirectory() as d:
            now=[datetime(2026,10,24,17,59,59,tzinfo=BEIJING)]
            daily=DailySchedule(d,'x',lambda:now[0]);schedule=TaskSchedule(d,'x',lambda:now[0])
            self.assertFalse(schedule.ready('guild_daily',daily.status()['guild_daily']))
            now[0]=now[0].replace(hour=18,minute=0,second=0)
            self.assertTrue(schedule.ready('guild_daily',daily.status()['guild_daily']))
            daily.mark_complete('guild_daily',daily.period())
            self.assertFalse(schedule.ready('guild_daily',daily.status()['guild_daily']))
            self.assertEqual(schedule.next_run('guild_daily',daily.status()['guild_daily']).hour,18)
            reload=DailySchedule(d,'x',lambda:now[0])
            self.assertTrue(reload.is_complete('guild_daily'))
            now[0]=datetime(2026,10,25,8,0,tzinfo=BEIJING)
            self.assertFalse(schedule.ready('guild_daily',reload.status()['guild_daily']))
            now[0]=now[0].replace(hour=18)
            self.assertTrue(schedule.ready('guild_daily',reload.status()['guild_daily']))

    def test_current_settings_and_other_records_preserved(self):
        original={'serial':'x','rules':[{'id':'driftinggirl_guild_open','enabled':False,'interval':123}], 'poll_interval':.6}
        before=deepcopy(original);updated=add(original)
        self.assertEqual(original,before)
        self.assertEqual(updated['rules'][:-1],before['rules'])
        self.assertEqual(add(updated),updated)

    def test_manual_earlier_time_does_not_bypass_18(self):
        with tempfile.TemporaryDirectory() as d:
            now=datetime(2026,10,24,12,tzinfo=BEIJING)
            daily=DailySchedule(d,'x',lambda:now);s=TaskSchedule(d,'x',lambda:now)
            s.set_next('guild_daily',None,daily.status()['guild_daily'])
            self.assertEqual(s.next_run('guild_daily',daily.status()['guild_daily']).hour,18)


class GuildJournalTests(unittest.TestCase):
    def controller(self,d):
        c=GuildDaily.__new__(GuildDaily)
        c.clock=lambda:100.;c.path=Path(d)/'guild_daily_state.json'
        c.journal=dict(pending=True,phase='stop',deadline=500,phase_deadline=200,attempts={})
        c.phase='stop';c.event=Mock();return c

    def test_attempt_reserved_durably_before_input_and_never_reset(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d)
            self.assertTrue(c.reserve({'kind':'cancel'}));self.assertTrue(c.reserve({'kind':'cancel'}))
            self.assertFalse(c.reserve({'kind':'cancel'}))
            self.assertEqual(json.loads(c.path.read_text())['attempts']['cancel'],2)
            c.clock=lambda:201
            self.assertFalse(c.reserve({'kind':'stop'}))

    def test_same_phase_does_not_renew_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c._set('stop',600)
            self.assertEqual(c.journal['phase_deadline'],200)

    def test_new_collection_budget_is_bounded_and_does_not_renew_saved_stage(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.journal['attempts']={'submit:0:used':1}
            c._set('collect',c.COLLECTION_LIMIT)
            self.assertEqual(c.journal['phase_deadline'],460)
            c.clock=lambda:459;c._set('collect',c.COLLECTION_LIMIT)
            self.assertEqual(c.journal['phase_deadline'],460)
            c.clock=lambda:461
            self.assertFalse(c.reserve({'kind':'panel-close'}))
            self.assertEqual(c.journal['attempts'],{'submit:0:used':1})

    def test_cached_object_is_not_multiple_confirmations(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.key=c.since=c.last_seen=c.last_frame=None;c.seen=0
            frame=np.zeros((4,4,3),np.uint8)
            for t in (100,102,104):
                c.clock=lambda t=t:t
                self.assertFalse(c._confirmed('x',frame))
            self.assertEqual(c.seen,1)

    def test_three_different_frames_three_seconds(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.key=c.since=c.last_seen=c.last_frame=None;c.seen=0
            for t,answer in ((100,False),(101,False),(103,True)):
                c.clock=lambda t=t:t
                self.assertEqual(c._confirmed('x',np.zeros((4,4,3),np.uint8)),answer)

    def test_expired_workflow_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.clock=lambda:501;c.journal['attempts']['cancel']=2
            with self.assertRaises(RuntimeError):c.step(np.zeros((4,4,3)),due=True,period='x')
            self.assertEqual(c.journal['deadline'],500)
            self.assertEqual(c.journal['attempts']['cancel'],2)
            self.assertEqual(c.phase,'blocked')

    def test_retirement_archives_exact_bytes_without_resetting_used_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d);c.journal.update(version=1,phase='blocked');c.phase='blocked'
            c.journal['attempts']={'cancel':2,'teleport-confirm':1};c._save()
            original=c.path.read_bytes();c.clock=lambda:501
            self.assertTrue(c.retire_expired_on_start())
            self.assertEqual(Path(c.journal['retirement']['archive']).read_bytes(),original)
            self.assertEqual(c.journal['deadline'],500)
            self.assertEqual(c.journal['attempts'],{'cancel':2,'teleport-confirm':1})
            self.assertFalse(c.active)

    def test_teleport_confirmation_can_only_be_reserved_once(self):
        with tempfile.TemporaryDirectory() as d:
            c=self.controller(d)
            self.assertTrue(c.reserve({'kind':'teleport-confirm'}))
            self.assertFalse(c.reserve({'kind':'teleport-confirm'}))


class GuildReturnMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.now=100.
        c=self.c=GuildDaily.__new__(GuildDaily)
        c.clock=lambda:self.now;c.path=Path(self.tmp.name)/'guild_daily_state.json'
        c.journal=dict(pending=True,phase='map_close',deadline=700,phase_deadline=220,
                       attempts={'map-select':1},map_goal='observatory',map_after='gate')
        c.phase='map_close';c.event=Mock();c.next_at=0;c.hold=None
        c.key=c.since=c.last_seen=c.last_frame=None;c.seen=0
        c.view=Mock();c.view._map.return_value=True
        c.bag=Mock();c.bag.moving.return_value=False
        c.stopped_map=Mock(return_value=True);c.normal_idle=Mock(return_value=False)
        c.signature=Mock(return_value=('normal','51','observatory'))
        self.inputs=[]
        def action(request):
            if c.validate(request,self.current) and c.reserve(request):
                self.inputs.append(request);return True
            return False
        c.action=action

    def step(self,t,frame=None):
        self.now=t;self.current=np.zeros((4,4,3),np.uint8) if frame is None else frame
        self.c.step(self.current,due=False,period='today')

    def test_incomplete_moving_map_is_not_a_closed_map(self):
        c=self.c;c.stopped_map.return_value=False;c.bag.moving.return_value=True
        for t in (100,102,104):self.step(t)
        self.assertEqual(c.phase,'map_close');self.assertEqual(self.inputs,[])
        self.assertEqual(c.journal['phase_deadline'],220)
        self.assertIn('地图仍打开',c.hold)

    def test_unknown_or_single_cached_main_does_not_advance(self):
        c=self.c;c.view._map.return_value=False
        for t in (100,102,104):self.step(t)
        self.assertEqual(c.phase,'map_close')
        c.normal_idle.return_value=True;frame=np.zeros((4,4,3),np.uint8)
        for t in (105,107,109):self.step(t,frame)
        self.assertEqual(c.phase,'map_close');self.assertEqual(self.inputs,[])
        self.assertEqual(c.journal['phase_deadline'],220)

    def test_close_requires_stable_map_then_positive_main_confirmation(self):
        c=self.c
        for t in (100,101,103):self.step(t)
        self.assertEqual([a['kind'] for a in self.inputs],['map-close'])
        self.assertEqual(c.phase,'map_close')
        c.view._map.return_value=False;c.normal_idle.return_value=True
        for t in (105,106):self.step(t)
        self.assertEqual(c.phase,'map_close')
        self.step(108);self.assertEqual(c.phase,'travel')
        self.assertEqual(c.journal['deadline'],700)
        self.assertEqual(c.journal['attempts'],{'map-select':1,'map-close':1})

    def test_old_travel_with_open_map_closes_without_renewing_deadline(self):
        c=self.c;c.phase='travel';c.journal['phase']='travel'
        for t in (100,101,103):self.step(t)
        self.assertEqual([a['kind'] for a in self.inputs],['map-close'])
        self.assertEqual(c.phase,'travel');self.assertEqual(c.journal['phase_deadline'],220)
        self.assertEqual(c.journal['deadline'],700)
        self.now=221
        with self.assertRaises(RuntimeError):self.step(221)
        self.assertEqual(len(self.inputs),1)

    def test_map_close_fresh_guard_rejects_movement_context_and_used_limit(self):
        c=self.c;f=np.zeros((4,4,3),np.uint8)
        a=dict(kind='map-close',x=668,y=1187,phase='map_close',signature=c.signature(f))
        self.assertTrue(c.validate(a,f))
        c.bag.moving.return_value=True;self.assertFalse(c.validate(a,f))
        c.bag.moving.return_value=False;c.signature.return_value=('changed',)
        self.assertFalse(c.validate(a,f))
        a['signature']=('changed',);c.journal['attempts']['map-close']=3
        self.assertFalse(c.validate(a,f))


class GuildResumeFishingTests(unittest.TestCase):
    step=GuildReturnMapTests.step

    def setUp(self):
        GuildReturnMapTests.setUp(self);c=self.c
        c.phase='resume_verify';c.journal.update(phase='resume_verify',origin={'route':True,'fishing':True})
        c.same_origin=Mock(return_value=True)
        c.sailing=Mock();c.sailing.running.return_value=True
        c.fishing=Mock();c.fishing.blue_fill.return_value=False
        c.fishing.real_battle.return_value=True
        c.view._idle_fishing.return_value=False
        c.journal['categories']=[0,1]

    def test_short_fights_across_cast_cycles_confirm_without_any_input(self):
        for t in (100,120):self.step(t)
        self.assertIsNone(self.c.done)
        self.step(145)
        self.assertTrue(self.c.done);self.assertEqual(self.inputs,[])
        self.assertEqual(self.c.journal['phase_deadline'],220)
        self.assertEqual(self.c.journal['attempts'],{'map-select':1})

    def test_auto_or_preparation_without_battle_does_not_confirm(self):
        self.c.fishing.real_battle.return_value=False
        for t in (100,120,145):self.step(t)
        self.assertIsNone(self.c.done);self.assertEqual(self.inputs,[])

    def test_context_change_clears_accumulated_battle_proof(self):
        self.step(100);self.step(120)
        self.c.same_origin.return_value=False;self.step(122)
        self.c.same_origin.return_value=True;self.step(125)
        self.assertIsNone(self.c.done)

    def test_resume_deadline_and_cached_frame_still_take_precedence(self):
        f=np.zeros((4,4,3),np.uint8)
        for t in (100,120,140):self.step(t,f)
        self.assertIsNone(self.c.done)
        with self.assertRaises(RuntimeError):self.step(221)
        self.assertEqual(self.c.journal['phase_deadline'],220)
        self.assertEqual(self.inputs,[])


class GuildRealFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        cls.folder=cls.root/'diagnostics/2026-10-24/guild-tasks'
        cls.visual=GuildVisual(cls.root)

    def frame(self,name):
        result=cv2.imread(str(self.folder/name))
        if result is None:self.skipTest('Local live evidence not distributed')
        return cv2.cvtColor(result,cv2.COLOR_BGR2RGB)

    def test_completed_indicator_is_a_beige_submit_button(self):
        f=self.frame('inspect-after-c279c0f5c5ed47708b6d8103e0015de6.png')
        self.assertEqual(self.visual.category(f),1)
        self.assertEqual(self.visual.claim(f),(527,488))
        self.assertFalse(self.visual.claimed_only(f))

    def test_last_submission_returns_disabled_claimed_rows(self):
        f=self.frame('collection-verification/c81fd0a7a47440f6b42c369e3cbe8de8.png')
        self.assertIsNone(self.visual.claim(f))
        self.assertTrue(self.visual.claimed_only(f))

    def test_remote_view_guild_without_task_tab_is_not_task_panel(self):
        f=self.frame('inspect-after-6e652f06e8e74121a5382dbc4f659347.png')
        self.assertFalse(self.visual.panel(f))

    def test_zero_currency_prompt_is_not_an_arrival_screen(self):
        f=self.frame('guild-travel-timeout.png')
        self.assertTrue(self.visual.teleport_prompt(f))
        self.assertIsNone(self.visual.point(f,'guild-location'))

    def test_zero_and_funded_currency_are_read_without_lowering_confidence(self):
        from drift.perception.digit_ocr import DigitReader
        t=GuildDaily.__new__(GuildDaily);t.visual=self.visual;t.reader=DigitReader(self.root)
        t.view=Mock();t.view._parallel.return_value=False;t.view._off_icon.return_value=True
        self.assertEqual(t.teleport_quote(self.frame('guild-travel-timeout.png')),{'balance':0,'cost':10})
        self.assertEqual(t.teleport_quote(self.frame('teleport-menu/5b5f5b1bd08c40b1920c8d0eb911b088.png')),{'balance':600,'cost':10})

    def test_actual_incomplete_map_is_still_map_and_never_main(self):
        folder=self.root/'diagnostics/2026-10-25/ds-followup/approved-finish'
        p=folder/'2b98de33dac94aef9a6f40a5af7e44ae.png'
        if not p.exists():self.skipTest('Local live evidence not distributed')
        f=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB)
        c=GuildDaily(self.root,Mock(),lambda _:False,lambda _:None,'x',state_path=folder/'unused-read-only.json')
        self.assertTrue(c.view._map(f));self.assertFalse(c.stopped_map(f))
        self.assertFalse(c.normal_idle(f))


if __name__=='__main__':unittest.main()
