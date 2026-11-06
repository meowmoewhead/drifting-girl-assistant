from copy import deepcopy
from datetime import datetime,timedelta,timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from drift.core.continuity import FreshConfirmation, reconcile_late_departure
from drift.core.lifecycle import BusinessProgress, UiOwnership, RuntimeLifecycle
from drift.core.outcome import result,Outcome
from drift.tasks.schedule import TaskSchedule


class ContinuityTests(unittest.TestCase):
    def test_cached_frame_is_not_three_confirmations(self):
        proof=FreshConfirmation()
        for now in (0,1,4):self.assertFalse(proof.observe('same','cached',now))
        self.assertFalse(proof.observe('same','two',5))
        self.assertTrue(proof.observe('same','three',6))

    def test_changed_context_and_gap_restart_confirmation_only(self):
        proof=FreshConfirmation();proof.observe('a','1',0);proof.observe('a','2',1)
        self.assertFalse(proof.observe('b','3',4))
        self.assertFalse(proof.observe('b','4',12))

    def test_late_departure_updates_target_but_preserves_budget(self):
        old={'parallel':True,'stage':10,'rank':37,'route':True,'fishing':True}
        origin={**old,'rank':11}
        j={'target':old,'farm_origin':deepcopy(old),'resume_pending':True,
           'deadline':300,'phase_deadline':180,'attempts':{'stop':1}}
        valid=lambda x:isinstance(x,dict) and x.get('parallel') is True
        self.assertTrue(reconcile_late_departure(j,origin,valid=valid))
        self.assertEqual(j['target']['rank'],11)
        self.assertTrue(j['capture_target_settings'])
        self.assertEqual((j['deadline'],j['phase_deadline'],j['attempts']),(300,180,{'stop':1}))
        self.assertEqual(j['departure_reconciliation']['previous_target']['rank'],37)

    def test_spent_departure_or_return_only_never_retargets(self):
        for update in ({'return_only':True},{'attempts':{'return-confirm':1}},
                       {'attempts':{'snapshot-open':1}}):
            j={'target':{'parallel':True,'stage':10,'rank':37},'attempts':{},**update};before=deepcopy(j)
            self.assertFalse(reconcile_late_departure(j,{'parallel':True,'stage':10,'rank':11,
                'route':True,'fishing':True},valid=lambda x:bool(x)))
            self.assertEqual(j,before)

    def test_original_world_never_overwrites_owed_parallel_target(self):
        j={'target':{'parallel':True,'stage':10,'rank':11},'attempts':{}};before=deepcopy(j)
        self.assertFalse(reconcile_late_departure(j,{'parallel':False},valid=lambda x:bool(x and x.get('parallel'))))
        self.assertEqual(j,before)


class BusinessHealthTests(unittest.TestCase):
    def setUp(self):
        self.now=0;self.events=[]
        self.h=BusinessProgress(lambda:self.now,self.events.append,idle_limit=9)

    def note(self,id,now,**kw):
        self.now=now;return self.h.observe(id,**kw)

    def test_idle_running_route_is_not_fishing(self):
        for i,now in enumerate((0,3,6,9)):v=self.note(str(i),now,context='rank0',kind='idle')
        self.assertEqual(v['state'],'stalled');self.assertEqual(len(self.events),1)
        self.assertIsNone(v['last_fishing_seconds_ago'])

    def test_cached_frames_never_prove_stall(self):
        for now in (0,3,9,30):v=self.note('cached',now,context='rank0',kind='idle')
        self.assertNotEqual(v['state'],'stalled')

    def test_fill_or_actual_battle_is_evidence_and_clears_warning(self):
        self.test_idle_running_route_is_not_fishing()
        v=self.note('battle',10,context='resource-r51',kind='battle')
        self.assertEqual(v['state'],'fishing')
        self.assertEqual(v['recent_evidence'][-1]['kind'],'battle')

    def test_menu_and_unknown_frames_do_not_infer_sleep(self):
        self.note('one',0,context='a',kind='idle')
        self.assertEqual(self.note('two',100,owner='restaurant')['state'],'occupied')
        self.assertEqual(self.note('three',200)['state'],'unknown')
        self.assertEqual(self.note('four',201,context='a',kind='idle')['state'],'waiting')

    def test_port_wait_is_distinct_and_notice_is_throttled(self):
        for i,now in enumerate((0,3,6,9,12,15)):
            v=self.note(str(i),now,context='port',kind='port')
        self.assertEqual(v['state'],'stalled');self.assertIn('港口',v['reason'])
        self.assertEqual(len(self.events),1)


class OwnershipTests(unittest.TestCase):
    def test_new_task_cannot_steal_current_ui(self):
        task=NS(active=True,journal={'deadline':100,'attempts':{'feed':1}},phase='detail')
        ui=UiOwnership(lambda:('main',{'restaurant_upgrade':task},False))
        self.assertFalse(ui.can_call('cruise_setup'))
        self.assertTrue(ui.can_call('restaurant_upgrade'))
        self.assertTrue(ui.can_call('observe'))

    def test_popup_context_keeps_original_budget(self):
        state={'popup':False};journal={'deadline':100,'attempts':{'submit':1}}
        task=NS(active=True,journal=journal,phase='collect')
        ui=UiOwnership(lambda:('main',{'guild_daily':task},state['popup']))
        state['popup']=True;self.assertFalse(ui.can_call('guild_daily'))
        self.assertEqual(ui.interrupted['owners']['guild_daily']['deadline'],100)
        state['popup']=False;self.assertTrue(ui.can_call('guild_daily'))
        self.assertTrue(ui.last_resume['requires_fresh_controller_validation'])
        self.assertEqual(journal,{'deadline':100,'attempts':{'submit':1}})

    def test_busy_method_false_does_not_block_all_tasks(self):
        task=NS(active=False,busy=lambda:False,journal={})
        ui=UiOwnership(lambda:('main',{'cruise_setup':task},False))
        self.assertTrue(ui.can_call('guild_daily'))


class OutcomeTests(unittest.TestCase):
    def test_failed_long_interval_retries_in_half_hour(self):
        self.assertEqual(result(False).delay(172800),1800)
        self.assertEqual(result(True).delay(172800),172800)
        self.assertEqual(result(False,retry_seconds=47).delay(172800),47)
        self.assertEqual(result(outcome='skipped').delay(900),900)

    def test_failed_daily_does_not_mark_complete_or_use_next_day(self):
        with tempfile.TemporaryDirectory() as d:
            now=datetime(2026,11,5,20,tzinfo=timezone(timedelta(hours=8)))
            s=TaskSchedule(d,'test',lambda:now)
            s.finish('guild_daily',86400,success=False,detail='未确认返程')
            self.assertEqual((s.next_run('guild_daily')-now).total_seconds(),1800)
            entry=s.entries['guild_daily'];self.assertEqual(entry['outcome'],'retry')
            self.assertEqual(entry['detail'],'未确认返程')

    def test_completed_retry_and_manual_dates_are_not_rebased(self):
        with tempfile.TemporaryDirectory() as d:
            now=datetime(2026,11,5,tzinfo=timezone.utc);s=TaskSchedule(d,'test',lambda:now)
            s.finish('restaurant_upgrade',172800,success=False)
            before=deepcopy(s.entries)
            s.refresh_intervals([{'id':'driftinggirl_restaurant_upgrade','interval':172800}],
                                [{'id':'driftinggirl_restaurant_upgrade','interval':3600}])
            self.assertEqual(s.entries,before)


if __name__=='__main__':unittest.main()
