"""Slow recognition must not turn an expiring workflow into another long input."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from drift.runtime_inputs import build_handlers
from drift.tasks.equipment_upgrade import EquipmentUpgrade
from drift.tasks.cruise_settings import SettingsView
from tests.test_equipment_upgrade import image
from tests.test_cruise_preferences import Harness

ROOT=Path(__file__).resolve().parents[1]


class EquipmentBudgetTests(unittest.TestCase):
    def make(self):
        actions=[]
        u=EquipmentUpgrade(ROOT,lambda *a:actions.append(('tap',a)),
            lambda *a:actions.append(('swipe',a)),lambda *a:actions.append(('hold',a)) or True,
            lambda _:None)
        u.phase='upgrade';u.started_at=100.;u.phase_since=100.;u.stable=1
        return u,actions

    def test_boundary_includes_hold_settle_and_close_time(self):
        u,_=self.make()
        last=100+u.CYCLE_SECONDS-u.HOLD_SECONDS-u.ACTION_SETTLE_SECONDS-u.CLOSE_RESERVE_SECONDS
        with patch('drift.tasks.equipment_upgrade.time.monotonic',return_value=last):
            self.assertTrue(u.hold_budget_available(12))
        with patch('drift.tasks.equipment_upgrade.time.monotonic',return_value=last+.001):
            self.assertFalse(u.hold_budget_available(12))
        self.assertEqual(u.started_at,100)

    def test_late_cycle_closes_without_another_hold_or_completed_cycle(self):
        u,actions=self.make()
        with patch('drift.tasks.equipment_upgrade.time.monotonic',return_value=260.):
            u.step(image('top'))
            self.assertEqual(u.phase,'close')
            self.assertTrue(u.interrupted)
            self.assertEqual(actions,[])
            u.step(image('top'))
            self.assertEqual([k for k,_ in actions],['tap'])
            u.next_action_at=0
            for _ in range(3):u.step(image('main'))
        self.assertFalse(u.active)
        self.assertEqual((u.holds,u.cycles),(0,0))
        self.assertEqual(u.started_at,100)

    def test_fully_completed_pass_is_still_counted(self):
        u,actions=self.make();u.holds=u.MAX_HOLDS
        with patch('drift.tasks.equipment_upgrade.time.monotonic',return_value=279.):
            u.step(image('top'));u.next_action_at=0
            for _ in range(3):u.step(image('main'))
        self.assertEqual(u.cycles,1)
        self.assertEqual([k for k,_ in actions],['tap'])


class RuntimeHoldBudgetTests(unittest.TestCase):
    def make(self,*,capture_delay=0,validate_delay=0,run_deadline=None,max_clicks=None):
        clock=[100.];commands=[];captures=[]
        def capture():clock[0]+=capture_delay;captures.append(True);return object()
        def validate(*a):clock[0]+=validate_delay;return True
        equipment=SimpleNamespace(validate_hold=validate,hold_budget_available=lambda s:clock[0]+s+11<=130)
        owner=SimpleNamespace(device=SimpleNamespace(io_lock=threading.RLock(),command=lambda *a:commands.append(a)),
            stop_event=threading.Event(),lock=threading.RLock(),state={'clicks':0,'matches':0},
            remember=lambda f:'fresh',event=lambda _:None)
        ctx={'self':owner,'runtime':SimpleNamespace(capture=capture),'equipment':equipment,
             'parallel':None,'stone':None,'st':SimpleNamespace(run_deadline=run_deadline),'max_clicks':max_clicks}
        handler=build_handlers(lambda:ctx,{'_time':lambda:SimpleNamespace(monotonic=lambda:clock[0])})['equipment_hold']
        return handler,owner,commands,captures

    def test_slow_new_capture_consumes_budget_and_issues_no_input(self):
        f,owner,commands,captures=self.make(capture_delay=8)
        self.assertFalse(f(620,886,12,'rod','rod'))
        self.assertEqual(captures,[True]);self.assertEqual(commands,[])
        self.assertEqual(owner.state['clicks'],0)

    def test_slow_visual_validation_is_rechecked_immediately_before_contact(self):
        f,owner,commands,_=self.make(validate_delay=8)
        self.assertFalse(f(620,886,12,'rod','rod'))
        self.assertEqual(commands,[]);self.assertEqual(owner.state['clicks'],0)

    def test_sufficient_budget_keeps_one_full_twelve_second_hold(self):
        f,owner,commands,_=self.make()
        self.assertTrue(f(620,886,12,'rod','rod'))
        self.assertEqual(len(commands),1);self.assertEqual(commands[0][-1],'12000')
        self.assertEqual(owner.state['clicks'],1)

    def test_run_time_limit_covers_the_whole_contact(self):
        f,owner,commands,captures=self.make(run_deadline=111)
        self.assertFalse(f(620,886,12,'rod','rod'))
        self.assertEqual(captures,[]);self.assertEqual(commands,[])
        self.assertEqual(owner.state['clicks'],0)

    def test_exhausted_click_limit_never_starts_capture_or_contact(self):
        f,owner,commands,captures=self.make(max_clicks=0)
        with self.assertRaises(InterruptedError):f(620,886,12,'rod','rod')
        self.assertEqual(captures,[]);self.assertEqual(commands,[])


class TicketBudgetTests(unittest.TestCase):
    def test_last_safe_boundary_and_no_reservation_after_it(self):
        with TemporaryDirectory() as root:
            h=Harness(root);c=h.controller;c.state='tickets';c.deadline=200
            c.journal['attempts']={'tickets':2};h.now=168
            self.assertTrue(c.allowed('tickets'))
            h.now+=.001
            before=deepcopy(c.journal)
            self.assertFalse(c.reserve('tickets'))
            self.assertEqual(c.journal,before);self.assertEqual(c.deadline,200)

    def test_optional_ticket_hold_is_skipped_and_existing_route_resumes(self):
        with TemporaryDirectory() as root:
            h=Harness(root,False);c=h.controller;h.menu=True
            h.values=SettingsView.wanted(h.options)
            c.state='tickets';c.scene=h.scene;c.deadline=h.now+31
            c.journal.update(pending=True,started_at=h.now,deadline=c.deadline,
                             attempts={'stop':1,'open':1,'tickets':2})
            deadline=c.deadline;h.tick(12)
            self.assertEqual([k for k,_ in h.receipts],['start'])
            self.assertTrue(h.route);self.assertFalse(h.menu)
            result=c.journal['last_result']
            self.assertTrue(result['success']);self.assertEqual(result['deadline'],deadline)
            self.assertEqual(result['attempts'],{'stop':1,'open':1,'tickets':2,'start':1})
            self.assertEqual(h.count,1)

    def test_slow_menu_recognition_rechecks_close_reserve_with_current_clock(self):
        with TemporaryDirectory() as root:
            h=Harness(root,False);c=h.controller;h.menu=True
            h.values=SettingsView.wanted(h.options);c.state='menu';c.scene=h.scene
            c.deadline=200;h.now=190
            c.journal.update(pending=True,deadline=200,attempts={})
            def observe(frame):
                h.now+=6
                return dict(scene=h.scene,menu=True)
            h.view.observe_setup=observe
            c.step(object())
            self.assertEqual(h.receipts,[('close',40)])
            self.assertFalse(c.busy)
            self.assertEqual(c.journal['last_result']['deadline'],200)
            self.assertFalse(c.journal['last_result']['success'])


if __name__=='__main__':unittest.main()
