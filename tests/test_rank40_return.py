"""Actual RANK40 modal/main replays and the bounded return workflow."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2

from parallel_world import ParallelWorld
from digit_ocr import DigitReader
from training_sailing import TrainingSailing
from boss_defense import BossDefense
from boss_training import training_options
from app import Clicker

ROOT=Path(__file__).resolve().parents[1]
FRAMES=ROOT/'tests/fixtures/parallel'

def frame(name):
    return cv2.cvtColor(cv2.imread(str(FRAMES/(name+'.png'))),cv2.COLOR_BGR2RGB)

class Rank40ReturnTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.w=ParallelWorld(ROOT,lambda *a:None,lambda *a:None,lambda _:None,
            DigitReader(ROOT),training_state_path=Path(self.temp.name)/'training.json')
        self.w.sailing=TrainingSailing(ROOT,self.w.training,lambda *a:False,lambda _:None)
        self.w.defense=BossDefense(ROOT,self.w.training,None,lambda _:None)
        self.w.rank=40;self.w.completed_rank_40=True
        self.actions=[]
        self.w.portal_action=self.action
        self.current=frame('rank40-return-main')

    def tearDown(self):self.temp.cleanup()

    def action(self,action):
        accepted=self.w.validate_portal(action,self.current)
        if accepted:self.actions.append(action)
        return accepted

    def step(self,name):
        self.current=frame(name)
        self.w.next_action_at=0
        return self.w.step(self.current)

    def test_actual_main_and_modal_have_distinct_guarded_actions(self):
        main=frame('rank40-return-main');modal=frame('rank40-return-confirm')
        self.assertTrue(self.w._return_ready(main))
        self.assertFalse(self.w._return_ready(main,True))
        self.assertTrue(self.w._return_ready(modal,True))
        self.assertFalse(self.w._return_ready(modal))
        before=deepcopy(self.w.training.state)
        self.w.phase='return_start'
        self.step('rank40-return-main')
        self.assertEqual(self.w.phase,'return_confirm')
        self.step('rank40-return-confirm')
        self.assertEqual(self.w.phase,'await_original')
        self.assertEqual([a['kind'] for a in self.actions],['return-request','return-confirm'])
        self.assertEqual(self.w.training.state,before)
        self.assertEqual(self.w.rounds,0)

    def test_fresh_modal_vanished_or_world_rank_auto_changed_refuses_input(self):
        modal=frame('rank40-return-confirm')
        action=dict(kind='return-confirm',signature=self.w._return_signature(modal))
        self.assertTrue(self.w.validate_portal(action,modal))
        for name in ('rank40-return-main','parallel-return-entry','portal-active-main'):
            self.assertFalse(self.w.validate_portal(action,frame(name)),name)
        for box in ((71,9,136,59),(13,45,60,72),(194,4,246,42),
                    (223,556,506,629),(63,741,356,834)):
            altered=modal.copy();x,y,X,Y=box;altered[y:Y,x:X]=0
            self.assertFalse(self.w.validate_portal(action,altered),box)
        self.assertFalse(self.w.validate_portal(action,modal//2))
        for rank,completed in ((39,True),(40,False)):
            self.w.rank=rank;self.w.completed_rank_40=completed
            self.assertFalse(self.w.validate_portal(action,modal))

    def test_rejected_fresh_input_does_not_advance_phase_or_reset_deadline(self):
        self.w.phase='return_confirm';self.w.phase_since=100;self.w.return_since=90
        self.w.portal_action=lambda action:False
        with patch('parallel_world.time.monotonic',return_value=105):
            self.step('rank40-return-confirm')
        self.assertEqual(self.w.phase,'return_confirm')
        self.assertEqual(self.w.phase_since,100)
        self.assertEqual(self.w.return_since,90)
        self.assertEqual(self.actions,[])

    def test_pending_prompt_does_not_repeat_the_request(self):
        self.w.phase='return_start'
        self.step('rank40-return-confirm')
        self.assertEqual(self.w.phase,'return_confirm')
        self.assertEqual(self.actions,[])
        self.step('rank40-return-confirm')
        self.assertEqual([a['kind'] for a in self.actions],['return-confirm'])

    def test_battle_does_not_request_return_or_cancel_fishing(self):
        self.w.phase='return_start'
        with patch.object(self.w.defense,'_match',return_value=True):
            self.step('rank40-return-main')
        self.assertEqual(self.actions,[])
        self.assertEqual(self.w.phase,'return_start')

    def test_return_deadline_survives_retries_waits_and_suspension(self):
        self.w.phase='await_original';self.w.phase_since=100;self.w.return_since=50
        with patch('parallel_world.time.monotonic',return_value=110):
            self.step('rank40-return-main')
        self.assertEqual(self.w.phase,'return_start')
        self.assertEqual(self.w.return_since,50)
        self.w.suspend(1000)
        self.w.next_action_at=10000
        with patch('parallel_world.time.monotonic',return_value=231):
            with self.assertRaisesRegex(RuntimeError,'返回现实世界.*180 秒'):
                self.w.step(frame('rank40-return-main'))
        self.assertEqual(self.actions,[])

    def test_new_world_clears_old_return_budget(self):
        self.w.return_since=100
        self.w._entered_parallel()
        self.assertIsNone(self.w.return_since)
        self.assertFalse(self.w.completed_rank_40)

    def test_input_rechecks_return_budget_after_outer_frame_work(self):
        modal=frame('rank40-return-confirm')
        self.w.return_since=100
        action=dict(kind='return-confirm',signature=self.w._return_signature(modal))
        with patch('parallel_world.time.monotonic',return_value=279.99):
            self.assertTrue(self.w.validate_portal(action,modal))
        with patch('parallel_world.time.monotonic',return_value=280.01):
            self.assertFalse(self.w.validate_portal(action,modal))

    def replay_worker(self,mode='valid'):
        modal=frame('rank40-return-confirm');clock=[0.];actions=[]
        class Device:
            def __init__(self,config,stop):
                self.stop=stop;self.io_lock=threading.RLock();self.count=0
            def screenshot(self):
                self.count+=1;clock[0]+=.05
                return modal.copy()
            def screenshot_fast(self):
                self.count+=1;clock[0]+=.4 if mode=='changed' else .05
                if mode=='stale':clock[0]+=.2
                if mode=='stop':self.stop.set()
                if mode=='changed':return frame('rank40-return-main')
                return modal.copy()
            def touch_fast(self,*a):raise AssertionError('No short SDK modal contact')
            def click(self,*a):raise AssertionError('No ADB or extra contact')
            def command(self,*a):
                if self.stop.is_set() or a!=('shell','input','swipe','208','788','208','788','80'):
                    raise AssertionError('Unexpected shell input')
                actions.append((208,788));self.stop.set()
            def stuck_record_add(self,_):pass
        root=Path(self.temp.name)/'worker';root.mkdir()
        shutil.copytree(ROOT/'assets',root/'assets')
        config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        config['rules']=[deepcopy(r) for r in config['rules'] if r['id']=='driftinggirl_parallel_world']
        config['rules'][0].update(enabled=True,training=training_options())
        config['poll_interval']=.1
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        def controller(*a,**kw):
            w=ParallelWorld(*a,**kw)
            w.phase='return_confirm';w.phase_since=0;w.return_since=0
            w.rank=40;w.completed_rank_40=True
            return w
        app=Clicker(root,Device);app.digit_reader=DigitReader(ROOT)
        try:
            with patch('app.ParallelWorld',side_effect=controller), \
                 patch('app.time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:clock[0])), \
                 patch('parallel_world.time',SimpleNamespace(monotonic=lambda:clock[0])):
                app.start(preview=False,max_clicks=1,timeout=1)
                app.thread.join(10)
            self.assertFalse(app.running())
            self.assertEqual(actions,[(208,788)] if mode=='valid' else [])
            self.assertIsNone(app.state['last_error'])
            self.assertLessEqual(app.state['clicks'],1)
        finally:
            app.stop()
            if app.thread:app.thread.join(3)
            for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_worker_fresh_modal_stationary_input_is_counted_once(self):self.replay_worker()
    def test_worker_stale_return_frame_never_touches(self):self.replay_worker('stale')
    def test_worker_changed_return_frame_never_touches(self):self.replay_worker('changed')
    def test_worker_stop_during_fresh_return_capture_never_touches(self):self.replay_worker('stop')
