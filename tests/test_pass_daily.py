"""Actual PASS cards, one input, independent verification and daily timing."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from app import Clicker
from drift.perception.digit_ocr import DigitReader
from drift.tasks.pass_daily import PassDaily,PASS_ID
from drift.tasks.daily import DailySchedule,BEIJING
from drift.tasks.schedule import TaskSchedule
from dev_tools.configure_pass_daily import add

ROOT=Path(__file__).resolve().parents[1]

class PassDailyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)
        cls.images={name:np.asarray(Image.open(ROOT/'assets/pass'/(name+'.png')).convert('RGB'))
                    for name in ('entry','daily-ready','daily-claimed')}

    def setUp(self):
        self.inputs=[];self.frame=self.images['daily-ready']
        def action(kind,x,y,context):
            self.assertTrue(self.c.validate(kind,self.frame,context,'2026-10-12'))
            self.assertTrue(self.c.reserve(kind));self.inputs.append((kind,x,y));return True
        self.c=PassDaily(ROOT,self.reader,action,lambda *_:None)

    def steps(self,name,due=True):
        self.frame=self.images[name]
        for _ in range(3):
            self.c.step(self.frame.copy(),due=due,period='2026-10-12',main_ready=name=='entry')

    def test_actual_ready_and_all_four_free_checks_are_distinct(self):
        self.assertEqual(self.c.count(self.images['daily-ready']),1491)
        self.assertEqual(self.c.count(self.images['daily-claimed']),1589)
        self.assertTrue(self.c.ready(self.images['daily-ready']))
        self.assertFalse(self.c.claimed(self.images['daily-ready']))
        self.assertTrue(self.c.claimed(self.images['daily-claimed']))
        self.assertFalse(self.c.ready(self.images['daily-claimed']))

    def test_claim_once_then_checks_close_and_actual_main_before_complete(self):
        self.steps('daily-ready')
        self.assertEqual(self.inputs,[('claim',205,530)])
        self.steps('daily-ready')
        self.assertEqual(len(self.inputs),1)
        self.steps('daily-claimed')
        self.assertEqual(self.inputs[-1],('close',637,280))
        self.assertIsNone(self.c.done)
        self.steps('entry')
        self.assertTrue(self.c.done);self.assertFalse(self.c.active)
        self.assertEqual(self.c.cycles,1)

    def test_previously_claimed_page_only_closes_without_reclaim(self):
        self.steps('daily-claimed');self.steps('entry')
        self.assertEqual(self.inputs,[('close',637,280)])
        self.assertTrue(self.c.done)

    def test_before_due_has_no_menu_input(self):
        self.steps('entry',due=False);self.steps('daily-ready',due=False)
        self.assertFalse(self.c.active);self.assertEqual(self.inputs,[])

    def test_under_120_fish_closes_without_completion(self):
        with patch.object(self.c,'count',return_value=119):self.steps('daily-ready')
        self.steps('entry')
        self.assertEqual(self.inputs,[('close',637,280)])
        self.assertFalse(self.c.done);self.assertEqual(self.c.cycles,0)

    def test_real_17_fish_low_confidence_closes_without_claim_or_completion(self):
        real=np.asarray(Image.open(ROOT/'tests/fixtures/pass-17-20261030.png').convert('RGB'))
        self.assertTrue(self.c.daily_page(real))
        self.assertIsNone(self.c.count(real))
        self.frame=real
        for _ in range(3):
            self.c.step(real.copy(),due=True,period='2026-10-12',main_ready=False)
        deadline=self.c.deadline
        self.assertEqual(self.inputs,[('close',637,280)])
        self.assertFalse(self.c.claim_issued);self.assertFalse(self.c.claim_verified)
        self.steps('entry')
        self.assertFalse(self.c.done);self.assertEqual(self.c.cycles,0)
        self.assertEqual(self.c.deadline,deadline)

    def test_unknown_count_cannot_claim_or_close_an_unrecognized_page(self):
        self.c.step(self.frame,due=True,period='2026-10-12',main_ready=False)
        context=self.c.context(self.frame)
        with patch.object(self.c,'count',return_value=None):
            self.assertFalse(self.c.validate('claim',self.frame,context,'2026-10-12'))
            for bad in (self.frame//2,np.zeros_like(self.frame)):
                self.frame=bad
                for _ in range(3):
                    self.c.step(bad.copy(),due=True,period='2026-10-12',main_ready=False)
                self.assertFalse(self.c.validate('close',bad,context,'2026-10-12'))
        self.assertEqual(self.inputs,[])

    def test_changed_fresh_page_period_deadline_and_missing_check_reject(self):
        self.c.step(self.frame,due=True,period='2026-10-12',main_ready=False)
        context=self.c.context(self.frame)
        self.assertTrue(self.c.validate('claim',self.frame,context,'2026-10-12'))
        for box in ((605,244,684,320),(283,306,436,335),(126,952,576,977),(72,10,136,62)):
            bad=self.frame.copy();x,y,X,Y=box;bad[y:Y,x:X]=0
            self.assertFalse(self.c.validate('claim',bad,context,'2026-10-12'))
        self.assertFalse(self.c.validate('claim',self.frame//2,context,'2026-10-12'))
        self.assertFalse(self.c.validate('claim',self.frame,context,'2026-10-13'))
        with patch('drift.tasks.pass_daily.time.monotonic',return_value=self.c.deadline):
            self.assertFalse(self.c.validate('claim',self.frame,context,'2026-10-12'))
        bad=self.images['daily-claimed'].copy();bad[867:913,169:238]=0
        self.assertFalse(self.c.claimed(bad));self.assertEqual(self.inputs,[])

    def test_pass_rule_append_preserves_existing_rules_and_parameters(self):
        original={'rules':[{'id':'other','enabled':False}],'poll_interval':.6}
        result=add(original)
        self.assertEqual(result['rules'][:-1],original['rules'])
        self.assertEqual(result['poll_interval'],.6)
        self.assertEqual(add(result),result)
        self.assertEqual(len(original['rules']),1)

    def test_battle_main_without_pass_entry_releases_only_after_new_frames_and_time(self):
        main=np.asarray(Image.open(ROOT/'tests/fixtures/pass-interrupted-main.png').convert('RGB'))
        self.assertFalse(self.c.entry(main))
        self.assertTrue(self.c.main_visible(main))
        self.assertFalse(self.c.main_visible(main//2))
        for name in ('daily-ready','daily-claimed'):
            self.assertFalse(self.c.main_visible(self.images[name]))
        self.c.phase='page';self.c.deadline=160;self.c.last_action_at=100
        self.c.started_period='2026-10-12';self.c.claim_issued=True
        same=main.copy()
        for now,img in ((103,same),(107,same),(108,main.copy())):
            with patch('drift.tasks.pass_daily.time.monotonic',return_value=now):
                self.c.step(img,due=True,period='2026-10-12',main_ready=True)
            self.assertTrue(self.c.active)
        with patch('drift.tasks.pass_daily.time.monotonic',return_value=109):
            self.c.step(main.copy(),due=True,period='2026-10-12',main_ready=True)
        self.assertFalse(self.c.active);self.assertFalse(self.c.done)
        self.assertEqual(self.c.deadline,160);self.assertEqual(self.c.cycles,0)
        self.assertTrue(self.c.claim_issued);self.assertEqual(self.inputs,[])

    def test_unknown_or_changed_context_resets_readonly_return_proof(self):
        main=self.images['entry'];self.c.phase='page';self.c.deadline=160
        self.c.last_action_at=100;self.c.started_period='2026-10-12'
        for now,img in ((103,main.copy()),(105,main.copy()),(106,main//2),(107,main.copy()),(109,main.copy())):
            with patch('drift.tasks.pass_daily.time.monotonic',return_value=now):
                self.c.step(img,due=True,period='2026-10-12',main_ready=True)
            self.assertTrue(self.c.active)
        with patch('drift.tasks.pass_daily.time.monotonic',return_value=160):
            with self.assertRaisesRegex(RuntimeError,'60秒'):
                self.c.step(main.copy(),due=True,period='2026-10-12',main_ready=True)
        self.assertEqual(self.inputs,[])

    def test_reset_plus_two_hours_and_restart_preserve_other_completions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);now=[datetime(2026,10,12,9,59,59,tzinfo=BEIJING)]
            daily=DailySchedule(root,'test',lambda:now[0])
            task=TaskSchedule(root,'test',lambda:now[0])
            daily.mark_complete('daily_reward',daily.period())
            original=deepcopy(daily.completed['daily_reward'])
            self.assertFalse(task.ready('pass_daily',daily.status()['pass_daily']))
            task.set_next('pass_daily',None,daily.status()['pass_daily'])
            self.assertFalse(task.ready('pass_daily',daily.status()['pass_daily']))
            now[0]=datetime(2026,10,12,10,tzinfo=BEIJING)
            self.assertTrue(task.ready('pass_daily',daily.status()['pass_daily']))
            daily.mark_complete('pass_daily',daily.period())
            task.finish('pass_daily',600,daily.status()['pass_daily'])
            self.assertEqual(task.next_run('pass_daily',daily.status()['pass_daily']).isoformat(),'2026-10-13T10:00:00+08:00')
            restored=DailySchedule(root,'test',lambda:now[0])
            self.assertEqual(restored.completed['daily_reward'],original)
            self.assertTrue(restored.is_complete('pass_daily'))
            now[0]=datetime(2026,10,13,8,tzinfo=BEIJING)
            self.assertFalse(restored.is_complete('pass_daily'))
            self.assertFalse(task.ready('pass_daily',restored.status()['pass_daily']))

    def test_scoped_worker_claims_and_preserves_other_daily_records(self):
        outer=self;clock=[0.];devices=[]
        class Device:
            def __init__(self,config,stop):
                self.stop=stop;self.actions=[];self.io_lock=threading.RLock();devices.append(self)
            def screenshot(self):
                clock[0]+=.3
                name='entry' if self.actions and self.actions[-1]==(637,280) else 'daily-claimed' if self.actions else 'daily-ready'
                self.image=outer.images[name].copy();return self.image
            def click(self,button):
                a,b,c,d=button.button;self.actions.append(((a+c)//2,(b+d)//2))
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets/pass',root/'assets/pass')
            config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules']=[deepcopy(r) for r in config['rules'] if r['id']==PASS_ID]
            config['poll_interval']=.1
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device);app.digit_reader=self.reader
            app.daily_schedule.clock=lambda:datetime(2026,10,12,12,tzinfo=BEIJING)
            app.task_schedule.clock=app.daily_schedule.clock
            app.daily_schedule.mark_complete('daily_reward',app.daily_schedule.period())
            original=deepcopy(app.daily_schedule.completed['daily_reward'])
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:1791806400+clock[0])):
                    app.start(preview=False,stop_on_main=True,max_clicks=4,timeout=35)
                    app.thread.join(10)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(devices[0].actions,[(205,530),(637,280)])
                self.assertTrue(app.daily_schedule.is_complete('pass_daily'))
                self.assertEqual(app.daily_schedule.completed['daily_reward'],original)
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

if __name__=='__main__':unittest.main()
