"""Real missed-close evidence and bounded full-runtime mailbox replays."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from app import Clicker
from drift.device.adb import AdbDevice
from drift.ui.mailbox_close import MailboxCloseGuard, CLOSES

ROOT = Path(__file__).resolve().parents[1]


class MailboxCloseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.config['rules'] = [r for r in cls.config['rules'] if r['id'].startswith('driftinggirl_mailbox_')]
        for r in cls.config['rules']:r['enabled'] = True
        cls.empty = cls.image('tests/fixtures/mailbox-close/close-missed.png')
        cls.current = cls.image('tests/fixtures/mailbox-close/still-open-rank38.png')
        cls.main = cls.image('tests/fixtures/legacy-captures/mailbox-return.png')

    @staticmethod
    def image(name):
        with Image.open(ROOT/name) as im:return np.asarray(im.convert('RGB'))

    def setUp(self):
        ctx = SimpleNamespace(root=ROOT, config=self.config)
        self.matchers = {r['id']:Clicker.load_rule(ctx,r) for r in self.config['rules']}
        self.guard = MailboxCloseGuard()
        self.origin = self.guard.context(self.empty,self.matchers)
        self.state = SimpleNamespace(matchers=self.matchers,stage='mailbox_return',
            stage_trigger=self.matchers[CLOSES[0]][0]['name'],stage_deadline=45,
            stage_ready_at=.8,handling_interrupt=False)
        self.guard.issued(self.origin,retry=False)

    def test_real_empty_modal_recognized_despite_background_rank_progress(self):
        self.assertEqual(self.origin,(CLOSES[0],(612,170,690,230)))
        self.assertEqual(self.guard.context(self.current,self.matchers),self.origin)

    def test_empty_text_cross_tabs_claim_control_and_brightness_are_required(self):
        for box in ((298,570,408,596),(615,175,685,225),(144,228,197,253),(488,987,582,1019)):
            with self.subTest(box=box):
                frame=self.empty.copy();x1,y1,x2,y2=box;frame[y1:y2,x1:x2]=0
                self.assertIsNone(self.guard.context(frame,self.matchers,CLOSES[0]))
        for frame in (self.main,(self.empty*.4).astype(np.uint8),
                      self.image('assets/mailbox-loading-native.png'),
                      self.image('tests/fixtures/legacy-captures/mailbox-before-claim.png'),
                      self.image('assets/navigation/parallel-boss-battle.png')):
            self.assertIsNone(self.guard.context(frame,self.matchers,CLOSES[0]))

    def test_retry_needs_three_consecutive_frames_over_three_seconds(self):
        self.assertFalse(self.guard.ready(self.empty,self.state,now=1))
        self.assertFalse(self.guard.ready(self.empty,self.state,now=3))
        self.assertTrue(self.guard.ready(self.empty,self.state,now=4))
        self.guard.issued(self.origin,retry=True)
        for now in (5,10,20,40):self.assertFalse(self.guard.ready(self.empty,self.state,now=now))
        with self.assertRaises(RuntimeError):self.guard.issued(self.origin,retry=True)

    def test_missing_modal_parts_are_not_proof_of_a_closed_mailbox(self):
        for box in ((615,175,685,225),(144,228,197,253),(488,987,582,1019)):
            frame=self.empty.copy();x1,y1,x2,y2=box;frame[y1:y2,x1:x2]=0
            self.assertIsNone(self.guard.context(frame,self.matchers,CLOSES[0]))
            self.assertTrue(self.guard.page_visible(frame,self.matchers))
        self.assertFalse(self.guard.page_visible(self.main,self.matchers))
        self.assertTrue(self.guard.page_visible(np.zeros((1,1,3),dtype=np.uint8),self.matchers))

    def test_interruption_unknown_page_stop_and_deadline_block_retry(self):
        for options in ({'stopped':True},{'interrupted':True}):
            for now in (1,3,4):self.assertFalse(self.guard.ready(self.empty,self.state,now=now,**options))
        for field,value in (('stage','main'),('stage_trigger','other'),('stage_deadline',4),
                            ('stage_deadline',None),('handling_interrupt',True)):
            state=deepcopy(self.state);setattr(state,field,value)
            self.assertFalse(self.guard.ready(self.empty,state,now=4))
        self.assertFalse(self.guard.ready(self.empty,self.state,now=1))
        self.assertFalse(self.guard.ready(np.zeros_like(self.empty),self.state,now=3))
        self.assertFalse(self.guard.ready(self.empty,self.state,now=4))
        self.assertFalse(self.guard.ready(self.empty,self.state,now=6))
        self.assertTrue(self.guard.ready(self.empty,self.state,now=7))

    def test_loading_does_not_mutate_saved_options_or_custom_geometry(self):
        config=deepcopy(self.config);ctx=SimpleNamespace(root=ROOT,config=config)
        for rule in config['rules']:Clicker.load_rule(ctx,rule)
        self.assertEqual(config,self.config)
        rule=deepcopy(next(r for r in config['rules'] if r['id']==CLOSES[0]))
        rule['click_source_area']=[613,170,690,230]
        loaded,button,_=Clicker.load_rule(ctx,rule)
        self.assertEqual(loaded,rule)
        self.assertFalse(button.mailbox_close_guarded)

    def test_online_rule_removal_or_disable_cannot_authorize_retry(self):
        state=deepcopy(self.state);state.matchers.pop(CLOSES[0])
        self.assertFalse(self.guard.ready(self.empty,state,now=4))
        state=deepcopy(self.state);state.matchers[CLOSES[0]][0]['enabled']=False
        self.assertFalse(self.guard.ready(self.empty,state,now=4))

    def test_mailbox_close_contacts_are_single_80ms_and_stop_aware(self):
        for name in CLOSES:
            device=AdbDevice(dict(resolution=[720,1280]),threading.Event());device.command=Mock()
            button=SimpleNamespace(name=name,button=(612,170,690,230))
            device.click(button)
            device.command.assert_called_once_with('shell','input','swipe','651','200','651','200','80')
            device.command.reset_mock();device.stop.set()
            with self.assertRaises(InterruptedError):device.click(button)
            device.command.assert_not_called()

    def replay(self, mode, max_clicks=5):
        frames=dict(main=self.image('tests/fixtures/legacy-captures/other-functions-start.png'),
            claim=self.image('tests/fixtures/legacy-captures/mailbox-before-claim.png'),empty=self.empty,returned=self.main)
        clock=[0];issued=[];fresh_miss=[None]
        ready=MailboxCloseGuard.ready
        def wrapped_ready(guard,*args,**kwargs):
            result=ready(guard,*args,**kwargs)
            if result and mode.startswith('fresh-'):fresh_miss[0]=mode
            return result
        class Device:
            def __init__(self,config,stop):self.stop,self.image,self.taps=stop,None,[]
            def screenshot(self):
                clock[0]+=1
                n=len(self.taps)
                name=('empty' if mode=='orphan' and n==0 else 'returned' if mode=='orphan' else
                      'main' if n==0 else 'claim' if n==1 else 'empty'
                      if n<4 or mode=='both-missed' else 'returned')
                if fresh_miss[0]=='fresh-stop':self.stop.set()
                if fresh_miss[0]=='fresh-deadline':clock[0]=max(clock[0],issued[2][1]+45)
                self.image=frames[name].copy()
                if fresh_miss[0]=='fresh-changed':self.image[175:225,615:685]=0
                return self.image
            def click(self,button):
                if self.stop.is_set():raise InterruptedError()
                self.taps.append(button.name);issued.append((button.name,clock[0]))
            def stuck_record_add(self,button):pass
        config=deepcopy(self.config);config['poll_interval']=.1
        for rule in config['rules']:rule.update(interval=.2,next_delay=0)
        with tempfile.TemporaryDirectory(prefix='mailbox-close-') as directory:
            root=Path(directory)
            for path in {r['asset'] for r in config['rules']} | {p for r in config['rules'] for p in r.get('asset_variants',[])}:
                (root/path).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,root/path)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])),patch.object(MailboxCloseGuard,'ready',wrapped_ready):
                    app.start(preview=False,stop_on_main=True,max_clicks=max_clicks,timeout=180)
                    app.thread.join(15)
                self.assertFalse(app.running())
                state=deepcopy(app.state)
                state['replay_end']=clock[0]
                receipts=[json.loads(p.read_text(encoding='utf-8')) for p in (root/'logs/mailbox-input').glob('input-*.json')]
                return state,issued,receipts,app.daily_schedule.is_complete('mailbox')
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_missed_close_retries_once_never_claims_again_and_records_new_frames(self):
        state,issued,receipts,_=self.replay('missed')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'],'cycle_complete')
        self.assertEqual([x[0] for x in issued],['driftinggirl_mailbox_open','driftinggirl_mailbox_claim_all',*([CLOSES[0]]*2)])
        self.assertEqual(len(receipts),2)
        retry=next(r for r in receipts if r['retry'])
        self.assertEqual(retry['attempt'],2)
        self.assertEqual(retry['original_return_deadline'],issued[2][1]+45)
        self.assertGreaterEqual(issued[3][1]-issued[2][1],4)
        for receipt in receipts:
            self.assertTrue(receipt['issued_only'])
            self.assertNotEqual(receipt['before_frame_id'],receipt['after_frame_id'])

    def test_two_missed_closes_still_stop_at_original_deadline(self):
        state,issued,_,_=self.replay('both-missed')
        self.assertIn('未识别到后续界面',state['last_error'])
        self.assertEqual(len(issued),4)
        self.assertGreaterEqual(state['replay_end']-issued[2][1],45)
        self.assertLess(state['replay_end']-issued[2][1],47)

    def test_stop_in_fresh_preinput_frame_suppresses_retry(self):
        state,issued,_,_=self.replay('fresh-stop')
        self.assertIsNone(state['last_error'])
        self.assertEqual(len(issued),3)

    def test_formal_click_limit_is_not_bypassed(self):
        state,issued,_,_=self.replay('missed',max_clicks=3)
        self.assertEqual(len(issued),3)
        self.assertEqual(state['stop_reason'],'click_limit')

    def test_changed_cross_in_fresh_preinput_frame_suppresses_retry(self):
        state,issued,_,_=self.replay('fresh-changed')
        self.assertIn('未识别到后续界面',state['last_error'])
        self.assertEqual(len(issued),3)

    def test_deadline_expiring_during_fresh_capture_suppresses_retry(self):
        state,issued,_,_=self.replay('fresh-deadline')
        self.assertIn('未识别到后续界面',state['last_error'])
        self.assertEqual(len(issued),3)

    def test_empty_orphan_closes_only_and_does_not_mark_old_claim_complete(self):
        state,issued,receipts,complete=self.replay('orphan')
        self.assertIsNone(state['last_error'])
        self.assertEqual([x[0] for x in issued],[CLOSES[0]])
        self.assertFalse(complete)
        self.assertEqual(len(receipts),1)


if __name__=='__main__':unittest.main()
