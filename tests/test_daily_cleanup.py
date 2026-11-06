"""A skipped reward must really close its verified shop, without a new purchase."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from app import Clicker

ROOT=Path(__file__).resolve().parents[1]

def image(path):return np.asarray(Image.open(ROOT/path).convert('RGB'))

class DailyCleanupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The 20261012 fixture actually has a READY right card and must not
        # authorise orphan cleanup. Use the real gray card for cleanup tests.
        cls.claimed=image('assets/daily-reward/premium-claimed.png')
        config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules=[deepcopy(r) for r in config['rules'] if r['id'].startswith('driftinggirl_daily_')]
        for r in cls.rules:r['enabled']=True
        cls.context=object.__new__(Clicker)
        cls.context.root=ROOT;cls.context.config=config
        cls.matchers={r['id']:cls.context.load_rule(r) for r in cls.rules}

    def test_real_claimed_shop_is_identified_with_fixed_rank_and_world_context(self):
        first=self.context.daily_cleanup_context(self.claimed,self.matchers)
        self.assertIsNotNone(first)
        self.assertEqual(first[2],(634,247,695,296))
        moved=self.claimed.copy();moved[10:62,72:126]=0
        self.assertNotEqual(self.context.daily_cleanup_context(moved,self.matchers),first)

    def test_free_claimed_but_bonus_ready_cannot_be_closed_as_an_orphan(self):
        ready=image('tests/fixtures/daily-cleanup/claimed-20261012.png')
        self.assertIsNone(self.context.daily_cleanup_context(ready,self.matchers))
        self.assertIsNotNone(self.context.daily_cleanup_context(ready,self.matchers,allow_ready=True))

    def test_actual_post_claim_shadow_variant_is_a_guarded_close(self):
        claimed=image('assets/daily-reward/bonus-claimed.png')
        self.assertIsNotNone(self.context.daily_cleanup_context(claimed,self.matchers))
        self.assertFalse(self.context.daily_cleanup_context(claimed//2,self.matchers))

    def test_paid_products_unclaimed_dark_or_unknown_screens_cannot_authorize_close(self):
        frames=[self.claimed//2,np.zeros_like(self.claimed),
                image('assets/daily-reward/premium-ready.png'),
                image('assets/daily-reward/shop-open.png'),image('assets/daily-reward/main.png')]
        missing_check=self.claimed.copy();missing_check[434:480,255:330]=0
        frames.append(missing_check)
        for f in frames:
            self.assertIsNone(self.context.daily_cleanup_context(f,self.matchers))
        self.assertIsNone(self.context.daily_cleanup_context(self.claimed[:900],self.matchers))
        disabled=dict(self.matchers)
        r,b,o=disabled['driftinggirl_daily_close'];disabled[r['id']]=({**r,'enabled':False},b,o)
        self.assertIsNone(self.context.daily_cleanup_context(self.claimed,disabled))

    def test_shipped_bonus_guard_uses_claim_independent_page_marks_without_saving_changes(self):
        legacy=deepcopy(next(r for r in self.rules if r['id']=='driftinggirl_daily_bonus'))
        legacy.update(asset='assets/daily-reward/premium-ready.png',
                      required_source_areas=[[177,336,310,366],[266,442,316,483]])
        before=deepcopy(legacy)
        rule,button,offset=self.context.load_rule(legacy)
        self.assertEqual(legacy,before)
        self.assertTrue(button.match(self.claimed,offset=offset,similarity=rule['threshold']))
        self.assertEqual(tuple(button.button),(410,430,605,500))
        self.assertEqual(rule['threshold'],before['threshold'])
        self.assertEqual(rule['required_source_areas'],[[171,250,263,280],[637,251,688,294]])

    def replay(self,*,timeout_path=False,no_return=False,stop_on_recheck=False):
        claimed=self.claimed
        frames=([image('assets/daily-reward/main.png'),image('assets/daily-reward/shop-open.png'),
                 image('assets/daily-reward/premium-ready.png'),claimed,
                 image('assets/daily-reward/return.png')] if timeout_path
                else [claimed,image('assets/daily-reward/return.png')])
        expected=([(648,367,709,423),(168,228,270,297),(210,433,377,510),(634,247,695,296)]
                  if timeout_path else [(634,247,695,296)])
        clock=[0.0];devices=[]
        class Device:
            def __init__(self,config,stop):
                self.stop=stop;self.taps=[];self.image=None;self.reads=0;devices.append(self)
            def screenshot(self):
                clock[0]+=.6;self.reads+=1
                index=min(len(self.taps),len(frames)-1)
                self.image=(claimed if no_return and self.taps else frames[index]).copy()
                if stop_on_recheck and self.reads==7:self.stop.set()
                return self.image
            def click(self,button):
                if self.stop.is_set():raise AssertionError('Input after stop')
                box=button.button;x=(box[0]+box[2])//2;y=(box[1]+box[3])//2
                if len(self.taps)>=len(expected):raise AssertionError('Extra input')
                a,b,c,d=expected[len(self.taps)]
                if not a<=x<=c or not b<=y<=d:raise AssertionError('Wrong target '+str((x,y)))
                self.taps.append((x,y))
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'assets/daily-reward',root/'assets/daily-reward')
            rules=deepcopy(self.rules)
            for r in rules:r.update(enabled=True,interval=.2,next_delay=0,stage_timeout=5)
            if timeout_path:
                # The live saved configuration still requires the unclaimed
                # gem fragment on this bonus step; reproduce that exact miss.
                bonus=next(r for r in rules if r['id']=='driftinggirl_daily_bonus')
                bonus['required_source_areas']=[[177,336,310,366],[267,442,316,483]]
            config=dict(adb_path='fake',serial='test',resolution=[720,1280],poll_interval=.1,
                        max_consecutive_clicks=2,rules=rules)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,stop_on_main=True,max_clicks=6,timeout=80)
                    app.thread.join(15)
                self.assertFalse(app.running())
                self.assertFalse(app.daily_schedule.is_complete('daily_reward'))
                if no_return:
                    self.assertIsNotNone(app.state['last_error'])
                    self.assertNotEqual(app.state['stage'],'main')
                    self.assertEqual(len(devices[0].taps),1)
                elif stop_on_recheck:
                    self.assertEqual(devices[0].taps,[])
                else:
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(app.state['stop_reason'],'cycle_complete')
                    self.assertEqual(len(devices[0].taps),len(expected))
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for h in list(app.log.handlers):h.close();app.log.removeHandler(h)
            return app

    def test_restart_on_orphan_claimed_shop_closes_once_without_new_claim(self):self.replay()
    def test_bonus_timeout_closes_real_shop_without_marking_daily_complete(self):self.replay(timeout_path=True)
    def test_close_without_verified_return_stops_instead_of_claiming_main(self):self.replay(no_return=True)
    def test_stop_during_fresh_close_recheck_prevents_input(self):self.replay(stop_on_recheck=True)

    def test_unverified_cleanup_consumes_the_total_window_then_becomes_retryable(self):
        # 认不出画面时:先吃满 40 秒总窗口(至少延长一次),再按"可重试"收尾,
        # 而不是在 20 秒阶段线就写错误截图停机。
        app = self.replay(no_return=True)
        messages = [event['message'] for event in app.events]
        self.assertTrue(any('延长到总截止再试' in message for message in messages), messages[-6:])
        self.assertEqual(app.state['stop_reason'], 'retryable')
        self.assertIn('每日奖励清算', app.state['last_error'])
        self.assertNotIn('未识别到后续界面', app.state['last_error'])

if __name__=='__main__':unittest.main()
