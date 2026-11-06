"""Replay a missed guild product tap without repurchasing or renewing a window."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker
from digit_ocr import DigitReader
from drift.ui.guild_selection import GuildSelectionGuard


class GuildSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {r['id']: r for r in cls.config['rules'] if r['id'].startswith('driftinggirl_guild_')}
        for rule in cls.rules.values():rule['enabled']=True
        cls.reader = DigitReader(ROOT)
        cls.shop = cls.image('tests/fixtures/sidebar-layout/guild-selection-missed.png')
        cls.dialog = cls.image('assets/guild-shop/material-min.png')
        cls.main = cls.image('tests/fixtures/sidebar-layout/event-absent.png')

    @staticmethod
    def image(path):
        with Image.open(ROOT/path) as im:
            return np.asarray(im.convert('RGB'))

    def setUp(self):
        self.key = 'driftinggirl_guild_material_select'
        self.ctx = SimpleNamespace(root=ROOT, config=self.config, lock=threading.RLock(),
                                   digit_reader=self.reader, state={'frames': 1, 'numbers': {}})
        self.matchers = {key: Clicker.load_rule(self.ctx, rule) for key, rule in self.rules.items()}
        self.purchase = lambda r,b,f: Clicker.purchase_condition(self.ctx,r,b,f)
        self.guard = GuildSelectionGuard()
        self.origin = GuildSelectionGuard.context(self.shop,self.matchers,self.key,self.purchase)
        self.guard.arm(self.key,self.origin)

    def observe(self, now, frame=None, **changes):
        options = dict(stage='guild_material_quantity',trigger=self.rules[self.key]['name'],
                       started=0,deadline=30,now=now,stopped=False)
        options.update(changes)
        return self.guard.observe(self.shop if frame is None else frame,self.matchers,self.purchase,**options)

    def test_real_missed_window_requires_three_frames_and_three_seconds(self):
        self.assertIsNotNone(self.origin)
        self.assertFalse(self.observe(2))
        self.assertFalse(self.observe(4))
        self.assertTrue(self.observe(5))

    def test_opened_quantity_dialog_permanently_disarms_cleanup(self):
        self.assertFalse(self.observe(2,self.dialog))
        self.assertIsNone(self.guard.rule_id)
        for now in (5,8,12):self.assertFalse(self.observe(now))

    def test_stop_deadline_changed_stage_and_trigger_never_close(self):
        for change in ({'stopped':True},{'deadline':5},{'stage':'guild_material_confirm'},
                       {'trigger':'other'},{'deadline':None}):
            with self.subTest(change=change):
                self.guard.arm(self.key,self.origin)
                self.assertFalse(self.observe(5,**change))
                self.assertIsNone(self.guard.rule_id)

    def test_changed_world_rank_stock_currency_or_obscured_frame_rejected(self):
        for box in ((72,10,126,62),(35,46,56,66),(373,502,441,530),
                    (560,350,641,377),(175,337,328,374)):
            frame=self.shop.copy();x1,y1,x2,y2=box;frame[y1:y2,x1:x2]=0
            with self.subTest(box=box):
                self.guard.arm(self.key,self.origin)
                for now in (2,4,6):self.assertFalse(self.observe(now,frame))
        dark=(self.shop.astype(float)*.45).astype(np.uint8)
        self.assertIsNone(GuildSelectionGuard.context(dark,self.matchers,self.key,self.purchase))

    def test_one_obscured_frame_restarts_consecutive_proof(self):
        self.assertFalse(self.observe(2))
        self.assertFalse(self.observe(4,np.zeros_like(self.shop)))
        self.assertFalse(self.observe(5))
        self.assertFalse(self.observe(7))
        self.assertTrue(self.observe(8))

    def test_complete_runtime_only_closes_missed_selection_and_preserves_daily(self):
        shop,main,special=self.shop,self.main,self.image('assets/guild-shop/special.png')
        class Device:
            def __init__(self,config,stop):self.stop,self.image,self.taps=stop,None,[]
            def screenshot(self):
                self.image=(main if self.taps and self.taps[-1].endswith('_close') else
                            shop if self.taps else special).copy()
                return self.image
            def click(self,button):
                if self.stop.is_set():raise InterruptedError()
                expected=['driftinggirl_guild_tab','driftinggirl_guild_material_select','driftinggirl_guild_close']
                if len(self.taps)>=3 or button.name!=expected[len(self.taps)]:
                    raise AssertionError('Unexpected input: '+button.name)
                self.taps.append(button.name)
            def stuck_record_add(self,button):pass
        with tempfile.TemporaryDirectory(prefix='guild-selection-') as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets/guild-shop',root/'assets/guild-shop')
            config=deepcopy(self.config);config['poll_interval']=.1
            keys=['driftinggirl_guild_tab',self.key,'driftinggirl_guild_material_plus','driftinggirl_guild_material_buy',
                  'driftinggirl_guild_material_confirm',
                  'driftinggirl_guild_close','driftinggirl_guild_return']
            config['rules']=[deepcopy(self.rules[key]) for key in keys]
            config['rules'][0]['stage']='main'
            config['rules'][0]['next_stage']='guild_material'
            for rule in config['rules']:rule.update(enabled=True,interval=.2,next_delay=0,stage_timeout=12)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device);app.digit_reader=self.reader
            try:
                app.start(preview=False,stop_on_main=True,max_clicks=4,timeout=16)
                app.thread.join(19)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'cycle_complete')
                self.assertEqual(app.state['clicks'],3)
                self.assertFalse(app.daily_schedule.is_complete('guild_shop'))
                evidence=list((root/'logs/guild-input').glob('input-*.json'))
                self.assertEqual(len(evidence),2)
                receipts=[json.loads(p.read_text(encoding='utf-8')) for p in evidence]
                receipt=next(r for r in receipts if r['rule_id']==self.key)
                self.assertTrue(receipt['issued_only'])
                self.assertEqual(receipt['hold_ms'],80)
                self.assertTrue(receipt['after_frame_id'])
                self.assertNotEqual(receipt['before_frame_id'],receipt['after_frame_id'])
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_new_real_fault_is_safe_list_but_modal_dark_unknown_and_disabled_are_not(self):
        fault=self.image('tests/fixtures/guild-selection/material-missed-20261016.png')
        self.assertIsNotNone(GuildSelectionGuard.list_context(fault,self.matchers))
        for frame in (self.dialog,self.image('assets/guild-shop/material-confirm.png'),
                      self.image('assets/guild-shop/potion-confirm.png'),fault//2,self.main):
            self.assertIsNone(GuildSelectionGuard.list_context(frame,self.matchers))
        changed=dict(self.matchers)
        rule,b,o=changed['driftinggirl_guild_close']
        changed['driftinggirl_guild_close']=({**rule,'enabled':False},b,o)
        self.assertIsNone(GuildSelectionGuard.list_context(fault,changed))

    def test_three_digit_balance_uses_fixed_guild_controls_without_changing_saved_rule(self):
        fault=self.image('tests/fixtures/guild-selection/close-balance-980-20261019.png')
        saved=deepcopy(self.rules['driftinggirl_guild_close'])
        r,b,o=Clicker.load_rule(self.ctx,self.rules['driftinggirl_guild_close'])
        self.assertTrue(b.match(fault,offset=o,similarity=r['threshold']))
        self.assertIsNotNone(GuildSelectionGuard.list_context(fault,self.matchers))
        self.assertEqual(self.rules['driftinggirl_guild_close'],saved)
        self.assertEqual(r['threshold'],.94)
        self.assertEqual(tuple(b.button),(633,244,700,304))
        custom={**saved,'id':'custom_guild_close'}
        old,button,offset=Clicker.load_rule(self.ctx,custom)
        self.assertEqual(old['required_source_areas'],saved['required_source_areas'])
        self.assertFalse(button.match(fault,offset=offset,similarity=old['threshold']))

    def test_balance_guard_repair_still_rejects_missing_tab_cross_modal_and_dim_list(self):
        fault=self.image('tests/fixtures/guild-selection/close-balance-980-20261019.png')
        for box in ((91,469,132,511),(637,251,688,294)):
            image=fault.copy();x,y,X,Y=box;image[y:Y,x:X]=255
            self.assertIsNone(GuildSelectionGuard.list_context(image,self.matchers))

    def test_three_digit_balance_still_reads_both_purchased_stock_limits(self):
        fault=self.image('tests/fixtures/guild-selection/close-balance-980-20261019.png')
        for item,total in (('potion',10),('material',5)):
            key='driftinggirl_guild_'+item+'_skip'
            r,b,o=self.matchers[key]
            self.assertTrue(b.match(fault,offset=o,similarity=r['threshold']),key)
            passed,_=self.purchase(r,b,fault)
            self.assertTrue(passed,key)
            reading=self.ctx.state['numbers'][key]
            self.assertTrue(reading['valid'])
            self.assertEqual(reading['value'],total)
            self.assertEqual(reading['remaining'],0)
        for image in (fault//2,self.main,self.dialog,
                      self.image('assets/guild-shop/material-confirm.png')):
            self.assertIsNone(GuildSelectionGuard.list_context(image,self.matchers))

    def test_seen_quantity_reason_survives_later_frames_for_error_diagnostics(self):
        self.observe(2,self.dialog)
        for now in (4,8,12):self.observe(now)
        self.assertEqual(self.guard.status()['reason'],'quantity_dialog_seen')

    def test_purchase_match_without_cleanup_context_cannot_send_selection(self):
        shop=self.shop.copy();shop[10:62,72:126]=0
        rule,button,offset=self.matchers[self.key]
        self.assertTrue(button.match(shop,offset=offset,similarity=rule['threshold']))
        self.assertTrue(self.purchase(rule,button,shop)[0])
        self.assertIsNone(GuildSelectionGuard.context(shop,self.matchers,self.key,self.purchase))
        clock=[0.0]
        class Device:
            def __init__(self,config,stop):self.image=None
            def screenshot(self):clock[0]+=1;self.image=shop.copy();return self.image
            def click(self,_):raise AssertionError('Selection without a cleanup context')
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory(prefix='guild-precheck-') as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets/guild-shop',root/'assets/guild-shop')
            config=deepcopy(self.config);config['poll_interval']=.1
            keys=[self.key,'driftinggirl_guild_material_plus','driftinggirl_guild_material_buy',
                  'driftinggirl_guild_material_confirm','driftinggirl_guild_close','driftinggirl_guild_return']
            config['rules']=[deepcopy(self.rules[k]) for k in keys]
            config['rules'][0]['stage']='main'
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device);app.digit_reader=self.reader
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,timeout=12);app.thread.join(10)
                self.assertFalse(app.running())
                self.assertEqual(app.state['clicks'],0)
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'timeout')
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def replay_orphan(self, *, change_before_click=False, stop_before_click=False, no_return=False):
        shop=self.image('tests/fixtures/guild-selection/material-missed-20261016.png')
        main=self.main
        clock=[0.0];devices=[]
        class Device:
            def __init__(self,config,stop):
                self.stop,self.image,self.reads,self.taps=stop,None,0,[];devices.append(self)
            def screenshot(self):
                clock[0]+=.8;self.reads+=1
                self.image=(main if self.taps and not no_return else shop).copy()
                # 5 frames confirm the list, then 3 close-rule frames and its
                # independent input recheck. Only the recheck changes below.
                if self.reads==9:
                    if change_before_click:self.image[10:62,72:126]=0
                    if stop_before_click:self.stop.set()
                return self.image
            def click(self,button):
                if self.stop.is_set():raise AssertionError('Input after stop')
                if button.name!='driftinggirl_guild_close' or self.taps:
                    raise AssertionError('Unexpected or repeated guild input')
                self.taps.append((button.name,clock[0]))
            def stuck_record_add(self,_):pass
        with tempfile.TemporaryDirectory(prefix='guild-orphan-') as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets/guild-shop',root/'assets/guild-shop')
            config=deepcopy(self.config);config['poll_interval']=.1
            config['rules']=[deepcopy(self.rules[k]) for k in
                             ('driftinggirl_guild_close','driftinggirl_guild_return')]
            for r in config['rules']:r.update(enabled=True,interval=.2,next_delay=0)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,stop_on_main=True,max_clicks=2,timeout=35)
                    app.thread.join(15)
                self.assertFalse(app.running())
                taps=devices[0].taps
                self.assertFalse(app.daily_schedule.is_complete('guild_shop'))
                if stop_before_click:
                    self.assertEqual(taps,[])
                elif no_return:
                    self.assertEqual(len(taps),1)
                    self.assertIsNotNone(app.state['last_error'])
                    self.assertLess(clock[0],26)
                else:
                    self.assertEqual(len(taps),1)
                    self.assertGreaterEqual(taps[0][1],3)
                    if change_before_click:self.assertGreater(taps[0][1],7.2)
                    self.assertIsNone(app.state['last_error'])
                    self.assertEqual(app.state['stop_reason'],'cycle_complete')
            finally:
                app.stop()
                if app.thread:app.thread.join(2)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)

    def test_restarted_real_guild_list_closes_once_without_purchases(self):self.replay_orphan()
    def test_orphan_close_rechecks_current_rank_before_input(self):self.replay_orphan(change_before_click=True)
    def test_stop_during_orphan_close_prevents_input(self):self.replay_orphan(stop_before_click=True)
    def test_orphan_close_keeps_twenty_second_bound_if_return_is_missing(self):self.replay_orphan(no_return=True)


if __name__=='__main__':unittest.main()
