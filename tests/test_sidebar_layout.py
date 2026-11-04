"""Replay the missing event icon without moving fixed footer guards."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import Clicker
from drift.ui.sidebar import SUPPORTED, SidebarLayoutButton


class SidebarLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules={r['id']:r for r in cls.config['rules']}
        cls.ctx=SimpleNamespace(root=ROOT,config=cls.config)
        cls.real=cls.image('tests/fixtures/sidebar-layout/event-absent.png')

    @staticmethod
    def image(path):
        with Image.open(ROOT/path) as source:return np.asarray(source.convert('RGB'))

    @staticmethod
    def shifted(frame):
        result=frame.copy()
        result[230:900,500:720]=0
        result[140:810,500:720]=frame[230:900,500:720]
        return result

    def load(self,key):
        return Clicker.load_rule(self.ctx,self.rules[key])

    def matches(self,key,frame):
        rule,b,offset=self.load(key)
        return bool(b.match(frame,offset=offset,similarity=rule['threshold'])),b

    def test_real_new_positions_and_fixed_footer_returns(self):
        positions={'guild_open':(648,280,707,335),
                   'guild_return':(658,288,693,320),  # read-only fixed cart core
                   'mailbox_open':(659,389,701,415),
                   'achievements_open':(650,460,690,497),
                   'mailbox_return':(211,1140,266,1189),
                   'mailbox_recover':(211,1140,266,1189),
                   'achievements_recover':(211,1140,266,1189)}
        for suffix,expected in positions.items():
            key='driftinggirl_'+suffix
            with self.subTest(key=key):
                with patch('drift.runtime.adapt_sidebar',lambda _,b:b):
                    self.assertFalse(self.matches(key,self.real)[0])
                passed,b=self.matches(key,self.real)
                self.assertTrue(passed)
                self.assertEqual(tuple(b.button),expected)
                self.assertGreaterEqual(b.last_similarity,self.rules[key]['threshold'])

    def test_missing_event_never_becomes_shop_cart_or_daily_reward(self):
        self.assertFalse(self.matches('driftinggirl_event_potion_open',self.real)[0])
        self.assertTrue(self.matches('driftinggirl_daily_open',self.real)[0])

    def test_old_layout_still_has_original_inputs_and_guards(self):
        for key in SUPPORTED:
            frame=self.image(self.rules[key]['asset'])
            with self.subTest(key=key):
                with patch('drift.runtime.adapt_sidebar',lambda _,b:b):
                    old_passed,old=self.matches(key,frame)
                passed,b=self.matches(key,frame)
                self.assertEqual(passed,old_passed)
                if passed:self.assertEqual(b.button,old.button)

    def test_crown_menu_moves_but_footer_stays_at_original_position(self):
        for suffix in ('resume','crown','fold','claim_menu_recover'):
            key='driftinggirl_achievements_'+suffix
            frame=self.shifted(self.image(self.rules[key]['asset']))
            with self.subTest(key=key):
                passed,b=self.matches(key,frame)
                self.assertTrue(passed)
                if suffix=='fold':self.assertEqual(tuple(b.button),(650,460,690,497))
                else:self.assertEqual(tuple(b.button),(555,535,592,556))
                if self.rules[key].get('required_source_areas'):
                    frame[1140:1189,211:266]=0
                    self.assertFalse(self.matches(key,frame)[0])

    def test_missing_n_badges_or_one_misaligned_companion_is_rejected(self):
        for key,box in [('driftinggirl_mailbox_open',(691,415,715,435)),
                        ('driftinggirl_achievements_open',(693,504,714,523))]:
            frame=self.real.copy();x1,y1,x2,y2=box;frame[y1:y2,x1:x2]=0
            self.assertFalse(self.matches(key,frame)[0])
        frame=self.real.copy();frame[389:415,659:701]=0
        for key in ('driftinggirl_mailbox_return','driftinggirl_mailbox_recover'):
            self.assertFalse(self.matches(key,frame)[0])

    def test_brightness_and_footer_controls_remain_required(self):
        for key in ('driftinggirl_guild_open','driftinggirl_mailbox_open',
                    'driftinggirl_achievements_open','driftinggirl_mailbox_return'):
            self.assertFalse(self.matches(key,(self.real*.45).astype(np.uint8))[0])
        for box in ((211,1140,266,1189),(542,1140,597,1183)):
            frame=self.real.copy();x1,y1,x2,y2=box;frame[y1:y2,x1:x2]=0
            self.assertFalse(self.matches('driftinggirl_mailbox_return',frame)[0])

    def test_unknown_frames_and_modals_do_not_match_shifted_entries(self):
        for frame in (np.zeros_like(self.real),self.image('assets/guild-shop/potion-confirm.png'),
                      self.image('assets/mailbox-native.png'),self.image('assets/daily-reward/premium-ready.png')):
            for key in ('driftinggirl_guild_open','driftinggirl_mailbox_open','driftinggirl_achievements_open'):
                self.assertFalse(self.matches(key,frame)[0])

    def test_ambiguous_duplicate_icons_refuse_input_and_do_not_count_absent(self):
        frame=self.real.copy()
        frame[479:525,659:715]=self.real[389:435,659:715]
        passed,b=self.matches('driftinggirl_mailbox_open',frame)
        self.assertFalse(passed)
        self.assertGreater(b.last_similarity,self.rules['driftinggirl_mailbox_open']['threshold'])

    def test_no_old_target_survives_a_new_unknown_frame(self):
        rule,b,off=self.load('driftinggirl_mailbox_open')
        self.assertTrue(b.match(self.real,off,rule['threshold']))
        self.assertFalse(b.match(np.zeros_like(self.real),off,rule['threshold']))
        self.assertIsNone(b.target_key)
        self.assertIsNone(b.active)

    def test_overlap_does_not_churn_stable_target_or_mutate_configuration(self):
        before=deepcopy(self.config)
        rule,b,off=self.load('driftinggirl_daily_open')
        keys=[]
        for _ in range(3):
            self.assertTrue(b.match(self.real,off,rule['threshold']))
            keys.append(b.target_key)
        self.assertEqual(len(set(keys)),1)
        self.assertEqual(self.config,before)

    def test_custom_asset_crop_or_scale_is_not_reinterpreted(self):
        for change in ({'asset':'assets/navigation/main.png'},
                       {'source_crop':[658,479,701,505]}, {'scales':[.95,1,1.05]}):
            rule={**self.rules['driftinggirl_mailbox_open'],**change}
            _,b,_=Clicker.load_rule(self.ctx,rule)
            self.assertNotIsInstance(b,SidebarLayoutButton)

    def test_input_uses_fresh_position_and_rejects_layout_change_between_votes(self):
        config=deepcopy(self.config)
        entry=deepcopy(self.rules['driftinggirl_mailbox_open'])
        entry.update(enabled=True,interval=.2,stable_frames=2,stop_after=True)
        entry.pop('next_stage',None)
        config.update(rules=[entry],poll_interval=.1)
        original=self.image('assets/main-ui-native.png');shifted=self.real
        clock=[0]
        class Device:
            def __init__(self,_config,stop):self.image=None;self.frames=0;self.taps=[]
            def screenshot(self):
                self.frames+=1;clock[0]+=.3
                self.image=(original if self.frames<=2 else shifted).copy()
                return self.image
            def click(self,button):self.taps.append(tuple(button.button))
            def stuck_record_add(self,_button):pass
        with tempfile.TemporaryDirectory(prefix='sidebar-replay-') as folder:
            root=Path(folder);asset=root/entry['asset'];asset.parent.mkdir(parents=True)
            shutil.copyfile(ROOT/entry['asset'],asset)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,Device)
            try:
                with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=False,max_clicks=1,timeout=10);app.thread.join(8)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.device.taps,[(659,389,701,415)])
                self.assertTrue(any('取消本次点击' in event['message'] for event in app.events))
            finally:
                app.stop()
                if app.thread:app.thread.join(1)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)


if __name__=='__main__':unittest.main()
