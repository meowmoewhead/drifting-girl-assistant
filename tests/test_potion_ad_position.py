"""Replay real zero-count potion ads and both positions of the weekly RAID layout."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import Clicker
from digit_ocr import DigitReader
from tools.configure_potion_ad_position import RULE_ID, update


class PotionAdPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rule = next(r for r in cls.config['rules'] if r['id']==RULE_ID)
        cls.reader = DigitReader(ROOT)

    def image(self,path):
        with Image.open(ROOT/path) as source:
            return np.asarray(source.convert('RGB'))

    def evaluate(self,frame,rule=None):
        rule = rule or self.rule
        context = SimpleNamespace(root=ROOT,config=self.config,digit_reader=self.reader,
            lock=threading.RLock(),state={'frames':1,'numbers':{}})
        _,button,offset = Clicker.load_rule(context,rule)
        if not button.match(frame,offset=offset,similarity=rule['threshold']):
            return False,None,None,button
        passed,value = Clicker.counter_condition(context,rule,button,frame)
        return passed,value,context.state['numbers'][RULE_ID],button

    def test_real_zero_counter_no_longer_skipped_for_low_confidence(self):
        for name in ('potion-up-paused.png','potion-up-0.png','potion-up-1.png','potion-up-2.png'):
            with self.subTest(name=name):
                frame=self.image('tests/fixtures/ad-position/'+name)
                old=deepcopy(self.rule)
                old['counter_condition']['source_area']=[650,900,718,930]
                old['counter_condition']['preprocess']='white_text'
                passed,value,info,_=self.evaluate(frame,old)
                self.assertFalse(passed)
                self.assertEqual(value,(0,40))
                self.assertIn('置信度',info['skip_reason'])
                passed,value,info,_=self.evaluate(frame)
                self.assertTrue(passed)
                self.assertEqual(value,(0,40))
                self.assertEqual(info['box'],[650,814,718,834])
                self.assertGreaterEqual(info['confidence'],.93)
                self.assertEqual(info['remaining'],40)

    def test_original_layout_and_previously_repaired_counters_still_work(self):
        original=self.image('assets/potion-ad-source.png')
        for name,value in [('counter-1.png',1)]+[(f'counter-2-{i}.png',2) for i in range(3)]:
            with self.subTest(name=name):
                frame=original.copy()
                frame[900:930,650:718]=self.image('tests/fixtures/potion-counter/'+name)
                passed,reading,info,_=self.evaluate(frame)
                self.assertTrue(passed)
                self.assertEqual(reading,(value,40))
                self.assertEqual(info['box'],[650,904,718,924])
                self.assertGreaterEqual(info['confidence'],.93)

    def test_counter_and_click_follow_the_icon_up_and_down(self):
        original=self.image('assets/potion-ad-source.png')
        for delta in (-180,-90,90):
            frame=original.copy()
            frame[650:1115,600:720]=0
            frame[898+delta:966+delta,640:720]=original[898:966,640:720]
            passed,value,info,button=self.evaluate(frame)
            self.assertTrue(passed)
            self.assertEqual(value,(1,40))
            self.assertEqual(info['box'],[650,904+delta,718,924+delta])
            x1,y1,x2,y2=button.button
            self.assertEqual(((x1+x2)//2,(y1+y2)//2),(678,944+delta))

    def test_growth_and_modal_backgrounds_are_not_potion_entries(self):
        for name in ('assets/ad-entry-source.png','assets/potion-ad-confirm-source.png',
                     'assets/raid/battle.png','assets/raid/result.png',
                     'assets/achievements-list-native.png','tests/fixtures/legacy-captures/potion-ad-after-continue.png'):
            with self.subTest(name=name):
                self.assertFalse(self.evaluate(self.image(name))[0])
        dimmed=(self.image('tests/fixtures/ad-position/potion-up-paused.png')*.6).astype(np.uint8)
        self.assertIsNone(self.evaluate(dimmed)[2])

    def test_missing_digits_are_not_treated_as_zero(self):
        frame=self.image('tests/fixtures/ad-position/potion-up-paused.png').copy()
        crop=frame[814:834,650:718]
        crop[np.all(crop>=200,axis=2)]=0
        passed,value,_,_=self.evaluate(frame)
        self.assertFalse(passed)
        self.assertTrue(value is None or value[0] is None)

    def test_migration_preserves_limits_switches_and_other_rules(self):
        before=deepcopy(self.config)
        entry=next(r for r in before['rules'] if r['id']==RULE_ID)
        entry.update(enabled=False,interval=321,search_area=[600,760,720,1060])
        entry['counter_condition']['source_area']=[650,900,718,930]
        after=update(before)
        self.assertEqual(update(after),after)
        self.assertEqual({k:v for k,v in before.items() if k!='rules'},
                         {k:v for k,v in after.items() if k!='rules'})
        for old,new in zip(before['rules'],after['rules']):
            if old['id']==RULE_ID:
                excluded=('search_area','color_condition','note','counter_condition')
                self.assertEqual({k:v for k,v in old.items() if k not in excluded},
                                 {k:v for k,v in new.items() if k not in excluded})
                self.assertEqual({k:v for k,v in old['counter_condition'].items() if k!='source_area'},
                                 {k:v for k,v in new['counter_condition'].items() if k!='source_area'})
            else:
                self.assertEqual(old,new)


if __name__=='__main__':
    unittest.main()
