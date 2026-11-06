"""The optional RAID icon must not strand a vertically shifted growth ad."""
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
from dev_tools.configure_ad_position import RULE_ID, update


class GrowthAdPositionTests(unittest.TestCase):
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
        found = button.match(frame,offset=offset,similarity=rule['threshold'])
        if not found:
            return False,None,None
        passed,value = Clicker.counter_condition(context,rule,button,frame)
        return passed,value,context.state['numbers'][RULE_ID]

    def test_real_shifted_entry_previously_failed_and_now_reads_zero_of_ten(self):
        for name in ('growth-up-before.png','growth-up-paused.png'):
            frame = self.image('tests/fixtures/ad-position/'+name)
            old = {**self.rule,'search_area':[600,850,720,1060]}
            self.assertEqual(self.evaluate(frame,old),(False,None,None))
            passed,value,info = self.evaluate(frame)
            self.assertTrue(passed)
            self.assertEqual(value,(0,10))
            self.assertEqual(info['box'],[654,814,718,835])
            self.assertGreaterEqual(info['confidence'],.93)

    def test_old_layout_and_relative_counter_still_work(self):
        passed,value,info = self.evaluate(self.image('assets/ad-entry-source.png'))
        self.assertTrue(passed)
        self.assertEqual(value,(1,10))
        self.assertEqual(info['box'],[654,904,718,925])

    def test_counter_and_click_translate_together_without_fixed_coordinates(self):
        original = self.image('assets/ad-entry-source.png')
        for delta in (-180,-90,90):
            frame = original.copy()
            frame[760:1115,600:720] = 0
            frame[890+delta:978+delta,630:720] = original[890:978,630:720]
            passed,value,info = self.evaluate(frame)
            self.assertTrue(passed)
            self.assertEqual(value,(1,10))
            self.assertEqual(info['box'],[654,904+delta,718,925+delta])
            context = SimpleNamespace(root=ROOT,config=self.config)
            _,button,offset = Clicker.load_rule(context,self.rule)
            self.assertTrue(button.match(frame,offset=offset,similarity=self.rule['threshold']))
            x1,y1,x2,y2 = button.button
            self.assertEqual(((x1+x2)//2,(y1+y2)//2),(649,933+delta))

    def test_wider_strip_does_not_match_potion_or_no_ad_screens(self):
        for name in ('assets/potion-ad-source.png','assets/raid/battle.png',
                     'assets/raid/result.png','assets/achievements-list-native.png',
                     'tests/fixtures/legacy-captures/potion-ad-after-continue.png'):
            with self.subTest(name=name):
                self.assertFalse(self.evaluate(self.image(name))[0])

    def test_missing_counter_is_not_treated_as_zero(self):
        frame=self.image('tests/fixtures/ad-position/growth-up-paused.png').copy()
        crop=frame[814:835,654:718]
        crop[np.all(crop>=200,axis=2)]=0
        passed,value,info=self.evaluate(frame)
        self.assertFalse(passed)
        self.assertTrue(value is None or value[0] is None)

    def test_migration_preserves_other_rules_and_limits(self):
        before=deepcopy(self.config)
        entry=next(r for r in before['rules'] if r['id']==RULE_ID)
        entry.update(enabled=False,interval=321,search_area=[600,850,720,1060])
        after=update(before)
        self.assertEqual(update(after),after)
        for old,new in zip(before['rules'],after['rules']):
            if old['id']==RULE_ID:
                self.assertEqual({k:v for k,v in old.items() if k not in ('search_area','color_condition','note')},
                                 {k:v for k,v in new.items() if k not in ('search_area','color_condition','note')})
            else:
                self.assertEqual(old,new)


if __name__=='__main__':
    unittest.main()
