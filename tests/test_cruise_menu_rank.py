"""RANK5 on a dimmed native settings header must retain its digit outline."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from digit_ocr import DigitReader
from drift.tasks.stone_farm import StoneFarm
from drift.tasks.cruise_settings import SettingsView
from tests.test_cruise_preferences import Harness

ROOT=Path(__file__).resolve().parents[1]

class ActualMenuRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)
    def setUp(self):
        self.w=StoneFarm(ROOT,self.reader,lambda _:False,lambda _:None,
            state_path=ROOT/'tests/nonexistent-menu-rank.json')
        self.w.enabled=lambda:True
        self.view=SettingsView(self.w,ROOT,self.reader)
    def frame(self,n=0):
        return np.array(Image.open(ROOT/f'tests/fixtures/cruise-menu-rank5/menu-{n}.png').convert('RGB'))
    def test_three_actual_stuck_menus_read_current_stage_and_rank(self):
        for n in range(3):
            f=self.frame(n)
            with self.subTest(n=n):
                self.assertEqual(self.w._sailing_menu_rank(f),5,self.w.menu_rank_probe)
                self.assertEqual(self.w.menu_rank_probe['method'],'outlined_digit')
                self.assertTrue(all(r['value']==5 and r['confidence']>=.95
                                    for r in self.w.menu_rank_probe['views']))
                self.assertEqual(self.view.observe_setup(f)['scene'],(10,5))
    def test_unreadable_dark_and_missing_strokes_are_not_guessed(self):
        f=self.frame();hidden=f.copy();hidden[9:59,71:136]=0
        for a in (f//2,hidden,np.zeros_like(f)):
            with self.subTest(total=int(a.sum())):self.assertIsNone(self.w._sailing_menu_rank(a))
    def test_any_strong_conflict_or_low_fallback_view_rejects(self):
        f=self.frame();good=[SimpleNamespace(value=5,confidence=.99) for _ in range(4)]
        cases=[([SimpleNamespace(value=6,confidence=.99)]+[SimpleNamespace(value=None,confidence=.2)]*3,good),
               ([SimpleNamespace(value=None,confidence=.2)]*4,good[:3]+[SimpleNamespace(value=5,confidence=.94)]),
               ([SimpleNamespace(value=None,confidence=.2)]*4,good[:3]+[SimpleNamespace(value=6,confidence=.99)])]
        for legacy,new in cases:
            with self.subTest(legacy=legacy,new=new), \
                 patch('drift.tasks.stone_farm.read_numeric_views',side_effect=[legacy,legacy]), \
                 patch('drift.tasks.parallel_screen.read_numeric_views',return_value=new):
                self.assertIsNone(self.w._sailing_menu_rank(f))
    def test_complete_known_settings_get_fresh_start_guard(self):
        from drift.tasks.cruise_options import normalize
        f=self.frame();context=self.view.observe_setup(f)
        options=normalize()['cruise'];options['auto_sale']=True;options['auto_upgrade']=True
        self.assertTrue(self.view.validate_action(f,context,'start',options))
        changed=f.copy();changed[45:71,40:68]=0
        self.assertFalse(self.view.validate_action(changed,context,'start',options))

class CompletedWindowTests(unittest.TestCase):
    def test_periodic_recheck_of_existing_menu_preserves_old_failure(self):
        with TemporaryDirectory() as root:
            h=Harness(root,False);h.menu=True
            h.controller.journal.update(pending=False,last_result=dict(success=False,deadline=90,
                attempts={'stop':1,'open':1},reason='old-window-expired'))
            old=json.loads(json.dumps(h.controller.journal['last_result']))
            history=Path(root)/'cruise_settings_state.history.jsonl'
            history.write_text(json.dumps(old)+'\n',encoding='utf-8')
            h.controller._save();h.tick(100)
            self.assertTrue(h.route)
            self.assertNotIn('stop',[kind for kind,_ in h.receipts])
            self.assertNotIn('open',[kind for kind,_ in h.receipts])
            self.assertNotIn('fish-start',[kind for kind,_ in h.receipts])
            self.assertEqual(json.loads(history.read_text(encoding='utf-8').splitlines()[0]),old)
            self.assertTrue(h.controller.journal['last_result']['success'])

if __name__=='__main__':unittest.main()
