from pathlib import Path
from types import SimpleNamespace as NS
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

import numpy as np
from PIL import Image
from digit_ocr import DigitReader
from drift.perception.menu_rank import outlined_menu_rank
from drift.tasks.stone_farm import StoneFarm
from drift.tasks.cruise_settings import SettingsView
from drift.tasks.cruise_options import normalize
from tests.test_cruise_preferences import Harness

ROOT=Path(__file__).resolve().parents[1]


class MenuRankFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)

    def frame(self,rank=7):
        name='cruise-menu-rank7/menu.png' if rank==7 else 'cruise-menu-rank5/menu-0.png'
        return np.array(Image.open(ROOT/'tests/fixtures'/name).convert('RGB'))

    def owner(self):
        return NS(reader=self.reader,sailing=NS(menu=lambda f:True),_parallel=lambda f:True,rank=39)

    def test_real_seven_keeps_outline_in_four_distinct_high_confidence_views(self):
        owner=self.owner()
        self.assertEqual(outlined_menu_rank(owner,self.frame()),7)
        self.assertEqual(owner.menu_rank_probe['profile'],'compact')
        self.assertEqual(len(owner.menu_rank_probe['views']),4)
        self.assertTrue(all(v['confidence']>=.95 and v['value']==7 for v in owner.menu_rank_probe['views']))

    def test_real_five_retains_its_wide_profile(self):
        owner=self.owner()
        self.assertEqual(outlined_menu_rank(owner,self.frame(5)),5)
        self.assertEqual(owner.menu_rank_probe['profile'],'wide')

    def test_actual_settings_context_and_pre_input_guard_use_current_seven(self):
        task=StoneFarm(ROOT,self.reader,lambda _:False,lambda _:None,
                       state_path=ROOT/'tests/nonexistent-menu-rank7.json')
        task.enabled=lambda:True
        task.rank=39
        view=SettingsView(task,ROOT,self.reader);frame=self.frame()
        context=view.observe_setup(frame)
        self.assertEqual(context['scene'],(10,7))
        options=normalize()['cruise'];options.update(auto_sale=True,auto_upgrade=True)
        self.assertTrue(view.validate_action(frame,context,'start',options))
        changed=frame.copy();changed[9:59,71:136]=0
        self.assertFalse(view.validate_action(changed,context,'start',options))

    def test_unknown_dark_world_and_missing_digits_are_rejected(self):
        owner=self.owner();frame=self.frame();hidden=frame.copy();hidden[9:59,71:136]=0
        for bad in (frame//2,hidden,np.zeros_like(frame),frame[:100]):
            with self.subTest(shape=bad.shape,total=int(bad.sum())):
                self.assertIsNone(outlined_menu_rank(owner,bad))
        owner._parallel=lambda _:False
        self.assertIsNone(outlined_menu_rank(owner,frame))

    def test_old_confident_conflict_and_out_of_range_are_never_overridden(self):
        for value in (6,99):
            with self.subTest(value=value):
                self.assertIsNone(outlined_menu_rank(self.owner(),self.frame(),[NS(value=value,confidence=.99)]))

    def test_low_compact_view_and_conflicting_other_profile_are_rejected(self):
        unknown=[NS(value=None,confidence=.2)]*6
        for value,confidence in ((7,.94),(6,.99)):
            readings=[NS(value=7,confidence=.99) for _ in range(8)]
            readings[7]=NS(value=value,confidence=confidence)
            if value==6:readings[0]=NS(value=6,confidence=.99)
            with self.subTest(value=value),patch('drift.perception.menu_rank.read_numeric_views',side_effect=[unknown,readings]):
                self.assertIsNone(outlined_menu_rank(self.owner(),self.frame()))

    def test_scene_recheck_is_a_new_periodic_run_and_keeps_expired_history(self):
        with TemporaryDirectory() as root:
            h=Harness(root,False);h.menu=True;h.scene=(10,7)
            old=dict(success=False,deadline=90,attempts={'stop':1,'open':1},reason='old-window-expired')
            h.controller.journal.update(pending=False,last_result=old)
            history=Path(root)/'cruise_settings_state.history.jsonl'
            history.write_text(json.dumps(old)+'\n',encoding='utf-8')
            h.controller._save();h.tick(100)
            self.assertTrue(h.route)
            self.assertEqual(json.loads(history.read_text('utf-8').splitlines()[0]),old)
            self.assertEqual([kind for kind,_ in h.receipts],['pearl','material','amber','gem','boss','auto_sale','auto_upgrade','tickets','tickets','start'])
            self.assertNotIn('fish-start',[kind for kind,_ in h.receipts])


if __name__=='__main__':unittest.main()
