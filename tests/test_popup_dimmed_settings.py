"""Real notice over cruise settings: strict closure and read-only return."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from drift.runtime import Clicker
from drift.core.session import LoopState
from drift.ui.popup_raid import PopupRaidReturn
from drift.ui.popup_cruise import PopupCruiseReturn
from drift.tasks.stone_farm import StoneFarm
from drift.tasks.cruise_settings import SettingsView
from digit_ocr import DigitReader
from digit_ocr import NumberReading

ROOT=Path(__file__).resolve().parents[1]
def image(name):
    return cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/popup-cruise'/name)),cv2.COLOR_BGR2RGB)


class DimmedNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rule=next(r for r in cls.config['rules'] if r['id']==PopupRaidReturn.CLOSE)

    def matches(self,frame):
        r,b,o=Clicker.load_rule(SimpleNamespace(root=ROOT,config=self.config),self.rule)
        return b.match(frame,offset=o,similarity=r['threshold'])

    def test_real_double_modal_notice_has_stable_identity_without_loosening_raid(self):
        a=image('notice.png');b=image('notice-heldout.png')
        self.assertTrue(self.matches(a));self.assertTrue(self.matches(b))
        self.assertIsNone(PopupRaidReturn.identity(a))
        key=PopupRaidReturn.notice_identity(a)
        self.assertIsNotNone(key)
        self.assertEqual(key,PopupRaidReturn.notice_identity(b))
        self.assertEqual(key,PopupRaidReturn.notice_identity(image('menu.png')))
        self.assertEqual(key,PopupRaidReturn.notice_identity(image('menu-heldout.png')))

    def test_dark_invalid_and_missing_rank_cannot_identify(self):
        frame=image('notice.png')
        for bad in (frame//2,frame[:900],np.zeros_like(frame)):
            self.assertIsNone(PopupRaidReturn.notice_identity(bad))
        bad=frame.copy();bad[10:62,72:126]=0
        self.assertIsNone(PopupRaidReturn.notice_identity(bad))

    def test_world_and_rank_changes_remain_bound_on_fresh_input(self):
        frame=image('notice.png');key=PopupRaidReturn.notice_identity(frame)
        world=frame.copy();world[45:72,13:68]=0
        self.assertNotEqual(PopupRaidReturn.notice_identity(world),key)
        rank=frame.copy();rank[10:62,72:98]=0
        self.assertNotEqual(PopupRaidReturn.notice_identity(rank),key)

    def test_known_notice_shape_is_still_required_in_addition_to_header(self):
        frame=image('notice.png')
        for x,y,X,Y in ([618,406,674,453],[315,465,350,492],[106,785,162,810]):
            bad=frame.copy();bad[y:Y,x:X]=0
            self.assertFalse(self.matches(bad))
        self.assertFalse(self.matches(image('menu.png')))


class CruisePopupReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.owner=StoneFarm(ROOT,self.reader,lambda _:False,lambda _:None,
                             state_path=Path(self.temp.name)/'empty.json')
        self.view=SettingsView(self.owner,ROOT,self.reader)
        self.setup=SimpleNamespace(view=self.view,deadline=75.,journal={'attempts':{'open':1}})
        self.st=LoopState(stage='main',stage_deadline=50.)
        self.guard=PopupCruiseReturn()
        self.notice=image('notice.png');self.menu=image('menu.png')

    def arm(self):
        self.assertTrue(self.guard.arm(self.st,self.notice,self.setup))
        self.st.handling_interrupt=True;self.st.stage='task_popup_return'
        self.st.stage_ready_at=100.;self.st.stage_deadline=120.

    def observe(self,at,frame=None,**kw):
        return self.guard.observe(self.st,(self.menu if frame is None else frame).copy(),self.setup,
                                  notice_gone=True,now=at,**kw)

    def test_actual_rank15_menu_uses_four_complete_high_confidence_views(self):
        self.assertEqual(self.owner._sailing_menu_rank(self.menu),15)
        self.assertEqual(self.view.observe_setup(self.menu)['scene'],(9,15))
        self.assertIsNone(self.view.observe_setup(self.notice))

    def test_complete_rank_still_rejects_low_confidence_conflict_and_strong_old_conflict(self):
        none=NumberReading('',None,0.)
        good=NumberReading('15',15,.999)
        cases=[([none]*4,[NumberReading('15',15,.949)]*4),
               ([none]*4,[good,good,good,NumberReading('16',16,.999)]),
               ([NumberReading('14',14,.999),none,none,none],[good]*4)]
        for old,new in cases:
            with self.subTest(old=old,new=new),patch('drift.tasks.stone_farm.read_numeric_views',side_effect=[old,new]):
                self.assertIsNone(self.owner._sailing_menu_rank(self.menu))

    def test_return_needs_three_new_frames_three_seconds_and_changes_no_budget(self):
        self.arm();old=deepcopy(self.setup.journal);deadline=self.setup.deadline
        self.assertFalse(self.observe(101));self.assertFalse(self.observe(102));self.assertTrue(self.observe(104))
        self.assertEqual(self.setup.journal,old);self.assertEqual(self.setup.deadline,deadline)
        self.assertEqual(self.st.stage_deadline,120.)

    def test_same_frame_and_fast_samples_never_complete(self):
        self.arm()
        frame=self.menu.copy()
        for at in (101,102,104):
            self.assertFalse(self.guard.observe(self.st,frame,self.setup,notice_gone=True,now=at))
        self.assertEqual(self.guard.frames,1)
        self.st.stage='main';self.st.handling_interrupt=False;self.arm()
        self.assertFalse(self.observe(104.1));self.assertFalse(self.observe(104.2));self.assertFalse(self.observe(104.3))

    def test_wrong_header_unknown_values_and_notice_reset_votes(self):
        self.arm();self.assertFalse(self.observe(101))
        for bad in (self.notice,self.menu//2,self.menu[:900]):
            self.assertFalse(self.observe(102,bad));self.assertEqual(self.guard.frames,0)
        world=self.menu.copy();world[45:72,13:68]=0
        self.assertFalse(self.observe(103,world))
        with patch.object(self.view,'values',return_value=None):
            self.assertFalse(self.observe(104))
        with patch.object(self.owner,'_off_icon',return_value=False):
            self.assertFalse(self.observe(105))

    def test_stop_or_popup_deadline_cannot_transfer_ownership(self):
        for kw in ({'stopped':True},{}):
            self.setUp();self.arm()
            self.assertFalse(self.observe(101));self.assertFalse(self.observe(102))
            self.assertFalse(self.observe(120 if not kw else 104,**kw))
            self.assertIsNone(self.guard.origin)

    def test_no_controller_or_wrong_stage_cannot_arm(self):
        self.assertFalse(self.guard.arm(self.st,self.notice,None))
        self.st.stage='raid_wait'
        self.assertFalse(self.guard.arm(self.st,self.notice,self.setup))
