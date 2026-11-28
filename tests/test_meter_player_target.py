"""Replay real hold frames and post-release fishing under the new policy."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from boss_training import BossTraining
from digit_ocr import DigitReader
from fishing_meter import FishingMeter
from drift.tasks.stone_farm import StoneFarm
from tests import test_stone_farm as stone_tests

ROOT=Path(__file__).resolve().parents[1]

def native(name):
    return cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/meter-player-target'/name)),cv2.COLOR_BGR2RGB)

class PlayerTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.inputs=[]
        self.training=BossTraining(ROOT,self.reader,lambda _:None,state_path=Path(tmp.name)/'training.json')
        self.training.state.update(mode='farming',rank=39)
        self.meter=FishingMeter(ROOT,self.training,lambda *a:self.inputs.append(a) or True,lambda _:None)
    def observe(self,name,now):
        with patch('fishing_meter.time.monotonic',return_value=now):return self.meter.confirm_start(native(name))
    def test_current_real_selection_is_within_tolerance_without_max_drag(self):
        from tests.test_fishing_meter import frame
        self.training.state.update(mode='idle')
        data=self.meter.measure(frame('ready-shifted'))
        self.assertEqual((data['player'],data['value'],data['goal']),(1353,1348,1353))
        self.assertTrue(self.meter.close_enough(data))
        self.assertEqual(self.inputs,[])
    def test_tolerance_and_clamping_keep_real_range_mandatory(self):
        d=dict(player=1000,value=990,low=800,high=1500,adaptive=True,y=1000,goal=1000,x=245)
        for v in (990,1000,1010):self.assertTrue(self.meter.close_enough({**d,'value':v}))
        for v in (989,1011,800,1501):self.assertFalse(self.meter.close_enough({**d,'value':v}))
        self.assertFalse(self.meter.close_enough({**d,'high':None}))
        self.assertTrue(self.meter.close_enough({**d,'player':1600,'value':1500}))
        self.assertTrue(self.meter.close_enough({**d,'player':700,'value':801}))
    def test_drag_aims_inside_track_instead_of_forcing_max(self):
        d=dict(player=1000,value=850,low=800,high=1500,adaptive=True,y=1075,goal=1000,x=245)
        p=self.meter.plan(d)
        self.assertGreater(p['to_y'],self.meter.TOP)
        self.assertLess(p['to_y'],d['y'])
    def test_actual_during_empty_then_after_fill_requires_three_new_frames_three_seconds(self):
        during=native('during.png')
        with patch('fishing_meter.time.monotonic',return_value=100):self.meter.arm_start_observation(during)
        self.assertEqual(self.meter.confirmed_starts,0)  # During image is diagnostic only.
        self.assertFalse(self.observe('after.png',100))  # Native checkbox is obscured by rewards.
        self.assertFalse(self.observe('battle.png',101))
        self.assertFalse(self.observe('battle.png',102.5))
        self.assertTrue(self.observe('battle.png',104))
        self.assertEqual((self.meter.confirmed_starts,self.meter.start_proof),(1,'post_release_fishing'))
        self.assertEqual(self.inputs,[])
    def test_actual_battle_can_confirm_without_fill_detector(self):
        battle=native('battle.png')
        self.assertTrue(self.meter.real_battle(battle))
        with patch('fishing_meter.time.monotonic',return_value=100):self.meter.arm_start_observation(battle)
        with patch.object(self.meter,'blue_fill',return_value=False):
            self.assertFalse(self.observe('battle.png',100))
            self.assertFalse(self.observe('battle.png',101.5))
            self.assertTrue(self.observe('battle.png',103))
        self.assertEqual(self.meter.dial_state,'battle')
    def test_single_frame_hook_unknown_and_long_gaps_never_prove_start(self):
        with patch('fishing_meter.time.monotonic',return_value=100):self.meter.arm_start_observation(native('during.png'))
        self.assertFalse(self.observe('battle.png',100))
        self.assertFalse(self.observe('after.png',101.5))
        self.assertFalse(self.observe('battle.png',103))
        self.assertFalse(self.observe('battle.png',111))
        self.assertEqual(self.meter.confirmed_starts,0)
        self.assertEqual(self.inputs,[])
    def test_same_array_cannot_be_counted_as_three_new_captures(self):
        frame=native('battle.png')
        with patch('fishing_meter.time.monotonic',return_value=100):self.meter.arm_start_observation(frame)
        for now in (100,101.5,103):
            with patch('fishing_meter.time.monotonic',return_value=now):self.assertFalse(self.meter.confirm_start(frame))
        self.assertEqual(self.meter.start_seen,1)
    def test_world_rank_stage_place_route_port_and_boss_reject_post_start(self):
        image=native('battle.png')
        with patch('fishing_meter.time.monotonic',return_value=100):self.meter.arm_start_observation(image)
        for x,y,X,Y in ((8,40,46,80),(72,12,130,50),(40,45,68,71),(286,12,434,34),(194,4,246,42)):
            wrong=image.copy();wrong[y:Y,x:X]=0
            with patch('fishing_meter.time.monotonic',return_value=101):self.assertFalse(self.meter.confirm_start(wrong))
        for method in ('boss_mode',):
            with patch.object(self.training,method,return_value=True):self.assertFalse(self.observe('after.png',103))
        with patch.object(self.meter,'at_port',return_value=True):self.assertFalse(self.observe('after.png',103))
        self.assertEqual(self.meter.confirmed_starts,0)
    def test_pending_start_never_repeats_a_hold_or_drag(self):
        with patch('fishing_meter.time.monotonic',return_value=100):self.meter.arm_start_observation(native('battle.png'))
        for now in (101,105,110,120):
            with patch('fishing_meter.time.monotonic',return_value=now):self.meter.step(native('after.png'))
        self.assertEqual(self.inputs,[])
        with patch('fishing_meter.time.monotonic',return_value=131):
            with self.assertRaises(RuntimeError):self.meter.step(native('after.png'))
    def test_timer_or_crossed_weapons_alone_are_not_actual_battle(self):
        for x,y,X,Y in ((188,125,610,165),(153,205,192,245),(307,962,415,1057)):
            image=native('battle.png');image[y:Y,x:X]=0
            self.assertFalse(self.meter.real_battle(image))
