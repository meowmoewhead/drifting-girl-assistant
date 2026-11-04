"""Native BOSS arrival must not starve read-only calibration completion."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from digit_ocr import DigitReader
from meter_calibration import MeterCalibration

ROOT=Path(__file__).resolve().parents[1]

def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/meter-max-drag'/name).convert('RGB'))

class MeterRouteCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.inputs=[]
        path=ROOT/'meter_calibration_state.json'
        is_file=Path.is_file
        with patch.object(Path,'is_file',lambda p:False if p==path else is_file(p)):
            self.c=MeterCalibration(ROOT,self.reader,lambda *a:self.inputs.append(a),
                lambda *a:self.inputs.append(a),lambda *a:None)
        c=self.c
        c.path=Path(self.temp.name)/path.name
        c.journal.update(pending=True,rank=37,stage=9,restore_running=True,
            deadline=1090,proof='paired_hold',resume_requested=True,
            attempts={'reveal':1,'drag':2,'start':1},sailing_attempts={'open':1,'start':1})
        c.world.training.state=dict(mode='farming',rank=37,sailing_enabled=False)
        c.phase='running'
        c._save()

    def test_real_boss_is_only_accepted_by_readonly_resume_probe(self):
        c=self.c
        for name in ('restored-boss-1.png','restored-boss-2.png','restored-boss-3.png'):
            f=frame(name)
            self.assertTrue(c.world.training.boss_mode(f))
            self.assertIsNone(c.screen(f))
            self.assertIsNone(c.fishing.measure(f))
            self.assertEqual(c.screen(f,resumed_route=True),
                dict(rank=37,stage=9,menu=False,route=True))
            self.assertTrue(c.fishing.blue_fill(f))
            with patch('meter_calibration.time.time',return_value=1000):
                for action in ('stop','open','start'):
                    self.assertFalse(c.validate_sailing(action,f))
                self.assertFalse(c.frame_blocker(f) is None)
        self.assertEqual(self.inputs,[])

    def test_completion_requires_three_new_frames_and_three_seconds(self):
        c=self.c
        with patch('meter_calibration.time.time',return_value=1000):
            for now,name in ((100,'restored-boss-1.png'),(101.5,'restored-boss-2.png')):
                with patch('meter_calibration.time.monotonic',return_value=now):c.step(frame(name))
                self.assertTrue(c.active)
            with patch('meter_calibration.time.monotonic',return_value=103):
                c.step(frame('restored-boss-3.png'))
        self.assertFalse(c.active)
        self.assertEqual(c.cycles,1)
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual(self.inputs,[])

    def test_restart_after_expiry_only_archives_and_confirms_no_more_input(self):
        c=self.c
        original=c.path.read_bytes()
        c.expired_observe_until=115
        with patch('meter_calibration.time.time',return_value=1100):
            self.assertFalse(c.input_allowed())
            for now,name in ((100,'restored-boss-1.png'),(101.5,'restored-boss-2.png'),(103,'restored-boss-3.png')):
                with patch('meter_calibration.time.monotonic',return_value=now):c.step(frame(name))
        self.assertFalse(c.active)
        resolution=c.journal['resolution']
        self.assertEqual(resolution['kind'],'verified_resume_after_expiry')
        self.assertEqual(Path(resolution['archive']).read_bytes(),original)
        self.assertEqual(resolution['sha256'],hashlib.sha256(original).hexdigest())
        self.assertEqual(c.journal['deadline'],1090)
        self.assertEqual(c.journal['attempts'],{'reveal':1,'drag':2,'start':1})
        self.assertEqual(c.journal['sailing_attempts'],{'open':1,'start':1})
        self.assertEqual(self.inputs,[])

    def test_missing_proof_resume_or_reserved_start_cannot_accept_boss(self):
        c=self.c;f=frame('restored-boss-1.png')
        for key,value in (('proof',None),('resume_requested',False),('restore_running',False),
                          ('sailing_attempts',{}),('pending',False)):
            old=c.journal[key];c.journal[key]=value
            self.assertIsNone(c.screen(f,resumed_route=True),key)
            c.journal[key]=old
        c.phase='prepare'
        self.assertIsNone(c.screen(f,resumed_route=True))
        self.assertEqual(self.inputs,[])

    def test_off_menu_obscured_wrong_rank_stage_or_world_cannot_confirm(self):
        c=self.c
        for name in ('restored-menu.png','stuck-before.png'):
            self.assertIsNone(c.screen(frame(name),resumed_route=True),name)
        # This native frame has a visible check/outline and real fill; its
        # former failure was the same checkbox glow fixed by this repair.
        self.assertIsNotNone(c.screen(frame('restored-obscured.png'),resumed_route=True))
        obscured=frame('restored-obscured.png').copy()
        obscured[1137:1192,208:269]=0
        self.assertIsNone(c.screen(obscured,resumed_route=True))
        f=frame('restored-boss-1.png')
        for attr,value in (('rank',38),('stage',8)):
            old=c.journal[attr];c.journal[attr]=value
            self.assertIsNone(c.screen(f,resumed_route=True),attr)
            c.journal[attr]=old
        self.assertIsNone(c.screen(np.zeros_like(f),resumed_route=True))
        self.assertEqual(self.inputs,[])

    def test_fill_loss_resets_continuous_completion_proof(self):
        c=self.c
        missing_fill=frame('restored-obscured.png').copy()
        missing_fill[945:1080,285:435]=0
        for now,f,wanted in ((100,frame('restored-boss-1.png'),False),
                            (101.5,missing_fill,False),
                            (103,frame('restored-boss-2.png'),False),
                            (104.5,frame('restored-boss-3.png'),False),
                            (106,frame('restored-boss-1.png'),True)):
            with patch('meter_calibration.time.monotonic',return_value=now):
                self.assertEqual(c._confirm_resumed_route(f),wanted)
        self.assertEqual(self.inputs,[])

if __name__=='__main__':unittest.main()
