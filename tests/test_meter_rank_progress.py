"""An expired prior-RANK record cannot seize an actually restored native route."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from drift.perception.digit_ocr import DigitReader
from drift.tasks.meter_calibration import MeterCalibration

ROOT=Path(__file__).resolve().parents[1]


class MeterRankProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)
        cls.frame=np.asarray(Image.open(ROOT/'tests/fixtures/meter-night-warning/progressed-route.png').convert('RGB'))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.inputs=[]
        path=ROOT/'meter_calibration_state.json';exists=Path.is_file
        with patch.object(Path,'is_file',lambda p:False if p==path else exists(p)):
            self.c=MeterCalibration(ROOT,self.reader,lambda *a:self.inputs.append(a),
                                    lambda *a:self.inputs.append(a),lambda *_:None)
        self.c.path=Path(self.temp.name)/'meter_calibration_state.json'
        self.c.journal.update(pending=True,rank=37,stage=9,restore_running=True,
            deadline=990,cycles=10,value=1250,high=1421,attempts={},sailing_attempts={'stop':1})
        self.c.phase='stop';self.c.expired_observe_until=115
        self.c.world.training.state=dict(mode='farming',rank=37,sailing_enabled=False,
            sailing_sleep_recovery=True,sailing_meter_adjustment=True,fishing_meter_limits={'high':1421})
        self.c.fishing.range_cache={'high':1421}
        self.c._save()

    def tearDown(self):self.temp.cleanup()

    def test_real_progressed_route_is_read_only_and_does_not_relax_input_guard(self):
        c=self.c;data=c._progressed_route(self.frame)
        self.assertEqual((data['rank'],data['stage'],data['player'],data['value']),(38,9,1365,1604))
        self.assertIsNone(c.screen(self.frame))
        with patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            self.assertFalse(c.validate_sailing('stop',self.frame))
            self.assertFalse(c.input_allowed())
        self.assertEqual(self.inputs,[])

    def test_three_seconds_archive_exact_failure_without_success_or_renewal(self):
        c=self.c;original=c.path.read_bytes()
        with patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            for now in (100,101.5):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    self.assertTrue(c.step(self.frame.copy()))
                self.assertTrue(c.active)
            with patch('drift.tasks.meter_calibration.time.monotonic',return_value=103):
                self.assertFalse(c.step(self.frame.copy()))
        self.assertFalse(c.active)
        result=c.journal['resolution']
        self.assertEqual(result['kind'],'externally_resumed_after_rank_progress')
        self.assertFalse(result['calibration_completed'])
        self.assertEqual(Path(result['archive']).read_bytes(),original)
        self.assertEqual(result['sha256'],hashlib.sha256(original).hexdigest())
        self.assertEqual((c.journal['rank'],c.journal['deadline'],c.journal['cycles']),(37,990,10))
        self.assertEqual(c.journal['attempts'],{})
        self.assertEqual(c.journal['sailing_attempts'],{'stop':1})
        self.assertEqual(c.world.training.state,{'mode':'idle','fishing_meter_limits':{'high':1421}})
        self.assertIsNone(c.world.rank)
        self.assertEqual(self.inputs,[])

    def test_unknown_world_rank_stage_off_and_missing_fill_are_rejected(self):
        c=self.c
        for target,attribute,value in ((c.world,'_parallel',False),(c.world,'_rank_value',37),
                (c.world,'_rank_value',40),(c.world,'_rank_value',None),(c,'_stage',8)):
            with self.subTest(attribute=attribute,value=value),patch.object(target,attribute,return_value=value):
                self.assertIsNone(c._progressed_route(self.frame))
        bad=self.frame.copy();bad[0:65,180:255]=0
        self.assertIsNone(c._progressed_route(bad))
        c.journal['restore_running']=False
        self.assertIsNone(c._progressed_route(self.frame))
        self.assertEqual(self.inputs,[])

    def test_obscured_frame_resets_continuous_evidence_without_inputs(self):
        c=self.c;original=c.path.read_bytes()
        with patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            for now,image,wanted in ((100,self.frame,False),(101.5,np.zeros_like(self.frame),False),
                (103,self.frame,False),(104.5,self.frame,False),(106,self.frame,True)):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    self.assertEqual(c._resolve_expired(image),wanted)
        self.assertEqual(Path(c.journal['resolution']['archive']).read_bytes(),original)
        self.assertEqual(self.inputs,[])

if __name__=='__main__':unittest.main()
