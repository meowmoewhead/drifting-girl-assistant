"""Near-player start tolerance, measured from the fresh input frame."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
from drift.perception.digit_ocr import DigitReader
from drift.tasks.meter_calibration import MeterCalibration

ROOT=Path(__file__).resolve().parents[1]
def image(name):
    return cv2.cvtColor(cv2.imread(str(ROOT/'tests/fixtures/meter-start-settle'/name)),cv2.COLOR_BGR2RGB)

def hidden_upper():
    # The old filename described an OCR failure, not actual occlusion: its
    # complete 1604 label is visible. Hide that label for this negative case.
    frame=image('unreadable-upper1513.png')
    frame[929:953,237:283]=0
    return frame

class MeterStartSettleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.inputs=[]
        path=ROOT/'meter_calibration_state.json';exists=Path.is_file
        with patch.object(Path,'is_file',lambda p:False if p==path else exists(p)):
            self.c=MeterCalibration(ROOT,self.reader,lambda *a:self.inputs.append(a),lambda *a:self.inputs.append(a),lambda *a:None)
        self.c.path=Path(self.tmp.name)/path.name
        self.c.world.training.state.update(mode='farming',rank=38)
        self.m=self.c.fishing;self.frame=image('ready1515.png')
        self.data=self.m.read(self.frame)
        self.assertEqual((self.data['value'],self.data['player'],self.data['low'],self.data['high']),(1515,1514,1350,1604))

    def test_real_small_settle_does_not_reject_allowed_near_player_start(self):
        expected={**self.data,'value':1513}
        self.assertTrue(self.m.validate('start',self.frame,expected))
        self.assertEqual(self.inputs,[])

    def test_validated_reading_is_from_current_frame_not_planned_value(self):
        expected={**self.data,'value':1513}
        actual=self.m.start_reading(self.frame,expected)
        self.assertEqual((actual['value'],actual['player'],actual['goal']),(1515,1514,1514))
        self.assertEqual(expected['value'],1513)

    def test_receipt_snapshot_belongs_to_only_the_validated_capture(self):
        self.assertTrue(self.m.validate('start',self.frame,{**self.data,'value':1513}))
        frame,actual=self.m.validated_start
        self.assertIs(frame,self.frame)
        self.assertEqual(actual['value'],1515)
        self.assertIsNot(frame,self.frame.copy())
        self.assertFalse(self.m.validate('start',hidden_upper(),self.data))
        self.assertIsNone(self.m.validated_start)

    def test_large_selection_change_and_changed_range_or_context_are_rejected(self):
        for changed in ({'value':1504},{'context':'changed'},{'low':1351},{'high':1603},{'adaptive':False}):
            with self.subTest(changed=changed):
                self.assertFalse(self.m.validate('start',self.frame,{**self.data,**changed}))
        self.assertEqual(self.inputs,[])

    def test_unreadable_current_upper_limit_is_not_replaced_by_expected_range(self):
        unknown=hidden_upper()
        self.assertIsNone(self.m.read(unknown))
        self.assertFalse(self.m.validate('start',unknown,self.data))
        self.assertEqual(self.inputs,[])

    def test_previously_low_confidence_frame_has_a_complete_independent_upper_label(self):
        data=self.m.read(image('unreadable-upper1513.png'))
        self.assertEqual((data['value'],data['player'],data['low'],data['high']),(1513,1514,1350,1604))

    def test_new_world_rank_boss_fill_and_port_cannot_use_the_tolerance(self):
        for target,method,value in ((self.m,'world_matches',False),(self.m.training,'boss_mode',True),
                                    (self.m,'at_port',True),(self.m,'busy',True)):
            with self.subTest(method=method),patch.object(target,method,return_value=value):
                self.assertFalse(self.m.validate('start',self.frame,{**self.data,'value':1513}))
        self.assertEqual(self.inputs,[])

    def test_current_reading_must_still_be_within_ten_levels_of_current_player(self):
        with patch.object(self.m,'read',return_value={**self.data,'player':1526}):
            self.assertFalse(self.m.validate('start',self.frame,self.data))

    def test_nonadaptive_start_retains_exact_selection_requirement(self):
        data={**self.data,'adaptive':False,'player':1516,'goal':1516}
        with patch.object(self.m,'read',return_value=data):
            self.assertFalse(self.m.validate('start',self.frame,{**data,'value':1513}))
            self.assertTrue(self.m.validate('start',self.frame,data))

    def test_idle_preparation_reads_numeric_fields_once_without_renewing_deadline(self):
        c=self.c;context=self.m._context(self.frame)
        c.journal.update(pending=True,rank=38,stage=9,restore_running=True,deadline=1000,
            window_restarted_at=800,attempts={'cancel':1,'drag':1},sailing_attempts={'stop':1},
            proof=None,stopped_context=context,stage_strokes=c.world._stage_strokes(self.frame).hex())
        c.phase='prepare';c.off_verified=True;c.context=context;c._save()
        with patch('drift.tasks.meter_calibration.time.time',return_value=900), \
             patch('drift.tasks.meter_calibration.time.monotonic',return_value=900), \
             patch.object(c,'_observe_cancel_window',return_value=False), \
             patch.object(self.m,'measure',wraps=self.m.measure) as measure:
            self.assertTrue(c.step(self.frame.copy()))
        self.assertEqual(measure.call_count,1)
        self.assertEqual(c.journal['deadline'],1000)
        self.assertEqual(c.journal['attempts'],{'cancel':1,'drag':1})
        self.assertEqual(self.inputs,[])

if __name__=='__main__':unittest.main()
