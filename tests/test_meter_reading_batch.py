"""Real R38 crop failures and read-only candidate consensus timing."""
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
from PIL import Image

from drift.perception.digit_ocr import NumberReading
from tests import test_meter_calibration as base


class MeterReadingBatchTests(unittest.TestCase):
    setUpClass=classmethod(base.MeterCalibrationTests.setUpClass.__func__)
    setUp=base.MeterCalibrationTests.setUp
    tearDown=base.MeterCalibrationTests.tearDown

    def test_actual_1604_margin_no_longer_discards_complete_range_reading(self):
        m=self.controller.fishing
        for name in ('prepare-1.png','prepare-2.png'):
            frame=np.asarray(Image.open(base.ROOT/'tests/fixtures/meter-reading-20261030'/name).convert('RGB'))
            data=m.measure(frame,remember=False)
            self.assertIsNotNone(data)
            self.assertEqual((data['value'],data['player'],data['high'],data['goal']),(1402,1542,1604,1542))
            self.assertIsNone(data['low'])
            # A covered lower label still needs revealing; this never guesses
            # the missing bound or starts fishing with an incomplete range.
            self.assertEqual(m.plan(data)['kind'],'reveal')
            self.assertFalse(m.close_enough(data))
            self.assertFalse(m.validate('start',frame,data))

    def test_batch_keeps_two_regions_confidence_and_conflict_rejection(self):
        class Reader:
            def __init__(self,readings):self.readings=readings;self.calls=[]
            def read(self,_):raise AssertionError('batch reader must not issue individual inferences')
            def read_many(self,views):self.calls.append(len(views));return self.readings
        good=lambda v,c=.99:NumberReading(str(v),v,c)
        cases=[([good(1542)]*4,1542),
               ([good(1542),good(1543),good(1542),good(1542)],None),
               ([good(1542),good(1542),good(1542,.5),good(1542,.5)],None)]
        training=self.controller.world.training
        frame=np.zeros((1280,720,3),dtype=np.uint8)
        for readings,expected in cases:
            reader=Reader(readings)
            with patch.object(training,'reader',reader):
                value=training._number(frame,((76,1248,137,1278),(79,1249,136,1277)))
            self.assertEqual(value,expected)
            self.assertEqual(reader.calls,[4])

    def test_candidate_can_collect_slow_votes_but_stop_start_proof_stays_strict(self):
        c=self.controller
        for now,expected in ((100,False),(111,False),(122,True)):
            with patch('meter_calibration.time.monotonic',return_value=now):
                self.assertEqual(c._confirmed('same',max_gap=c.CANDIDATE_FRAME_GAP),expected)
        c.key=None;c.last_seen=0
        for now in (100,111,122):
            with patch('meter_calibration.time.monotonic',return_value=now):
                self.assertFalse(c._confirmed('same'))

    def test_candidate_context_changes_or_long_gaps_still_reset_consensus(self):
        c=self.controller
        for now,key in ((100,'one'),(111,'one'),(122,'two'),(138,'two')):
            with patch('meter_calibration.time.monotonic',return_value=now):
                self.assertFalse(c._confirmed(key,max_gap=c.CANDIDATE_FRAME_GAP))
        self.assertEqual(c.seen,1)


if __name__=='__main__':unittest.main()
