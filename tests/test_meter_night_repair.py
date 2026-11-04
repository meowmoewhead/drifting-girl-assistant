"""Replay unmodified native frames from the 2026-10-11 failed adjustment."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from boss_training import BossTraining
from digit_ocr import DigitReader
from fishing_meter import FishingMeter
from meter_calibration import MeterCalibration

ROOT = Path(__file__).resolve().parents[1]


def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/fishing-meter'/(name+'.png')).convert('RGB'))


class NightMeterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.inputs = []
        self.training = BossTraining(ROOT, self.reader, lambda _: None,
                                     state_path=Path(self.temp.name)/'training_state.json')
        self.training.state.update(rank=39, sailing_sleep_recovery=True,
                                   sailing_meter_adjustment=True)
        self.meter = FishingMeter(ROOT, self.training,
                                  lambda *a: self.inputs.append(a) or True, lambda _: None)

    def test_actual_partial_and_full_night_fill_are_busy(self):
        for name in ('night-hold-during','night-hold-after','night-second-during',
                     'night-second-after','night-filled-prebattle'):
            with self.subTest(name=name):
                native = frame(name)
                self.assertTrue(self.meter.blue_fill(native))
                self.assertTrue(self.meter.busy(native))
                self.assertIsNone(self.meter.read(native))
                self.assertFalse(self.meter.validate('drag',native,{}))
                self.assertFalse(self.meter.validate('start',native,{}))

    def test_empty_night_dial_and_gray_hook_are_not_fill(self):
        for name in ('night-full-range','night-after-drag','night-ready'):
            with self.subTest(name=name):
                self.assertFalse(self.meter.blue_fill(frame(name)))
                self.assertIsNotNone(self.meter.idle_icon(frame(name)))

    def test_warm_color_requires_both_interior_sides_at_the_same_height(self):
        native, empty = frame('night-hold-after').copy(), frame('night-ready')
        native[982:1068,400:420] = empty[982:1068,400:420]
        self.assertFalse(self.meter.blue_fill(native))
        native = empty.copy()
        native[982:1068,340:380] = (121,119,94)  # central hook/animation cannot prove fill
        self.assertFalse(self.meter.blue_fill(native))
        native = empty.copy()
        native[982:1068,280:300] = (121,119,94)  # outside rim cannot prove fill
        native[982:1068,421:440] = (121,119,94)
        self.assertFalse(self.meter.blue_fill(native))

    def test_near_player_selection_preserves_independently_read_upper_limit(self):
        ready = frame('night-ready')
        self.assertEqual(self.meter.read(frame('night-full-range'))['high'],1810)
        data = self.meter.read(ready)
        self.assertEqual((data['value'],data['y'],data['high'],data['goal']),(1806,937,1810,1810))
        self.assertEqual(self.meter.range_cache['high'],1810)
        self.assertTrue(self.meter.close_enough(data))
        self.assertTrue(self.meter.validate('start',ready,data))
        self.assertEqual(self.inputs,[])

    def test_top_without_an_independent_upper_limit_must_reveal(self):
        ready = frame('night-ready')
        data = self.meter.read(ready)
        self.assertEqual(data['value'],1806)
        self.assertIsNone(data['high'])
        self.assertEqual(self.meter.plan(data)['kind'],'reveal')
        self.assertFalse(self.meter.validate('start',ready,data))

    def test_a_new_start_with_fill_is_not_cancelled_again(self):
        self.meter.current,self.meter.goal = 1691,1810
        self.meter.starts,self.meter.phase,self.meter.since = 1,'starting',100
        with patch('fishing_meter.time.monotonic',return_value=104):
            self.meter.prepare(frame('night-hold-after'))
        self.assertEqual(self.inputs,[])
        self.assertEqual(self.meter.starts,1)
        self.assertEqual(self.meter.confirmed_starts,0)  # One frame is not continuous proof.

    def test_hidden_upper_limit_can_trigger_only_below_the_track_stop(self):
        data = dict(player=1841,value=1779,high=None,y=952,adaptive=True)
        # 1841-1779=62：用旧阈值 50 验证"上限未知也能触发"的机制，
        # 生产阈值（GAP）另调，不在本用例范围内。
        with patch.object(MeterCalibration,'GAP',50):
            self.assertTrue(MeterCalibration.needs_adjustment(data))
        self.assertFalse(MeterCalibration.needs_adjustment({**data,'y':937}))
        self.assertFalse(MeterCalibration.needs_adjustment({**data,'player':1829}))
        self.assertFalse(MeterCalibration.needs_adjustment({**data,'adaptive':False}))

    def test_saved_gap_without_a_guessed_upper_limit_only_allows_old_cast_cancel(self):
        self.meter.current,self.meter.goal = 1691,None
        native = frame('night-filled-prebattle')
        self.assertFalse(self.meter.cancellable_cast(native))
        self.meter.saved_player_gap = True
        self.assertTrue(self.meter.cancellable_cast(native))
        self.assertFalse(self.meter.validate('drag',native,{}))
        self.assertFalse(self.meter.validate('start',native,{}))
        self.training.state['sailing_meter_adjustment'] = False
        self.assertFalse(self.meter.cancellable_cast(native))

    def test_actual_stopped_text_edge_69_to_70_keeps_context(self):
        during = frame('night-hold-during')
        after = frame('night-stop-text')
        self.assertEqual(int(during[32,339].max()),69)
        self.assertEqual(int(after[32,339].max()),70)
        self.assertEqual(self.meter._context(during),self.meter._context(after))
        self.assertEqual(self.meter._context(frame('night-full-range')),
                         self.meter._context(after))

    def test_actual_other_location_and_changed_rank_still_rejected(self):
        stopped = frame('night-stop-text')
        self.assertNotEqual(self.meter._context(stopped),
                            self.meter._context(frame('night-moving-location')))
        changed = stopped.copy()
        changed[12:50,72:130] = 0
        self.assertNotEqual(self.meter._context(stopped),self.meter._context(changed))


if __name__ == '__main__':
    unittest.main()
