"""Replay the actual glowing cancel X and keep ordinary-cast/frame guards."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from boss_training import BossTraining
from digit_ocr import DigitReader
from fishing_meter import FishingMeter

ROOT = Path(__file__).resolve().parents[1]


def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/meter-cancel-glow'/name).convert('RGB'))


class MeterCancelGlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.inputs = []
        training = BossTraining(ROOT,self.reader,lambda _:None,
                               state_path=Path(temp.name)/'training.json')
        training.state.update(mode='farming',rank=38,sailing_enabled=False,
                              sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.meter = FishingMeter(ROOT,training,lambda *a:self.inputs.append(a),lambda _:None)
        self.meter.current,self.meter.player,self.meter.goal = 1409,1472,1604
        self.meter.saved_player_gap = True

    def test_actual_warning_keeps_original_threshold_and_accepts_fixed_orange_x(self):
        image = frame('warning-x.png')
        score = cv2.minMaxLoc(cv2.matchTemplate(image[1066:1110,634:682],
            self.meter.assets['cancel-cast'],cv2.TM_CCOEFF_NORMED))[1]
        self.assertLess(score,.94)
        self.assertGreater(score,.93)
        self.assertTrue(self.meter.cancel_button(image))
        self.assertTrue(self.meter.cancellable_cast(image))
        self.assertTrue(self.meter.validate('cancel',image,self.meter.cast_data(image)))
        self.assertFalse(self.meter.validate('start',image,{}))
        self.assertFalse(self.meter.validate('drag',image,{}))
        self.assertEqual(self.inputs,[])

    def test_actual_three_dots_require_all_dots_and_missing_x_still_cannot_cancel(self):
        dots = frame('waiting-dots-x.png')
        self.assertTrue(self.meter.cancel_button(dots))
        self.assertTrue(self.meter.cancellable_cast(dots))
        for x in (322,360,398):
            unknown = dots.copy()
            unknown[997:1024,x-13:x+13] = 0
            self.assertFalse(self.meter.cancellable_cast(unknown))
        missing = frame('warning-no-x.png')
        self.assertFalse(self.meter.cancel_button(missing))
        self.assertFalse(self.meter.cancellable_cast(missing))
        self.assertEqual(self.inputs,[])

    def test_no_color_wrong_size_boss_and_changed_rank_reject_the_actual_x(self):
        image = frame('warning-x.png')
        gray = image.copy()
        crop = gray[1066:1110,634:682]
        crop[:] = cv2.cvtColor(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)
        self.assertFalse(self.meter.cancel_button(gray))
        self.assertFalse(self.meter.cancel_button(image[:1279]))
        with patch.object(self.meter.training,'boss_mode',return_value=True):
            self.assertFalse(self.meter.cancellable_cast(image))
        self.meter.training.state['rank'] = 37
        self.assertIsNone(self.meter.cast_data(image))
        self.assertEqual(self.inputs,[])

    def test_unknown_route_only_waits_for_another_new_stopped_frame(self):
        first,second = frame('warning-x.png'),frame('warning-x.png')
        expected = self.meter.cast_data(first)
        captures,routes = iter((first,second)),iter((None,False))
        result = self.meter.scan_cancel_frame(lambda:next(captures),lambda _:next(routes),
                                             expected,lambda:False)
        self.assertIs(result,second)
        self.assertEqual(self.inputs,[])

    def test_new_running_arrow_or_changed_world_abandons_before_any_input(self):
        image = frame('warning-x.png')
        expected = self.meter.cast_data(image)
        self.assertIsNone(self.meter.scan_cancel_frame(lambda:image.copy(),lambda _:True,
                                                       expected,lambda:False))
        changed = image.copy()
        changed[45:71,40:68] = 0
        self.assertIsNone(self.meter.scan_cancel_frame(lambda:changed,lambda _:None,
                                                       expected,lambda:False))
        self.assertEqual(self.inputs,[])

    def test_unknown_frames_keep_both_six_second_and_forty_frame_limits(self):
        image = frame('warning-x.png')
        expected = self.meter.cast_data(image)
        count,clock = [0],[100.]
        def capture():
            count[0] += 1
            clock[0] += .01
            return image.copy()
        with patch('fishing_meter.time.monotonic',side_effect=lambda:clock[0]):
            self.assertIsNone(self.meter.scan_cancel_frame(capture,lambda _:None,expected,lambda:False))
        self.assertEqual(count[0],40)
        count[0] = 0
        def slow_capture():
            count[0] += 1
            clock[0] += 2.1
            return image.copy()
        with patch('fishing_meter.time.monotonic',side_effect=lambda:clock[0]):
            self.assertIsNone(self.meter.scan_cancel_frame(slow_capture,lambda _:None,expected,lambda:False))
        self.assertEqual(count[0],3)
        self.assertEqual(self.inputs,[])


if __name__ == '__main__':
    unittest.main()
