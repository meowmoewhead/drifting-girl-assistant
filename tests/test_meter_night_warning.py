"""The real dim warning must retain ordinary-fish, stop-X and context guards."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from drift.tasks.meter_calibration import _ScreenContext
from drift.tasks.fishing_meter import FishingMeter
from drift.perception.digit_ocr import DigitReader
ROOT=Path(__file__).resolve().parents[1]

class MeterNightWarningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.reader=DigitReader(ROOT)
    def setUp(self):
        self.world=_ScreenContext(ROOT,self.reader,lambda:None)
        self.world.training.state.update(mode='farming',rank=37,sailing_enabled=False,
            sailing_sleep_recovery=True,sailing_meter_adjustment=True)
        self.inputs=[]
        self.meter=FishingMeter(ROOT,self.world.training,lambda *a:self.inputs.append(a),lambda *_:None)
        self.meter.current,self.meter.goal=1250,1421
        self.meter.saved_player_gap=True
        self.frame=np.asarray(Image.open(ROOT/'tests/fixtures/meter-night-warning/ordinary-stopped.png').convert('RGB'))

    def test_real_night_warning_can_only_use_the_saved_low_correction_path(self):
        self.assertFalse(self.meter.ordinary_preparation(self.frame))
        self.assertTrue(self.meter.cancellable_cast(self.frame))
        self.assertTrue(self.meter.validate('cancel',self.frame,self.meter.cast_data(self.frame)))
        self.world.training.state['sailing_meter_adjustment']=False
        self.assertFalse(self.meter.cancellable_cast(self.frame))
        self.assertEqual(self.inputs,[])

    def test_missing_dot_stem_x_or_boss_and_dark_frames_are_rejected(self):
        for (x,y,X,Y) in ((342,1027,378,1041),(342,960,378,1026),(634,1066,682,1110)):
            bad=self.frame.copy();bad[y:Y,x:X]=0
            self.assertFalse(self.meter.cancellable_cast(bad))
        self.assertFalse(self.meter.cancellable_cast(self.frame//2))
        with patch.object(self.world.training,'boss_mode',return_value=True):
            self.assertFalse(self.meter.cancellable_cast(self.frame))
        self.assertFalse(self.meter.cancellable_cast(self.frame[:1000]))
        self.assertEqual(self.inputs,[])

    def test_wrong_rank_digits_location_and_running_route_cannot_cancel(self):
        expected=self.meter.cast_data(self.frame)
        changed=self.frame.copy();changed[12:34,286:434]=0
        self.assertFalse(self.meter.validate('cancel',changed,expected))
        changed=self.frame.copy();changed[1033:1066,214:274]=0
        self.assertFalse(self.meter.validate('cancel',changed,expected))
        self.assertIsNone(self.meter.scan_cancel_frame(lambda:self.frame.copy(),lambda _:True,expected,lambda:False))
        self.world.training.state['rank']=38
        self.assertIsNone(self.meter.cast_data(self.frame))
        self.assertEqual(self.inputs,[])

    def test_existing_combat_result_and_sleep_screens_do_not_match_the_night_warning(self):
        for name in ('tests/fixtures/meter-max-drag/restored-boss-1.png',
                     'assets/fishing-meter/sleep.png','assets/recovery/login.png'):
            frame=np.asarray(Image.open(ROOT/name).convert('RGB'))
            self.assertFalse(self.meter.ordinary_preparation(frame,allow_fill=True))
        self.assertEqual(self.inputs,[])

if __name__=='__main__':unittest.main()
