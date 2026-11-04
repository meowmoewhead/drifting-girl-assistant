"""A checked adaptive AUTO may glow without losing the low-meter trigger."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from drift.perception.digit_ocr import DigitReader
from drift.tasks.meter_calibration import MeterCalibration
ROOT=Path(__file__).resolve().parents[1]

class MeterAutoGlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)
        cls.frame=np.asarray(Image.open(ROOT/'tests/fixtures/meter-night-warning/adaptive-glow.png').convert('RGB'))
        cls.boss=np.asarray(Image.open(ROOT/'tests/fixtures/meter-night-warning/current-boss.png').convert('RGB'))
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.inputs=[]
        path=ROOT/'meter_calibration_state.json';exists=Path.is_file
        with patch.object(Path,'is_file',lambda p:False if p==path else exists(p)):
            self.c=MeterCalibration(ROOT,self.reader,lambda *a:self.inputs.append(a),
                lambda *a:self.inputs.append(a) or True,lambda *_:None)
        self.c.path=Path(self.temp.name)/'meter_calibration_state.json'
    def tearDown(self):self.temp.cleanup()

    def test_real_glow_keeps_check_outline_and_complete_numeric_reading(self):
        c=self.c;f=self.frame
        self.assertIsNone(c.world.training._match(f,'main-auto',(276,881,310,913),.91))
        self.assertTrue(c.fishing.adaptive_auto_on(f))
        data=c.fishing.measure(f,remember=False)
        self.assertEqual((data['player'],data['value'],data['low'],data['high']),(1575,1514,1350,1604))
        # 这张 fixture 的差距是 61 级：用历史阈值 50 验证"超过阈值即需校准"的机制，
        # 生产阈值（GAP）另调，不在这条用例里。
        with patch.object(MeterCalibration,'GAP',50):
            self.assertTrue(c.needs_adjustment(data))
        self.assertEqual(self.inputs,[])

    def test_actual_gap_above50_confirms_without_stopping_route(self):
        c=self.c
        # 这张截图停在主界面，没有普通鱼准备/明确休眠窗口，所以候选窗口策略
        # 由 test_meter_continuation 单独覆盖；这里只验证"差距超过阈值就确认并保留航巡"
        # 的机制，用一个满足窗口的读数代替真实等待。
        with patch.object(MeterCalibration,'GAP',50), \
                patch.object(c.fishing,'ordinary_preparation',return_value=True), \
                patch('drift.tasks.meter_calibration.time.time',return_value=1000):
            for now in (100,101.5):
                with patch('drift.tasks.meter_calibration.time.monotonic',return_value=now):
                    self.assertFalse(c.step(self.frame.copy()))
                self.assertFalse(c.active);self.assertEqual(self.inputs,[])
            with patch('drift.tasks.meter_calibration.time.monotonic',return_value=103):
                self.assertTrue(c.step(self.frame.copy()))
        self.assertTrue(c.active);self.assertEqual(c.journal['rank'],38)
        self.assertEqual(c.journal['trigger_player'],1575)
        self.assertEqual(c.journal['value'],1514)
        self.assertEqual(self.inputs,[])
        # 确认后直接进入执行窗口（prepare）；航巡上下文的核对挪到了窗口第一步，
        # 所以这里断言的是"没有停航巡"这件事本身。
        self.assertEqual(c.phase,'prepare')
        self.assertEqual(c.journal['route_policy'],'preserve')
        self.assertTrue(c.journal['restore_running'])
        self.assertFalse(c.off_verified)

    def test_missing_check_outline_dark_unknown_and_wrong_dimensions_are_rejected(self):
        m=self.c.fishing
        for box in ((283,887,305,910),(274,879,282,915),(276,881,310,913)):
            f=self.frame.copy();x,y,X,Y=box;f[y:Y,x:X]=0
            self.assertFalse(m.adaptive_auto_on(f),str(box))
        # Normalized pattern scores alone do not establish brightness; the
        # independent footer brightness guard must still reject a dark page.
        self.assertFalse(m._main_visible(self.frame//2))
        self.assertIsNone(m.measure(self.frame//2,remember=False))
        self.assertFalse(m.adaptive_auto_on(self.frame[:1000]))
        self.assertEqual(self.inputs,[])

    def test_existing_fill_and_current_task_boss_still_block_drag_start_and_cancel(self):
        c=self.c;m=c.fishing;data=m.measure(self.frame,remember=False)
        for action in ('drag','start'):
            self.assertFalse(m.validate(action,self.frame,data))
        self.assertTrue(c.world.training.boss_mode(self.boss))
        self.assertIsNone(c.screen(self.boss))
        self.assertIsNone(m.measure(self.boss,remember=False))
        self.assertFalse(m.cancellable_cast(self.boss))
        self.assertEqual(self.inputs,[])

if __name__=='__main__':unittest.main()
