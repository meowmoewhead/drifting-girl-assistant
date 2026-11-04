"""已取消但等不到恢复时,要能说清是哪一步没满足(而不是笼统"在等待")。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.tasks.meter_calibration import MeterCalibration


def make_calibrator(*, screen, fill=False, battle=False, value=None, player=None):
    """构造一个只用于 _observe_retired 的最小实例。"""
    cal = object.__new__(MeterCalibration)
    cal.retired_frame = None
    cal.key = cal.seen_at = None
    cal.seen = 0
    cal.hold = None
    cal.observe_unmet = None
    cal.last_report = 0
    cal.reported_reasons = {}
    cal.events = []
    cal.event = lambda message: cal.events.append(message)
    cal.journal = {'retirement': {}}
    cal.screen = lambda frame, observe_retired=False: screen
    cal.fishing = SimpleNamespace(
        blue_fill=lambda frame, main_known=False: fill,
        real_battle=lambda frame: battle,
        # box 必须非空,否则 selected 不会被调用,数字永远读不到
        box=lambda frame: (0, 0, 1, 1),
        selected=lambda frame, box: value,
        _context=lambda frame: 'ctx')
    cal.world = SimpleNamespace(training=SimpleNamespace(_number=lambda frame, areas: player))
    cal._confirmed = lambda key, seconds=3: False
    return cal


FRAME = np.zeros((1280, 720, 3), dtype=np.uint8)
SCREEN = dict(stage=9, rank=39, route=True, menu=False)


class ObserveRetiredReasonTests(unittest.TestCase):
    def test_unknown_screen_says_so(self):
        cal = make_calibrator(screen=None)
        cal._observe_retired(FRAME)
        self.assertIn('平行世界主画面', cal.observe_unmet)
        self.assertIn('当前未满足', cal.hold)

    def test_no_fill_and_no_battle_names_the_gray_hook_case(self):
        cal = make_calibrator(screen=SCREEN, fill=False, battle=False)
        cal._observe_retired(FRAME)
        self.assertIn('圆内浅蓝填充', cal.observe_unmet)
        self.assertIn('鱼钩灰着', cal.observe_unmet)

    def test_unreadable_numbers_are_distinguished(self):
        cal = make_calibrator(screen=SCREEN, fill=True, value=None, player=1600)
        cal._observe_retired(FRAME)
        self.assertIn('标尺数字读不清', cal.observe_unmet)

        cal = make_calibrator(screen=SCREEN, fill=True, value=1500, player=None)
        cal._observe_retired(FRAME)
        self.assertIn('角色等级读不清', cal.observe_unmet)

    def test_reason_clears_once_numbers_are_readable(self):
        cal = make_calibrator(screen=SCREEN, fill=True, value=1500, player=1600)
        cal._observe_retired(FRAME)
        self.assertIsNone(cal.observe_unmet)


if __name__ == '__main__':
    unittest.main()
