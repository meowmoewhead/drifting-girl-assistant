"""触发前的门控拒绝要写清具体条件(不是笼统"不是主画面"/"没读清")。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.tasks.meter_calibration import MeterCalibration

FRAME = np.zeros((1280, 720, 3), dtype=np.uint8)


def make_screen(*, parallel=True, boss=False, menu=False, route=True, port=False,
                rank=39, stage=9, journal=None, active=False):
    cal = object.__new__(MeterCalibration)
    cal.screen_reason = None
    # active 是只读属性,由 journal['pending'] 推导
    base = dict(rank=39, stage=9, pending=bool(active))
    base.update(journal or {})
    cal.journal = base
    cal.world = SimpleNamespace(
        _parallel=lambda frame: parallel,
        _rank_value=lambda frame: rank,
        training=SimpleNamespace(boss_mode=lambda frame: boss),
    )
    cal.sailing = SimpleNamespace(
        menu=lambda frame: menu,
        running=lambda frame: route,
        _fixed=lambda frame, name, box, threshold: True,
    )
    cal.fishing = SimpleNamespace(
        _main_visible=lambda frame: True,
        at_port=lambda frame: port,
    )
    cal._stage = lambda frame: stage
    return cal


class ScreenReasonTests(unittest.TestCase):
    def test_not_parallel_world_names_the_world_check(self):
        cal = make_screen(parallel=False)
        self.assertIsNone(cal.screen(FRAME))
        self.assertIn('不在平行世界画面', cal.screen_reason)

    def test_boss_marker_is_reported_separately(self):
        cal = make_screen(boss=True)
        self.assertIsNone(cal.screen(FRAME))
        self.assertIn('BOSS 标记', cal.screen_reason)

    def test_port_or_unknown_route_is_reported(self):
        cal = make_screen(route=None)
        self.assertIsNone(cal.screen(FRAME))
        self.assertIn('航巡状态未知', cal.screen_reason)

    def test_context_mismatch_reports_both_ranks(self):
        cal = make_screen(rank=38, stage=8, journal=dict(rank=39, stage=9), active=True)
        self.assertIsNone(cal.screen(FRAME))
        self.assertIn('与世界上下文不符', cal.screen_reason)
        self.assertIn('RANK38/8', cal.screen_reason)
        self.assertIn('RANK39/9', cal.screen_reason)

    def test_good_frame_has_no_reason(self):
        cal = make_screen()
        self.assertIsNotNone(cal.screen(FRAME))
        self.assertIsNone(cal.screen_reason)


class MeasureUnmetTests(unittest.TestCase):
    def build(self, **overrides):
        cal = object.__new__(MeterCalibration)
        fishing = dict(
            _main_visible=lambda frame: True,
            world_matches=lambda frame: True,
            at_port=lambda frame: False,
            adaptive_auto_on=lambda frame: True,
            box=lambda frame: (0, 0, 1, 1),
            selected=lambda frame, box: 1500,
        )
        fishing.update(overrides)
        cal.fishing = SimpleNamespace(**fishing)
        cal.world = SimpleNamespace(training=SimpleNamespace(_number=lambda frame, areas: 1600))
        return cal

    def test_each_step_gets_its_own_wording(self):
        cases = [
            (dict(_main_visible=lambda f: False), '主画面元素不完整'),
            (dict(world_matches=lambda f: False), '世界不匹配'),
            (dict(at_port=lambda f: True), '船在港口'),
            (dict(adaptive_auto_on=lambda f: False), '自适应 AUTO 未开启'),
            (dict(box=lambda f: None), '计量框没找到'),
            (dict(selected=lambda f, b: None), '计量数字读不清'),
        ]
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, self.build(**overrides)._measure_unmet(FRAME))

    def test_level_reading_failure_is_distinct(self):
        cal = self.build()
        cal.world = SimpleNamespace(training=SimpleNamespace(_number=lambda frame, areas: None))
        self.assertIn('角色等级读不清', cal._measure_unmet(FRAME))


if __name__ == '__main__':
    unittest.main()

