"""圆盘选中值必须两路投票,不能一路差一点就否决整帧。

真机现场(2026-10-11 11:5x,角色 Lv1829、圆盘 1691):
- 视图0 的 OCR 置信 **0.9641**(差一点没过 0.97),视图1 是 **1.0000**;
- 原实现遇到视图0 不合格就 `return None`,于是 measure() 永远拿不到值,
  日志一分钟一条"计量与角色等级没读清",校准完全动不了;
- 数字框顶端被金框/指针压住一点点就会这样,所以是"随机但高频"的失败。

修好后:单路可信就用它;两路都读出则必须一致;两路都不可信才算读不清。
"""
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from digit_ocr import DigitReader
from drift.tasks.parallel_world import ParallelWorld

ROOT = Path(__file__).resolve().parents[1]
#: 现场帧与其中数字框的位置(探针实测)
LIVE_FRAMES = {
    "probe-1.png": (210, 977, 281, 1018),
    "probe-2.png": (210, 977, 281, 1018),
    "probe-3.png": (210, 977, 281, 1018),
    "meter-check.png": (210, 1065, 281, 1106),
}


def make_fishing(root=ROOT):
    reader = DigitReader(ROOT)

    def no_nav(*args, **kwargs):
        raise AssertionError("test must not navigate")

    world = ParallelWorld(root, no_nav, no_nav, lambda message: None, reader,
                          training={"enabled": True},
                          fishing_action=lambda *a: None, sailing_action=lambda *a: None,
                          defense_action=lambda *a: None)
    return world.fishing


class SelectedVoteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fishing = make_fishing()

    def test_live_frames_now_read_the_selected_value(self):
        """现场帧里圆盘数字必须读得出来(修复前这里全是 None)。"""
        expected = {"probe-1.png": 1691, "probe-2.png": 1691, "probe-3.png": 1691,
                    "meter-check.png": 1195}
        for name, box in LIVE_FRAMES.items():
            path = ROOT / "tests" / "fixtures" / "legacy-captures" / name
            if not path.is_file():
                continue
            frame = np.asarray(Image.open(path).convert("RGB"))
            self.assertEqual(self.fishing.selected(frame, box), expected[name], name)

    def test_one_view_short_of_the_threshold_still_votes(self):
        """一路 0.964、一路 1.000 时必须出值 —— 这正是原来被否决的场景。"""
        frame = np.asarray(Image.open(ROOT / "tests/fixtures/legacy-captures/probe-2.png").convert("RGB"))
        box = (210, 977, 281, 1018)
        self.assertEqual(self.fishing.selected(frame, box), 1691)

    def test_disagreeing_views_are_not_a_value(self):
        """两路都可信但不一致时必须判读不清,不能任选一路。

        注意这帧的真实情况是"视图0 0.9641 不过线、视图1 1.0000",所以要让两路都可信,
        得人为把视图0 的置信度抬到 0.99 并改成另一个值。
        """
        fishing = make_fishing()          # 独立实例,避免污染共享 reader
        reader = fishing.training.reader
        hits = []

        class ConflictingReader:
            def read_many(self,views):
                results = reader.read_many(views)
                for index,view in enumerate(views):
                    shape = getattr(view, "shape", None)
                    if shape == (27, 60, 3):
                        hits.append(shape)
                        result=results[index]
                        results[index]=type(result)(result.text,1500,.99)
                return results
        fishing.training.reader=ConflictingReader()
        try:
            frame = np.asarray(Image.open(ROOT / "tests/fixtures/legacy-captures/probe-2.png").convert("RGB"))
            self.assertIsNone(fishing.selected(frame, (210, 977, 281, 1018)))
            self.assertEqual(len(hits), 1, "必须真的评估到第一个视图")
        finally:
            fishing.training.reader=reader

    def test_both_views_unreadable_is_still_none(self):
        """两路都不可信时仍应判读不清(不能因为放宽投票就把噪声当数字)。"""
        blank = np.zeros((1280, 720, 3), dtype=np.uint8)
        self.assertIsNone(self.fishing.selected(blank, (210, 977, 281, 1018)))

    def test_measure_returns_the_live_gap(self):
        """整链走通:读数 -> 差距,且差距确实超过 50 级阈值。"""
        frame = np.asarray(Image.open(ROOT / "tests/fixtures/legacy-captures/probe-2.png").convert("RGB"))
        data = self.fishing.measure(frame, remember=False)
        self.assertIsNotNone(data, "现场帧必须能读出完整计量")
        self.assertEqual(data["value"], 1691)
        self.assertEqual(data["player"], 1829)
        self.assertGreater(data["player"] - data["value"], 50)


if __name__ == "__main__":
    unittest.main()
