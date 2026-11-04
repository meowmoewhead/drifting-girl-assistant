"""标尺校准必须说明"为什么现在没在校准"。

用户反馈"标尺等级校准似乎没用"。实测原因不是读数坏了,而是触发条件不满足时
这条链**一句话都不说**:日志里只有点击记录,面板上 `player_level` 又是空的,
于是看起来就像功能坏了。这里守住可观测性:

- 不在平行世界 / 数字没读清 / 差距不够 时,`step()` 都要给出可读原因;
- 原因要进 `status()['hold']`,面板据此显示,不必翻日志;
- 原因每 60 秒最多记一条,不能刷屏;
- 真触发时 `hold` 要清空。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from digit_ocr import DigitReader
from meter_calibration import MeterCalibration

ROOT = Path(__file__).resolve().parents[1]


class CalibrationObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.events = []
        journal = ROOT / "meter_calibration_state.json"
        original_is_file = Path.is_file
        with patch.object(Path, "is_file", lambda p: False if p == journal else original_is_file(p)):
            self.controller = MeterCalibration(
                ROOT, self.reader,
                lambda *args: False, lambda *args: True, self.events.append)
        self.controller.path = Path(self.temp.name) / "meter_calibration_state.json"
        self.controller.world.training._save = self.controller._save

    def tearDown(self):
        self.temp.cleanup()

    def test_unreadable_screen_reports_a_reason_and_surfaces_it(self):
        blank = np.zeros((1280, 720, 3), dtype=np.uint8)
        self.assertFalse(self.controller.step(blank))
        self.assertFalse(self.controller.status()["pending"])
        hold = self.controller.status()["hold"]
        self.assertTrue(hold, "必须留下暂不触发的原因")
        self.assertIn("平行世界", hold)
        self.assertEqual(len(self.events), 1)
        self.assertIn("标尺校准", self.events[0])
        self.assertIn(hold, self.events[0])

    def test_reason_is_throttled_to_one_event_per_minute(self):
        blank = np.zeros((1280, 720, 3), dtype=np.uint8)
        for _ in range(5):
            self.controller.step(blank)
        self.assertEqual(len(self.events), 1, "同一分钟内只应记一条")
        # 状态里的原因始终是最新的(面板要能看到)
        self.assertTrue(self.controller.status()["hold"])

    def test_status_exposes_hold_field(self):
        self.assertIn("hold", self.controller.status())
        self.assertIsNone(self.controller.status()["hold"])

    def test_journal_records_are_not_touched_by_reporting(self):
        blank = np.zeros((1280, 720, 3), dtype=np.uint8)
        before = json.dumps(self.controller.journal, sort_keys=True)
        self.controller.step(blank)
        self.assertEqual(json.dumps(self.controller.journal, sort_keys=True), before,
                         "只报原因不应改写校准记录")


if __name__ == "__main__":
    unittest.main()
