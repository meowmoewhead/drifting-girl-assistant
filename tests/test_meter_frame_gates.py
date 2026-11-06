"""meter_frame_allowed 的守门项要被拆开,否则 90 秒超时无法定位。

2026-10-10 那次失败留下的现场记录:``sailing_attempts`` 只有一次 ``stop``、
``attempts`` 全空 —— 整段 90 秒都卡在 prepare 阶段的每帧准入检查里,而那个检查
原先写成一个布尔表达式,不成立时什么都不说。

守门分两类,分别可断言:
- :meth:`window_blocker` 记录/时间类(90 秒窗口、是否已确认停航)
- :meth:`frame_blocker`  画面类(平行世界 / 航行状态 / 阶段笔画 / 地点)
"""
from pathlib import Path
import json
import shutil
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from digit_ocr import DigitReader
from meter_calibration import MeterCalibration

ROOT = Path(__file__).resolve().parents[1]


class MeterFrameBlockerTests(unittest.TestCase):
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
                ROOT, self.reader, lambda *args: False, lambda *args: True, self.events.append)
        self.controller.path = Path(self.temp.name) / "meter_calibration_state.json"
        self.controller.world.training._save = self.controller._save

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def blank():
        return np.zeros((1280, 720, 3), dtype=np.uint8)

    def test_window_gates_report_their_own_reason(self):
        # 没有进行中的校准 → 窗口/开关类守门先说话
        self.assertIn("90 秒", self.controller.window_blocker())
        # 进行中但尚未确认停航
        self.controller.journal.update(pending=True, deadline=1e12)
        self.assertIn("停航", self.controller.window_blocker())
        # 确认停航后窗口类守门通过
        self.controller.off_verified = True
        self.assertIsNone(self.controller.window_blocker())
        # 窗口过期
        self.controller.journal["deadline"] = 0
        self.assertIn("90 秒", self.controller.window_blocker())

    def test_frame_gates_report_their_own_reason(self):
        # 尺寸
        self.assertIn("1280x720", self.controller.frame_blocker(np.zeros((10, 10, 3), dtype=np.uint8)))
        # 空白帧必然不在平行世界
        self.assertIn("平行世界", self.controller.frame_blocker(self.blank()))

    def test_combined_blocker_prefers_the_window_reason(self):
        # 窗口没开就不该去做画面判定,否则报出的原因会误导
        self.assertIn("90 秒", self.controller.meter_frame_blocker(self.blank()))

    def test_hold_reason_is_recorded_for_the_prepare_phase(self):
        self.controller.journal.update(pending=True, deadline=1e12)
        self.controller.off_verified = True
        self.controller._set("prepare")
        self.assertFalse(self.controller.meter_frame_allowed(self.blank()))
        self.assertTrue(self.controller.status()["hold"])
        self.assertIn("平行世界", self.controller.status()["hold"])

    def test_hold_is_cleared_when_the_gates_pass_again(self):
        self.controller.hold = "旧原因"
        self.controller.journal.update(pending=True, deadline=1e12)
        self.controller.off_verified = True
        with patch.object(self.controller, "meter_frame_blocker", return_value=None):
            self.assertTrue(self.controller.meter_frame_allowed(self.blank()))
        self.assertIsNone(self.controller.status()["hold"])

    def test_calibration_world_satisfies_the_runtime_action_wrapper(self):
        """运行期的 fishing_action 包装器把 calibrator.world 当平行世界对象用。

        2026-10-10 真机第一次成功触发校准后,就因为这里缺 .fishing 直接崩掉
        (``'_ScreenContext' object has no attribute 'fishing'``)。这次触发真金白银
        地停了航,所以这些接口必须在无 ParallelWorld 的情况下也齐备。
        """
        world = self.controller.world
        for name in ("fishing", "sailing", "training", "_parallel"):
            self.assertTrue(hasattr(world, name), f"缺少 {name}")
        self.assertIs(world.fishing, self.controller.fishing)
        self.assertIs(world.sailing, self.controller.sailing)
        self.assertTrue(world.training.options["auto_sailing"])
        self.assertTrue(world.training.options["meter_auto_adjust"])
        # 未开始时状态为空;开始校准会先停航并把 sailing_enabled 置 False
        self.assertFalse(world.training.state.get("sailing_enabled", False))

    def isolated_state(self, payload):
        """借项目根做一次"带上已存在记录"的构造,并在构造后立刻把路径指到临时文件。

        ``MeterCalibration`` 的构造会按 root 读取 ``meter_calibration_state.json`` 并
        解析素材,所以记录文件必须在构造前就位、素材又必须能找到 —— 这里先把记录写进
        真实路径再还原,避免污染仓库里的那份状态。
        """
        journal = ROOT / "meter_calibration_state.json"
        backup = journal.read_bytes() if journal.is_file() else None

        def restore():
            if backup is None:
                journal.unlink(missing_ok=True)
            else:
                journal.write_bytes(backup)
        self.addCleanup(restore)
        journal.write_text(json.dumps(payload), encoding="utf-8")
        return ROOT

    def test_stale_pending_record_keeps_its_restore_obligation_and_input_budget(self):
        """没有新画面证明恢复时，重启不能清空原记录或刷新输入额度。"""
        root = self.isolated_state({
            "version": 1, "pending": True, "cycles": 0, "next_check": 0, "rank": 39,
            "stage": 8, "restore_running": True, "deadline": time.time() - 600,
            "value": 1420, "high": 1604, "attempts": {"start": 1}, "sailing_attempts": {"stop": 1},
        })
        original = (root / 'meter_calibration_state.json').read_bytes()
        events = []
        controller = MeterCalibration(root, self.reader, lambda *args: False,
                                      lambda *args: True, events.append)
        controller.world.training._save = controller._save
        self.assertTrue(controller.active)
        self.assertTrue(controller.journal['restore_running'])
        self.assertEqual(controller.journal['attempts'], {'start': 1})
        self.assertEqual(controller.journal['sailing_attempts'], {'stop': 1})
        self.assertEqual((root / 'meter_calibration_state.json').read_bytes(), original)
        self.assertEqual(events, [])
        self.assertIn("90 秒", controller.window_blocker())

    def test_fresh_pending_record_is_kept(self):
        """还在窗口内的进行中记录必须保留(要接着恢复原航巡状态)。"""
        root = self.isolated_state({
            "version": 1, "pending": True, "cycles": 0, "next_check": 0, "rank": 39,
            "stage": 8, "restore_running": True, "deadline": time.time() + 60,
            "value": 1420, "high": 1604, "attempts": {},
        })
        events = []
        controller = MeterCalibration(root, self.reader, lambda *args: False,
                                      lambda *args: True, events.append)
        self.assertTrue(controller.journal["pending"])
        self.assertEqual(events, [])

    def test_input_window_starts_when_the_adjustment_can_actually_begin(self):
        """等船停下、等画面稳定的时间不该吃掉输入预算。

        90 秒限的是"能发输入的那一段"。原先从触发就计时,2026-10-10 那次失败里
        整段时间还没发出任何计量输入(attempts 为空)窗口就走完了。进入 prepare
        时重新计时,输入次数上限不变。
        """
        controller = self.controller
        controller.journal.update(pending=True, deadline=time.time() - 1, attempts={},
                                  stage_strokes="00")
        controller.off_verified = True
        self.assertIn("90 秒", controller.window_blocker())
        controller._set("prepare")
        self.assertIsNone(controller.window_blocker(), "进入 prepare 后应重新有一整段窗口")
        self.assertAlmostEqual(controller.journal["deadline"], time.time() + controller.LIMIT, delta=2)
        # 重复调用不会无限续期
        first = controller.journal["deadline"]
        controller._set("prepare")
        self.assertEqual(controller.journal["deadline"], first)
        # 重新停航再回到 prepare 也不能续期。
        controller._set('stop')
        controller._set('prepare')
        self.assertEqual(controller.journal['deadline'], first)
        # 上限仍然是 90 秒,不是"永久允许"
        controller.journal["deadline"] = time.time() - 1
        self.assertIn("90 秒", controller.window_blocker())


if __name__ == "__main__":
    unittest.main()
