"""任务弹窗红叉被吃掉时的重按决策。

之前的行为:``driftinggirl_task_popup_close`` 是 ``stage="*"`` 的中断规则,
一旦切到 ``task_popup_return``,扫描过滤条件只保留 ``stage ==
task_popup_return``,通配规则不再参与,于是弹窗只要没关掉就只能等到 15 秒
期限触发"未识别到后续界面"停机。这里固定住"重按/放弃"的边界。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.core.matcher import popup_reclick_decision


class PopupReclickDecisionTest(unittest.TestCase):
    def test_notice_gone_waits_for_return_rule(self):
        self.assertEqual('wait', popup_reclick_decision(False, True, 0))
        self.assertEqual('wait', popup_reclick_decision(False, True, 3))

    def test_notice_visible_before_delay_still_waits(self):
        self.assertEqual('wait', popup_reclick_decision(True, False, 0))

    def test_notice_visible_after_delay_reclicks(self):
        self.assertEqual('reclick', popup_reclick_decision(True, True, 0))
        self.assertEqual('reclick', popup_reclick_decision(True, True, 2))

    def test_limit_is_respected(self):
        self.assertEqual('give_up', popup_reclick_decision(True, True, 3))
        self.assertEqual('give_up', popup_reclick_decision(True, True, 4))
        self.assertEqual('give_up', popup_reclick_decision(True, True, 1, limit=1))

    def test_custom_limit_boundary(self):
        for count in (0, 1, 2):
            self.assertEqual('reclick', popup_reclick_decision(True, True, count, limit=3))


if __name__ == '__main__':
    unittest.main()
