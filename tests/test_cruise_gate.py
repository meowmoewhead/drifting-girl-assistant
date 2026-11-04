"""航巡放行条件的回归。

2026-10-26 12:38 的事故:标尺窗口以 ``workflow_limit`` 归档后,归档记录里
``released`` 一直是 False,``awaiting_recovery`` 因此长期为真;只要
``meter_active`` 把它算成"标尺在跑",航巡看门狗与开局配置就永远不会启动,
船在港口停了 40 分钟。标尺解除归档又要求原生航巡在跑,两边互锁。

这里钉住三件事:
1. 归档的标尺(active=False)不挡航巡;
2. 还没结束的标尺窗口(active=True)必须挡;
3. 其余阻塞项(阶段/中断/开局配置/日常清算/采集/其它任务)语义不变。
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from drift.tasks.cruise_gate import BLOCKER_KEYS, cruise_blockers


class _Meter:
    """只保留 `active` 的最小标尺替身。"""

    def __init__(self, active=False, awaiting_recovery=False):
        self.active = active
        self.awaiting_recovery = awaiting_recovery


class _Task:
    def __init__(self, active):
        self.active = active


class _MethodActive:
    def active(self):  # 方法形式的 active 也要能读
        return True


class CruiseGateTest(unittest.TestCase):
    def gates(self, **kwargs):
        return cruise_blockers(**kwargs)

    def test_retired_meter_does_not_block_cruise(self):
        # 事故现场:退役标尺 awaiting_recovery=True,但 active=False。
        gates = self.gates(calibrator=_Meter(active=False, awaiting_recovery=True))
        self.assertFalse(gates['meter_active'])
        self.assertEqual([], [k for k, v in gates.items() if v])

    def test_live_meter_window_blocks_cruise(self):
        gates = self.gates(calibrator=_Meter(active=True))
        self.assertTrue(gates['meter_active'])
        self.assertTrue(any(gates.values()))

    def test_missing_meter_blocks_nothing(self):
        self.assertFalse(self.gates(calibrator=None)['meter_active'])

    def test_callable_active_is_called(self):
        self.assertTrue(self.gates(calibrator=_MethodActive())['meter_active'])
        self.assertTrue(self.gates(stone=_MethodActive())['stone_active'])

    def test_stage_and_interrupt_block(self):
        self.assertTrue(self.gates(stage='raid_wait')['stage'])
        self.assertFalse(self.gates(stage='main')['stage'])
        self.assertTrue(self.gates(handling_interrupt=True)['handling_interrupt'])

    def test_setup_only_blocks_when_asked(self):
        self.assertTrue(self.gates(setup_busy=True)['setup_active'])
        self.assertFalse(self.gates(setup_busy=True, include_setup=False)['setup_active'])
        # 开局配置自己动手时要忽略自己,否则它永远点不下第一下
        self.assertFalse(any(self.gates(setup_busy=True, include_setup=False).values()))

    def test_daily_cleanup_blocks(self):
        self.assertTrue(self.gates(daily_cleanup=True)['daily_cleanup'])

    def test_stone_and_other_tasks_block(self):
        self.assertTrue(self.gates(stone=_Task(True))['stone_active'])
        self.assertFalse(self.gates(stone=_Task(False))['stone_active'])
        self.assertTrue(self.gates(tasks=(None, _Task(False), _Task(True)))['other_task'])
        self.assertFalse(self.gates(tasks=(None, _Task(False)))['other_task'])
        self.assertFalse(self.gates(tasks=())['other_task'])

    def test_all_idle_lets_cruise_run(self):
        gates = self.gates(calibrator=_Meter(active=False, awaiting_recovery=True),
                           stone=_Task(False), tasks=(None, _Task(False)))
        self.assertEqual([], [k for k, v in gates.items() if v])

    def test_keys_match_status_payload(self):
        # 面板 /api/status 的 cruise_gate 直接展开这个 dict,键不能变。
        self.assertEqual(set(BLOCKER_KEYS), set(self.gates()))


if __name__ == '__main__':
    unittest.main()
