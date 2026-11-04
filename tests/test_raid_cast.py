"""RAID 甩杆辅助：没有真实战斗时才补点，最多两次，战斗一出现就放手。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift.tasks.raid_cast import RaidCastAssist


class Harness:
    def __init__(self, battle_after=None):
        self.now = 0.0
        self.battle = False
        self.battle_after = battle_after      # 第 N 次点击后战斗成立
        self.clicks = []
        self.events = []
        self.frame = object()
        self.assist = RaidCastAssist(
            battle=lambda frame: self.battle,
            cast=self._cast, event=self.events.append, clock=lambda: self.now)

    def _cast(self, x, y, label):
        self.clicks.append((x, y))
        if self.battle_after is not None and len(self.clicks) >= self.battle_after:
            self.battle = True
        return True

    def tick(self, seconds=1.0, times=1):
        for _ in range(times):
            self.now += seconds
            self.assist.step(self.frame)


class RaidCastTests(unittest.TestCase):
    def test_action_waits_for_first_delay(self):
        h = Harness()
        h.assist.arm()
        h.tick(1.0, 2)
        self.assertEqual(h.clicks, [])          # 2 秒时还没到 due（首次延迟 3 秒）
        h.tick(1.0, 1)                           # t=3.0 到点
        self.assertEqual(h.clicks, [(360, 1040)])

    def test_no_tap_when_battle_already_running(self):
        h = Harness()
        h.battle = True
        h.assist.arm()
        h.tick(1.0, 10)
        self.assertEqual(h.clicks, [])
        self.assertFalse(h.assist.armed)        # 战斗成立后放手

    def test_second_tap_after_retry_delay_then_gives_up(self):
        h = Harness(battle_after=None)           # 点了也不开打
        h.assist.arm()
        h.tick(1.0, 3)                            # 第 1 次
        self.assertEqual(len(h.clicks), 1)
        h.tick(1.0, 13)                           # 还没到 15 秒
        self.assertEqual(len(h.clicks), 1)
        h.tick(1.0, 5)                            # 到点后第 2 次
        self.assertEqual(len(h.clicks), 2)
        h.tick(1.0, 40)                           # 用尽后不再点
        self.assertEqual(len(h.clicks), 2)
        self.assertFalse(h.assist.armed)
        self.assertTrue(any('停止补点' in e for e in h.events))

    def test_tap_stops_as_soon_as_battle_starts(self):
        h = Harness(battle_after=1)
        h.assist.arm()
        h.tick(1.0, 4)
        self.assertTrue(h.battle)
        h.tick(1.0, 10)
        self.assertEqual(len(h.clicks), 1)
        self.assertFalse(h.assist.armed)

    def test_not_armed_does_nothing(self):
        h = Harness()
        h.tick(1.0, 30)
        self.assertEqual(h.clicks, [])


if __name__ == '__main__':
    unittest.main()
