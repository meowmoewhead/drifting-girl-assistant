"""开局航巡配置：逐项补差，全部确认后才点开始。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drift.tasks.cruise_setup import CLOSE_AT, CruiseSetup, DESIRED


class FakeSailing:
    """脚本化的菜单：selected/sale/upgrade 由字典控制，点击回调改字典。"""

    def __init__(self, *, running=False, menu=False, values=None, sale=False, upgrade=False):
        self._running = running
        self._menu = menu
        self.values = dict(values or DESIRED)
        self.sale = sale
        self.upgrade = upgrade

    def running(self, frame):
        return self._running

    def menu(self, frame):
        return self._menu

    def selected(self, frame, name):
        return self.values.get(name)

    def sale_selected(self, frame):
        return self.sale

    def upgrade_selected(self, frame):
        return self.upgrade


class Harness:
    def __init__(self, sailing, start_running=True):
        self.now = 0.0
        self.start_running = start_running
        self.sailing = sailing
        self.clicks = []
        self.events = []
        self.frame = object()
        self.setup = CruiseSetup(sailing=sailing, click=self._click,
                                 event=self.events.append, clock=lambda: self.now)

    def _click(self, x, y, label):
        self.clicks.append((x, y, label))
        if (x, y) == (233, 45):
            self.sailing._menu = True
        elif (x, y) == (360, 873):
            self.sailing._running = self.start_running
            self.sailing._menu = False          # 点开始会关掉设置菜单
        elif (x, y) == (576, 647):
            self.sailing.sale = True
        elif (x, y) == (576, 761):
            self.sailing.upgrade = True
        elif (x, y) == CLOSE_AT:
            self.sailing._menu = False          # 放弃时把菜单收回去
        else:                                  # 目标勾选：翻转对应横坐标的值
            for name, tx in (("pearl", 272), ("material", 345), ("amber", 417),
                             ("gem", 490), ("boss", 562)):
                if tx == x:
                    self.sailing.values[name] = not self.sailing.values[name]
                    break
        return True

    def tick(self, seconds=1.0, times=1):
        for _ in range(times):
            self.now += seconds
            self.frame=object()
            self.setup.step(self.frame)


class CruiseSetupTests(unittest.TestCase):
    def test_already_running_does_nothing(self):
        h = Harness(FakeSailing(running=True, menu=False))
        h.tick(1.0, 10)
        self.assertEqual(h.clicks, [])
        self.assertTrue(h.setup.done)

    def test_full_sequence_opens_sets_sale_upgrade_and_starts(self):
        sail = FakeSailing(values={"pearl": False, "material": True, "amber": True,
                                   "gem": True, "boss": False}, sale=False, upgrade=False)
        h = Harness(sail)
        h.tick(1.0, 40)                       # 打开菜单 → 逐项补 → 开始
        kinds = [c[2] for c in h.clicks]
        self.assertEqual(h.clicks[0], (233, 45, "航巡开局配置：打开航巡设置"))
        self.assertIn("航巡开局配置：勾选 pearl", kinds)
        self.assertIn("航巡开局配置：勾选 material", kinds)
        self.assertIn("航巡开局配置：勾选 boss", kinds)
        self.assertIn("航巡开局配置：勾选自动售卖", kinds)
        self.assertIn("航巡开局配置：勾选自动强化", kinds)
        self.assertEqual(h.clicks[-1], (360, 873, "航巡开局配置：开始航巡"))
        self.assertTrue(sail.values["pearl"] and not sail.values["material"])
        self.assertTrue(sail.sale and sail.upgrade)
        self.assertTrue(h.setup.done)

    def test_no_start_until_everything_matches(self):
        sail = FakeSailing(values=dict(DESIRED), sale=False, upgrade=True)
        h = Harness(sail)
        h.tick(1.0, 12)
        kinds = [c[2] for c in h.clicks]
        self.assertIn("航巡开局配置：勾选自动售卖", kinds)
        self.assertEqual(h.clicks[-1][0], 360)   # 补完才点开始
        self.assertTrue(h.setup.done)

    def test_menu_disappears_gives_up_without_blind_clicks(self):
        sail = FakeSailing(values=dict(DESIRED), sale=False, upgrade=True)
        h = Harness(sail)
        h.tick(1.0, 3)                        # 连续确认停航后打开菜单
        sail._menu = False                     # 菜单没出现
        h.tick(1.0, 25)
        self.assertTrue(all(c[0] != 360 for c in h.clicks))
        self.assertGreater(h.setup.failures, 0)

    def test_unknown_readings_do_not_trigger_clicks(self):
        sail = FakeSailing(values={"pearl": None, "material": None, "amber": None,
                                   "gem": None, "boss": None}, sale=None, upgrade=None)
        h = Harness(sail)
        h.tick(1.0, 30)
        self.assertEqual([c for c in h.clicks if c[0] == 360], [])
        self.assertGreater(h.setup.failures, 0)


    def test_menu_closed_with_unknown_running_keeps_ownership_until_verified(self):
        sail = FakeSailing(values=dict(DESIRED), sale=True, upgrade=True)
        h = Harness(sail, start_running=None)   # 点开始后 running 仍判不出来
        h.tick(1.0, 12)                        # 打开菜单 → 全部符合 → 点开始
        self.assertEqual(h.clicks[-1][0], 360)
        sail._menu = False
        h.tick(1.0, 10)
        self.assertFalse(h.setup.done)
        self.assertTrue(h.setup.busy)
        self.assertIn("等待连续确认", h.setup.hold)
        actions=list(h.clicks)
        sail._running=True
        h.tick(1.0,2)
        self.assertFalse(h.setup.done)
        h.tick()
        self.assertTrue(h.setup.done)
        self.assertFalse(h.setup.busy)
        self.assertEqual(h.clicks,actions)

    def test_unknown_route_never_opens_auto(self):
        h=Harness(FakeSailing(running=None,menu=False))
        h.tick(1,30)
        self.assertEqual(h.clicks,[])
        self.assertFalse(h.setup.done)

    def test_unverified_start_expires_without_reopening_unknown_auto(self):
        h=Harness(FakeSailing(sale=True,upgrade=True),start_running=None)
        h.tick(1,12);h.sailing._menu=False
        actions=list(h.clicks)
        h.tick(1,80)
        self.assertEqual(h.clicks,actions)
        self.assertGreater(h.setup.failures,0)

    def test_scene_change_rejects_further_settings_input(self):
        h=Harness(FakeSailing(menu=True))
        current={'scene':(9,30),'menu':True}
        h.setup.observe=lambda _:current
        h.tick()
        current={'scene':(9,31),'menu':True}
        h.tick()
        # 世界/RANK 变了就不再发任何设置输入，但仍必须把菜单收回去，
        # 否则主界面规则集认不出画面会永久空点（2026-10-29 实测卡死）。
        self.assertEqual(h.clicks, [(CLOSE_AT[0], CLOSE_AT[1],
                                     '航巡开局配置：放弃并关闭设置菜单')])
        self.assertEqual(h.setup.failures,1)
        self.assertFalse(h.sailing._menu)

    def test_give_up_with_closed_menu_sends_no_click(self):
        # 菜单已经被别的东西关掉了：放弃时不该再补一次多余的点击。
        h=Harness(FakeSailing(menu=False))
        h.setup.state='menu'
        h.setup.deadline=h.now+50
        h.tick()
        self.assertEqual(h.clicks,[])
        self.assertEqual(h.setup.failures,1)

    def test_repeated_frame_cannot_confirm_stopped_route(self):
        h=Harness(FakeSailing())
        for _ in range(5):
            h.now+=1;h.setup.step(h.frame)
        self.assertEqual(h.clicks,[])

if __name__ == "__main__":
    unittest.main()
