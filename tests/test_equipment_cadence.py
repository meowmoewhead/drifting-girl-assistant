"""Verify reduced UI actions, cooldowns and safe failed-entry backoff."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from equipment_upgrade import EquipmentUpgrade
from tests.test_equipment_upgrade import ROOT, image


class EquipmentCadenceTests(unittest.TestCase):
    def controller(self):
        self.actions=[]
        return EquipmentUpgrade(ROOT,lambda *a:self.actions.append(('tap',a[-1])),
            lambda *a:self.actions.append(('swipe',a[-1])),
            lambda *a:self.actions.append(('hold',a[3])) or True,lambda _:None)

    def run_cycle(self,affordable=None):
        screen,clock=['main'],[10.]
        u=self.controller()
        def tap(*args):
            self.actions.append(('tap',args[-1]))
            screen[0]='top' if '打开' in args[-1] else 'main'
        def swipe(*args):
            self.actions.append(('swipe',args[-1]))
            screen[0]='top' if args[1]<args[3] else 'bottom'
        def hold(*args):
            self.actions.append(('hold',args[3],args[2]))
            clock[0]+=args[2]
            return True
        u.click,u.swipe,u.hold=tap,swipe,hold
        if affordable is not None:u.affordable=lambda frame,key:key in affordable
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:clock[0])):
            for _ in range(100):
                clock[0]+=1
                u.step(image(screen[0]))
                if u.cycles:break
        self.assertEqual(u.cycles,1)
        self.assertFalse(u.active)
        self.assertEqual([a for a in self.actions if a[0]=='tap'],
            [('tap','武器强化：打开鱼饼面板'),('tap','武器强化：关闭鱼饼面板')])
        self.assertEqual(len([a for a in self.actions if a[0]=='swipe']),1)
        return u,clock

    def test_five_weapons_once_then_no_reentry_before_saved_interval(self):
        u,clock=self.run_cycle()
        self.assertEqual([a[1] for a in self.actions if a[0]=='hold'],['rod','harpoon','cannon','turret','magic'])
        self.assertEqual(u.MAX_HOLDS,5)
        self.assertEqual(u.interval,900)
        self.assertEqual([a[2] for a in self.actions if a[0]=='hold'],[12]*5)
        count=len(self.actions)
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:clock[0])):
            self.assertEqual(u.status()['next_check_in_seconds'],900)
            clock[0]=u.next_check_at-.01
            for _ in range(3):self.assertFalse(u.step(image('main')))
            self.assertEqual(len(self.actions),count)
            clock[0]=u.next_check_at
            self.assertFalse(u.step(image('main')))
            self.assertTrue(u.step(image('main')))
        self.assertEqual(len(self.actions),count+1)

    def test_partly_affordable_pass_does_not_loop_back_to_spend_more(self):
        self.run_cycle({'rod'})
        self.assertEqual([a for a in self.actions if a[0]=='hold'],[('hold','rod',12)])

    def test_no_affordable_weapon_still_closes_after_one_inspection(self):
        self.run_cycle(set())
        self.assertEqual([a for a in self.actions if a[0]=='hold'],[])

    def test_saved_user_interval_is_not_clamped_by_controller(self):
        u=self.controller();u.interval=720
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:10)):
            u._finish(True)
        self.assertEqual(u.next_check_at,730)
        self.assertEqual(u.cycles,0)

    def test_failed_open_returns_without_inputs_or_completed_upgrade_only_after_three_main_frames(self):
        u=self.controller();u.phase='opening';u.phase_since=10
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:15)):
            for _ in range(2):self.assertTrue(u.step(image('main')))
            self.assertEqual(u.phase,'opening')
            self.assertFalse(u.step(image('main')))
        self.assertEqual(self.actions,[])
        self.assertEqual(u.cycles,0)
        self.assertEqual(u.next_check_at,915)

    def test_unknown_after_failed_open_keeps_original_deadline(self):
        u=self.controller();u.phase='opening';u.phase_since=10
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:36)):
            with self.assertRaisesRegex(RuntimeError,'入口后未确认'):
                u.step(np.zeros((1280,720,3),dtype=np.uint8))
        self.assertEqual(self.actions,[])
        self.assertEqual(u.cycles,0)
        self.assertEqual(u.phase_since,10)

    def test_opening_guard_declines_battle_before_any_input(self):
        u=self.controller();u.opening_guard=lambda f:False
        for _ in range(4):self.assertFalse(u.step(image('main')))
        self.assertEqual(self.actions,[])
        self.assertEqual(u.phase,'idle')

    def test_new_capture_must_confirm_entry_and_context(self):
        for frame in ((image('main')*.4).astype(np.uint8),image('top'),
                      np.asarray(Image.open(ROOT/'tests/fixtures/equipment/opening-interrupted-rank35.png').convert('RGB'))):
            u=self.controller();u.capture=lambda:frame
            for _ in range(4):self.assertFalse(u.step(image('main')))
            self.assertEqual(self.actions,[])
            self.assertEqual(u.phase,'idle')


if __name__=='__main__':unittest.main()
