"""Protect RAID quest rewards from generic quest and battle inputs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import shutil
import unittest

import numpy as np
from PIL import Image

from drift.tasks.raid_reward import AUTO, AUTO_START, RaidQuestVisual, RaidReward


ROOT = Path(__file__).resolve().parents[1]


def load(filename):
    with Image.open(ROOT/'assets'/filename) as image:
        return np.array(image.convert('RGB'))


class RaidQuestRewardTests(unittest.TestCase):
    def setUp(self):
        fixture = TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        self.directory = Path(fixture.name)
        (self.directory/'assets/task-popup').mkdir(parents=True)
        shutil.copy2(ROOT/'assets/task-popup/kraken-arrival.png', self.directory/'assets/task-popup')
        shutil.copytree(ROOT/'assets/raid-reward', self.directory/'assets/raid-reward')
        self.visual = RaidQuestVisual(ROOT)
        self.travel = load('task-popup/kraken-arrival.png')
        self.ready = self.travel.copy()
        self.ready[721:751,564:606] = self.visual.templates['check']
        self.ready[797:843,487:606] = self.visual.templates['finish']
        self.main = self.travel.copy()
        self.main[380:875] = 0
        roster = self.visual.templates['roster']
        self.main[398:449,31:83] = roster
        self.main[393:404,8:19] = (24,201,249)
        self.frame_id = 0
        self.mode = True
        self.menu = self.battle = self.moving = False
        self.enabled = True
        self.now = 1795771000.
        self.inputs = []
        self.events = []
        self.scene = SimpleNamespace(
            sailing=SimpleNamespace(running=lambda f:self.mode,
                menu=lambda f:self.menu, sale_selected=lambda f:True, upgrade_selected=lambda f:True),
            _off_icon=lambda f:not self.mode,
            settings_key=lambda f:((True,False,True,True,True),'rare'),
            _battle=lambda f:self.battle, fishing=SimpleNamespace(_main_visible=lambda f:not self.menu),
        )
        self.task = self.new_task()

    def new_task(self):
        return RaidReward(self.directory,self.scene,lambda f:self.moving,self.send,
                          self.events.append,lambda:self.enabled,'emulator-5554',clock=lambda:self.now)

    def send(self, kind, point, options):
        self.assertTrue(self.task.reserve(kind))
        self.inputs.append((kind,point))
        return True

    def see(self, frame, count=3, *, fresh=True):
        for _ in range(count):
            self.frame_id += int(fresh)
            self.now += .65
            self.task.step(frame.copy(), self.frame_id)

    def test_three_success_fixture_waits_for_alchemy(self):
        card = self.visual.card(self.travel)
        self.assertEqual((585,734), card.route)
        self.assertIsNone(card.finish)
        for name in ('task-popup/notice.png','task-popup/kraken-stage3.png','raid/task-popup.png',
                     'raid/result-reference.png'):
            self.assertIsNone(self.visual.card(load(name)), name)

    def test_transient_overlay_or_unchanged_frame_cannot_claim(self):
        self.see(self.ready, count=7, fresh=False)
        self.assertEqual([], self.inputs)
        self.battle = True
        self.see(self.ready)
        self.assertEqual([], self.inputs)

    def test_claim_reservation_survives_restart(self):
        self.see(self.ready)
        self.assertEqual([('finish',(546,820))],self.inputs)
        self.task = self.new_task()
        self.see(self.ready)
        self.assertEqual(1,len(self.inputs))
        self.assertIn('issued_unverified',self.task.status()['rewards'].values())

    def test_reward_requires_post_click_visual_effect(self):
        self.see(self.ready)
        self.assertFalse(self.task.observe_claim(self.main,self.ready))
        after = self.main.copy()
        after[557:590,340:374] = self.visual.templates['reward-effect']
        self.assertTrue(self.task.observe_claim(after,self.ready))
        self.assertIn('confirmed',self.task.status()['rewards'].values())
        self.assertFalse(self.task.observe_claim(after,self.ready))

    def test_same_rank_and_world_required_on_fresh_frame(self):
        self.see(self.ready,1)
        point = self.visual.card(self.ready).finish
        self.assertTrue(self.task.validate('finish',point,self.ready,None))
        changed = self.ready.copy()
        changed[:79] = load('task-popup/notice.png')[:79]
        self.assertFalse(self.task.validate('finish',point,changed,None))

    def test_alchemy_route_stops_and_restores_running_auto(self):
        self.see(self.travel)
        self.assertEqual('close-entry',self.inputs[-1][0])
        self.see(self.main)
        self.assertEqual(('stop',AUTO),self.inputs[-1])
        self.mode = False
        self.see(self.main,6)
        self.assertEqual('reopen',self.inputs[-1][0])
        self.see(self.travel)
        self.assertEqual(('route',(585,734)),self.inputs[-1])
        self.moving = True
        self.see(self.main)
        self.assertEqual('route',self.inputs[-1][0])
        self.moving = False
        self.see(self.main)
        self.assertEqual('arrive',self.inputs[-1][0])
        self.see(self.ready)
        self.see(self.main,6)
        self.assertEqual(('menu',AUTO),self.inputs[-1])
        self.menu = True
        self.see(self.main)
        self.assertEqual(('start',AUTO_START),self.inputs[-1])
        self.menu, self.mode = False, True
        self.see(self.main)
        self.assertFalse(self.task.active)
        self.assertEqual(1,sum(kind=='finish' for kind,_ in self.inputs))

    def test_expired_and_disabled_tasks_do_not_submit(self):
        self.see(self.ready,1)
        self.now = self.task.journal['step_deadline']
        self.see(self.ready,1)
        self.assertFalse(self.task.active)
        self.enabled = False
        self.see(self.ready)
        self.assertEqual([],self.inputs)


if __name__=='__main__':
    unittest.main()
