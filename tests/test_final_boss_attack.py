"""40 级 BOSS 的组织攻击与点击攻击只在打 40 BOSS 时生效。

用户要求把这两种攻击从全局开关里拎出来,单独成项、只在打 40 BOSS 时生效、开了立即生效。

游戏事实:平行世界最后一关就是 RANK 39 的海神关,打赢后 RANK 变 40(通关);
海神遭遇判定本就写死只在 RANK 39 成立。

攻击必须从当前截图解析；同 RANK 普通鱼、连续窗口内目标变化都不能继承强制开关。
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from digit_ocr import DigitReader
from drift.tasks.boss_defense import BossDefense
from drift.tasks.final_boss_attack import FINAL_BOSS_RANK, forced_attacks, is_final_boss, options_for
from drift.tasks.parallel_world import ParallelWorld

ROOT = Path(__file__).resolve().parents[1]


class Reader:
    def read(self, _):
        from types import SimpleNamespace
        return SimpleNamespace(value=39, confidence=.999)


class TrainingStub:
    def __init__(self, options):
        self.options = options
        self.state = {'rank': None}


def make_world(root, rank, options=None):
    """造一个真的 ParallelWorld(不做导航),用它验证调用点的开关注入。"""
    reader = DigitReader(ROOT)
    world = ParallelWorld(ROOT, lambda *a: None, lambda *a: None, lambda message: None,
                          reader, training=options, training_state_path=Path(root) / 'state.json',
                          fishing_action=lambda *a: None, sailing_action=lambda *a: None,
                          defense_action=lambda *a: None)
    world.rank = rank
    return world


def frame(name):
    return np.asarray(Image.open(ROOT / 'tests/fixtures/boss-defense' / f'{name}.png').convert('RGB'))


class FinalBossAttackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def defense(self, options):
        training = TrainingStub(options)
        return BossDefense(ROOT, training, lambda *args: None, lambda message: None), training

    def test_only_rank_39_counts_as_the_final_boss(self):
        self.assertTrue(is_final_boss(FINAL_BOSS_RANK))
        for rank in (None, 0, 30, 38, 40, '39', 39.0, True):
            self.assertFalse(is_final_boss(rank), rank)

    def test_both_attacks_are_forced_on_during_the_final_boss(self):
        for options in ({'boss_interrupt': False, 'battle_tap': False},
                        {'boss_interrupt': True, 'battle_tap': False},
                        {'boss_interrupt': False, 'battle_tap': True}):
            view = options_for(TrainingStub(dict(options)), 39, encounter=True)
            self.assertTrue(view['boss_interrupt'], options)
            self.assertTrue(view['battle_tap'], options)

    def test_other_ranks_keep_the_configured_values(self):
        """别处完全听配置的:这两个攻击只在打 40 BOSS 时被强制打开。

        本机配置 ``battle_tap`` 默认关,所以普通关卡不会点屏幕;这里同时验证
        "配置开着就照配置来",避免把用户的设置悄悄改掉。
        """
        off = {'boss_interrupt': False, 'battle_tap': False}
        training = TrainingStub(off)
        for rank in (0, 12, 38, 40, None):
            view = options_for(training, rank, encounter=True)
            self.assertIs(view, off, rank)
            self.assertFalse(view['boss_interrupt'], rank)
            self.assertFalse(view['battle_tap'], rank)
        on = {'boss_interrupt': True, 'battle_tap': True}
        kept = options_for(TrainingStub(on), 38)
        self.assertIs(kept, on)
        self.assertTrue(kept['battle_tap'])

    def test_world_resolves_only_the_current_sea_god_frame_without_changing_config(self):
        world = make_world(self.temp.name, 38)
        # 38 关:完全听配置(boss_interrupt 默认开、battle_tap 默认关)
        self.assertTrue(world.resolved_attack_options()['boss_interrupt'])
        self.assertFalse(world.resolved_attack_options()['battle_tap'])
        self.assertEqual(world.final_boss_attack_forced(), (False, False))
        self.assertFalse(world.final_boss_attack_enabled())
        world.rank = 39
        sea_god = frame('native-right-charge-low')
        ordinary = frame('task-ordinary-fish')
        self.assertFalse(world.final_boss_attack_enabled())
        self.assertFalse(world.final_boss_attack_enabled(ordinary))
        self.assertEqual(world.final_boss_attack_forced(ordinary), (False, False))
        self.assertTrue(world.final_boss_attack_enabled(sea_god))
        self.assertEqual(world.final_boss_attack_forced(sea_god), (False, True))
        with world.attack_options(sea_god) as attack:
            self.assertTrue(attack['boss_interrupt'])
            self.assertTrue(attack['battle_tap'])
            self.assertFalse(world.defense.training.options['battle_tap'], "不临时覆盖配置")
            self.assertTrue(world.defense.damage_battle(sea_god, 39))
            self.assertFalse(world.defense.damage_battle(ordinary, 39), "新帧不能继承海神开关")
        self.assertFalse(world.training.options['battle_tap'], "原配置保持不变")

    def test_switches_are_restored_even_when_the_body_raises(self):
        world = make_world(self.temp.name, 39)
        with self.assertRaises(RuntimeError):
            with world.attack_options(frame('native-right-charge-low')) as options:
                self.assertTrue(options['battle_tap'])
                self.assertFalse(world.training.options['battle_tap'])
                raise RuntimeError('内部出错')
        self.assertFalse(world.training.options['battle_tap'], "异常也必须还原")

    def test_trainer_defaults_are_untouched(self):
        options = {'boss_interrupt': False, 'battle_tap': False}
        training = TrainingStub(options)
        options_for(training, 39)
        self.assertEqual(training.options, {'boss_interrupt': False, 'battle_tap': False})

    def test_rank_39_without_a_new_encounter_keeps_the_configured_switches(self):
        training = TrainingStub({'boss_interrupt': False, 'battle_tap': False})
        for encounter in (False, None, 1, 'sea-god'):
            self.assertIs(options_for(training, 39, encounter=encounter), training.options)
        self.assertIs(options_for(training, 39), training.options)

    def test_ordinary_task_fish_never_starts_forced_attacks(self):
        world = make_world(self.temp.name, 39, {'boss_interrupt': False, 'battle_tap': False})
        ordinary = frame('task-ordinary-fish')
        self.assertFalse(world.defense.encounter(ordinary, 39))
        self.assertFalse(world.defense.damage_battle(ordinary, 39))
        self.assertFalse(world.defense.step(ordinary, 39))
        with world.attack_options(ordinary), \
             patch.object(world.defense, 'action', side_effect=AssertionError('ordinary fish input')):
            self.assertFalse(world.defense.damage_battle(ordinary, 39))
            self.assertFalse(world.defense.step(ordinary, 39))

    def test_configured_ordinary_battle_taps_are_preserved(self):
        world = make_world(self.temp.name, 39, {'boss_interrupt': False, 'battle_tap': True})
        ordinary = frame('task-ordinary-fish')
        self.assertTrue(world.defense.damage_battle(ordinary, 39))

    def test_burst_rechecks_forced_switches_on_every_new_capture(self):
        world = make_world(self.temp.name, 39, {'boss_interrupt': False, 'battle_tap': False})
        defense = world.defense
        sea_god = frame('native-right-charge-low')
        ordinary = frame('task-ordinary-fish')
        hits = []
        # A fixed mocked context makes this specifically exercise per-frame
        # encounter gating, rather than passing only via the title-change guard.
        for frames, expected in (([ordinary], 0), ([sea_god, ordinary], 1)):
            captures = iter(frames)
            with world.attack_options(sea_god), \
                 patch.object(defense, '_combat_context', return_value='unchanged'):
                count = defense.burst(lambda: next(captures), lambda target: hits.append(target),
                                      39, lambda: False, max_frames=len(frames))
            self.assertEqual(count, expected)
        self.assertEqual(len(hits), 1)
        self.assertEqual(world.training.options['battle_tap'], False)

    def test_guardian_forces_only_damage_without_changing_saved_options(self):
        for configured_interrupt in (False,True):
            original={'boss_interrupt':configured_interrupt,'battle_tap':False}
            training=TrainingStub(original)
            view=options_for(training,38,guardian=True)
            self.assertTrue(view['battle_tap'])
            self.assertEqual(view['boss_interrupt'],configured_interrupt)
            self.assertEqual(forced_attacks(training,38,guardian=True),(False,True))
            self.assertFalse(original['battle_tap'])
        for rank in (None,37,39,40,'38',38.0,True):
            self.assertIs(options_for(training,rank,guardian=True),original)
        for guardian in (False,None,1,'guardian'):
            self.assertIs(options_for(training,38,guardian=guardian),original)

    def test_three_real_guardian_frames_only_authorize_neutral_damage(self):
        world=make_world(self.temp.name,38,{'boss_interrupt':False,'battle_tap':False})
        for name in ('guardian-battle-0','guardian-battle-1','guardian-battle-2'):
            picture=frame(name)
            with self.subTest(name=name),world.attack_options(picture) as options:
                self.assertTrue(options['battle_tap'])
                self.assertFalse(options['boss_interrupt'])
                self.assertEqual(world.defense.encounter_rank(picture),38)
                self.assertTrue(world.defense.guardian_encounter(picture,38))
                self.assertFalse(world.defense.owns(picture,38))
                self.assertIsNone(world.defense.locate(picture,38))
                self.assertTrue(world.defense.damage_battle(picture,38))
                self.assertEqual(world.final_boss_attack_forced(picture),(False,True))
        self.assertFalse(world.training.options['battle_tap'])

    def test_guardian_guard_requires_current_name_rank_world_skull_and_full_battle(self):
        world=make_world(self.temp.name,38,{'boss_interrupt':False,'battle_tap':False})
        original=frame('guardian-battle-0')
        boxes=((65,9,139,63),(8,40,46,80),(248,32,295,81),(187,168,254,199),
               (188,125,610,165),(153,205,192,245),(307,962,415,1057),
               (208,1137,269,1192),(539,1137,600,1186),(276,881,310,913))
        candidates=[original//2,original[:640],frame('task-ordinary-fish'),
                    frame('native-right-charge-low')]
        for x,y,X,Y in boxes:
            invalid=original.copy();invalid[y:Y,x:X]=0;candidates.append(invalid)
        # Same task, RANK and world, but an ordinary fish title: no attack.
        ordinary=original.copy()
        ordinary[168:199,187:330]=frame('task-ordinary-fish')[168:199,187:330]
        candidates.append(ordinary)
        with world.attack_options(original):
            for i,picture in enumerate(candidates):
                self.assertFalse(world.defense.damage_battle(picture,38),i)
            for rank in (None,37,39,40,'38',38.0):
                self.assertFalse(world.defense.damage_battle(original,rank),rank)

    def test_guardian_burst_rechecks_new_frame_and_never_searches_tentacles(self):
        world=make_world(self.temp.name,38,{'boss_interrupt':False,'battle_tap':False})
        original=frame('guardian-battle-0')
        ordinary=original.copy()
        ordinary[168:199,187:330]=frame('task-ordinary-fish')[168:199,187:330]
        for pictures,expected in (([ordinary],0),([original,ordinary],1),
                                  ([original,frame('native-right-charge-low')],1)):
            captures=iter(pictures);hits=[];world.defense.next_damage_at=0
            with world.attack_options(original), \
                 patch.object(world.defense,'_combat_context',return_value='same-context'), \
                 patch.object(world.defense,'_locate_core',side_effect=AssertionError('no tentacle detection')):
                count=world.defense.burst(lambda:next(captures),hits.append,38,
                                         lambda:False,max_frames=len(pictures))
            self.assertEqual(count,expected)
            self.assertTrue(all(t==world.defense.damage_target() for t in hits))
            self.assertEqual(world.defense.last_burst['defense_taps'],0)

    def test_guardian_burst_keeps_damage_interval_and_total_limits(self):
        world=make_world(self.temp.name,38,{'boss_interrupt':False,'battle_tap':False})
        picture=frame('guardian-battle-0');clock=[100.];hits=[];captures=[]
        capture_seconds=[.05]
        def capture():
            clock[0]+=capture_seconds[0];captures.append(clock[0]);return picture.copy()
        def hit(target):hits.append((clock[0],target))
        for seconds in (.05,.2):
            clock[0]=100.;capture_seconds[0]=seconds;hits.clear();captures.clear()
            world.defense.next_damage_at=0
            with world.attack_options(picture),patch('drift.tasks.boss_defense.time.monotonic',side_effect=lambda:clock[0]):
                count=world.defense.burst(capture,hit,38,lambda:False)
            self.assertGreater(count,0)
            self.assertLessEqual(count,12)
            if seconds==.2:self.assertEqual(count,12)
            self.assertLessEqual(len(captures),40)
            self.assertTrue(all(b[0]-a[0]>=.15 for a,b in zip(hits,hits[1:])))
            self.assertTrue(all(h[1]['kind']=='damage' for h in hits))
        with world.attack_options(picture):
            self.assertEqual(world.defense.burst(lambda:picture,hits.append,38,lambda:True),0)


if __name__ == "__main__":
    unittest.main()
