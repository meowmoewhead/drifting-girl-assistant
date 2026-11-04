"""Check the real game frames and the irreversible Rank 40 guard."""
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import cv2
from PIL import Image

from digit_ocr import DigitReader, NumberReading
from parallel_world import ParallelWorld
from boss_defense import BossDefense


ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / 'tests' / 'fixtures' / 'parallel'


def frame(name):
    return np.asarray(Image.open(FRAMES / (name + '.png')).convert('RGB'))


class Reader:
    def __init__(self, value):
        self.value = value

    def read(self, _):
        return NumberReading(str(self.value), self.value, .999)


class ParallelWorldTests(unittest.TestCase):
    def test_worker_completion_template_missing_blocks_initialization(self):
        with tempfile.TemporaryDirectory(prefix='missing-completion-') as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'assets',root/'assets')
            (root/'assets/recovery/parallel-complete.png').unlink()
            with self.assertRaisesRegex(RuntimeError,'通关弹窗模板不完整'):
                ParallelWorld(root,lambda *a:None,lambda *a:None,lambda *a:None,Reader(40),
                              completion_action=lambda:True)

    def test_native_completion_notice_dismissal_does_not_prove_full_cycle(self):
        w=self.world
        w.reader=DigitReader(ROOT)
        modal=frame('complete-tier8-equipment')
        self.assertTrue(w.completion_visible(modal))
        w.completion_action=lambda:self.actions.append(('completion',)) or True
        for _ in range(3):
            self.assertTrue(w.completion_step(modal))
        self.assertEqual(self.actions,[('completion',)])
        self.assertFalse(w.completed_rank_40)
        self.assertIsNone(w.rank)
        self.assertEqual(w.rounds,0)
        self.assertTrue(w.completion_step(modal))
        self.assertEqual(self.actions,[('completion',)])
        main=frame('parallel-main')
        self.assertTrue(w.completion_step(main,ready=True))
        self.assertFalse(w.completion_step(main,ready=True))

    def test_completion_requires_actual_rank_world_words_and_bright_button(self):
        w=self.world
        w.reader=Reader(40)
        modal=frame('complete-tier8-equipment')
        self.assertTrue(w.completion_visible(modal))
        for area in ((13,46,39,72),(306,591,412,619),(305,650,502,687),
                     (230,730,497,760),(306,807,419,868)):
            altered=modal.copy();x,y,X,Y=area;altered[y:Y,x:X]=0
            self.assertFalse(w.completion_visible(altered),area)
        self.assertFalse(w.completion_visible((modal*.8).astype(np.uint8)))
        self.assertFalse(w.completion_visible(frame('parallel-main')))
        w.reader=Reader(39)
        self.assertFalse(w.completion_visible(modal))

    def test_completion_fresh_callback_rejection_retains_deadline_without_input(self):
        w=self.world;w.reader=Reader(40)
        w.completion_action=lambda:False
        modal=frame('complete-tier8-equipment')
        with patch('parallel_world.time.monotonic',return_value=1):
            for _ in range(3):self.assertTrue(w.completion_step(modal))
        self.assertFalse(w.completion_clicked)
        self.assertEqual(self.actions,[])
        with patch('parallel_world.time.monotonic',return_value=17):
            with self.assertRaisesRegex(RuntimeError,'未安全消失'):
                w.completion_step(modal)

    def test_completion_occlusion_is_not_clearance_and_never_repeats_confirm(self):
        w=self.world;w.reader=Reader(40)
        w.completion_action=lambda:self.actions.append(('completion',)) or True
        modal=frame('complete-tier8-equipment')
        with patch('parallel_world.time.monotonic',return_value=1):
            for _ in range(3):w.completion_step(modal)
        with patch('parallel_world.time.monotonic',return_value=5):
            self.assertTrue(w.completion_step(np.zeros_like(modal),ready=False))
            self.assertTrue(w.completion_step(modal))
        self.assertEqual(self.actions,[('completion',)])
        self.assertFalse(w.completed_rank_40)
        self.assertEqual(w.rounds,0)

    def test_native_qualified_retry_at_selected_boss_keeps_the_map_closed(self):
        w = self.world
        w.phase,w.rank,w.target_rank = 'play',39,39
        w.reader = w.training.reader = DigitReader(ROOT)
        w.training.begin(39,1827,1810)
        sea=np.asarray(Image.open(ROOT/'tests/fixtures/training/sea-god-meter-high.png').convert('RGB'))
        for _ in range(3):
            w.next_action_at=0
            w.step(sea)
        self.assertEqual(w.training.state['mode'],'trial')
        self.assertEqual(w.training.player_level,1827)
        self.assertEqual(w.phase,'play')
        self.assertEqual(w.target_rank,39)
        self.assertEqual(self.actions,[])
        self.assertFalse(w.completed_rank_40)
        self.assertEqual(w.rounds,0)

    def test_native_qualified_retry_does_not_reuse_another_selected_rank(self):
        w = self.world
        w.phase,w.rank,w.target_rank = 'play',39,38
        w.reader = w.training.reader = DigitReader(ROOT)
        w.training.begin(39,1827,1810)
        sea=np.asarray(Image.open(ROOT/'tests/fixtures/training/sea-god-meter-high.png').convert('RGB'))
        for _ in range(2):
            w.next_action_at=0
            w.step(sea)
        self.assertEqual(w.phase,'find_boss')
        self.assertEqual(self.actions[-1][1:3],(350,30))
        self.assertFalse(w.completed_rank_40)

    def test_qualified_retry_from_an_ordinary_training_point_still_navigates(self):
        w = self.world
        w.phase,w.rank,w.target_rank = 'play',34,34
        w.training.begin(34,650,868)
        w.training.player_level=848
        ordinary=np.asarray(Image.open(ROOT/'tests/fixtures/training/fishing-main.png').convert('RGB'))
        self.assertFalse(w.training.boss_mode(ordinary))
        with patch.object(w,'_rank'),patch.object(w.training,'levels',return_value=(848,None)):
            w.step(ordinary)
        self.assertEqual(w.phase,'find_boss')
        self.assertEqual(self.actions[-1][1:3],(350,30))

    def test_current_boss_retry_waits_for_sailing_to_relinquish_control(self):
        w = self.world
        w.phase,w.rank,w.target_rank = 'play',39,39
        w.reader = w.training.reader = DigitReader(ROOT)
        w.training.begin(39,1827,1810)
        w.training.player_level=1827
        sea=np.asarray(Image.open(ROOT/'tests/fixtures/training/sea-god-meter-high.png').convert('RGB'))
        w.training.levels(sea)
        with patch.object(w,'_stop_sailing',return_value=True) as stop:
            # Keep the real shutdown dependency active while its stop guard
            # owns the UI; skipping the map cannot bypass that controller.
            from training_sailing import TrainingSailing
            w.sailing=TrainingSailing(ROOT,w.training,lambda *a:None,lambda _:None)
            w.training.set_sailing(True,True)
            self.assertTrue(w.step(sea))
        stop.assert_called_once()
        self.assertEqual(w.phase,'play')
        self.assertTrue(w.training.state['sailing_enabled'])
        self.assertEqual(self.actions,[])

    def test_native_sea_god_proves_return_without_a_next_task_arrow_or_click(self):
        w=self.world
        w.rank=39;w.phase='await_boss_refresh';w.map_swipes=24;w.map_refreshes=1
        w.defense=BossDefense(ROOT,w.training,None,lambda _:None)
        sea=np.asarray(Image.open(ROOT/'tests/fixtures/boss-defense/ship-fire.png').convert('RGB'))
        w.observe(sea)
        self.assertEqual(w.phase,'play')
        self.assertEqual(w.target_rank,39)
        self.assertEqual((w.map_swipes,w.map_refreshes),(0,0))
        self.assertFalse(w.completed_rank_40)
        self.assertEqual(w.rounds,0)
        self.assertEqual(self.actions,[])
        for altered in (sea//2,np.asarray(Image.open(ROOT/'tests/fixtures/boss-defense/task-ordinary-fish.png').convert('RGB'))):
            w.phase='await_boss_refresh';w.observe(altered)
            self.assertEqual(w.phase,'await_boss_refresh')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.actions = []
        self.reader = Reader(3)
        self.world = ParallelWorld(ROOT,
                                   lambda x, y, label: self.actions.append(('tap', x, y, label)),
                                   lambda *args: self.actions.append(('swipe', *args)),
                                   lambda _: None, self.reader,
                                   training_state_path=Path(self.temp.name)/'training_state.json')

    def tearDown(self):
        self.temp.cleanup()

    def step(self, name):
        self.world.next_action_at = 0
        return self.world.step(frame(name))

    def test_real_frames_distinguish_world_map_and_boss(self):
        p = self.world
        self.assertTrue(p._parallel(frame('parallel-main')))
        self.assertFalse(p._parallel(frame('observatory-main')))
        self.assertTrue(p._map(frame('parallel-map-rank2')))
        self.assertFalse(p._map(frame('parallel-main')))
        target = p._match(frame('parallel-map-rank2'), 'map-boss', (20, 175, 700, 1115), .88)
        self.assertEqual(target[:2], (385, 423))
        self.assertIsNone(p._match(frame('observatory-main'), 'map-boss', (20, 175, 700, 1115), .88))

    def test_rank_progression_selects_one_boss_then_waits(self):
        for _ in range(3):
            self.step('parallel-rank3-check')
        self.assertEqual(self.world.phase, 'find_boss')
        self.assertEqual(self.actions[-1][1:3], (351, 96))
        self.step('parallel-map-rank2')
        self.assertEqual(self.world.target_rank, 3)
        self.assertEqual(self.actions[-1][1:3], (385, 423))
        self.step('parallel-map-rank2')
        count = len(self.actions)
        for _ in range(4):
            self.assertFalse(self.step('parallel-rank3-check'))
        self.assertEqual(len(self.actions), count)
        self.reader.value = 2  # A bad OCR result cannot lower this world's rank.
        for _ in range(4):
            self.step('parallel-rank3-check')
        self.assertEqual(self.world.rank, 3)
        self.assertEqual(self.world.target_rank, 3)

    def test_restart_with_parallel_map_already_open_uses_its_boss_target(self):
        for _ in range(3):
            self.step('parallel-map-rank2')
        self.assertEqual(self.world.phase, 'find_boss')
        self.assertEqual(self.actions, [])
        self.step('parallel-map-rank2')
        self.assertEqual(self.actions[-1][1:3], (385, 423))

    def test_final_boss_preparation_does_not_freeze_the_previous_quest_level(self):
        w = self.world
        w.phase,w.rank,w.target_rank = 'play',39,38
        with patch.object(w,'_rank'), patch.object(w.training,'boss_mode',return_value=True):
            with patch.object(w.training,'levels',return_value=(1584,1604)):
                self.step('parallel-rank3-check')
            self.assertEqual(w.training.state['mode'],'idle')
            self.assertEqual(w.phase,'find_boss')
            w.phase,w.target_rank = 'play',39
            with patch.object(w.training,'levels',return_value=(1622,1810)):
                self.step('parallel-rank3-check')
        self.assertEqual(w.training.state['mode'],'farming')
        self.assertEqual(w.training.state['boss_level'],1810)
        self.assertEqual(w.training.status()['retry_level'],1810)

    def test_native_rank38_skull_matches_body_and_complete_icon(self):
        p = self.world
        for name,center in (('parallel-next-task-rank38',(163,428)),
                            ('parallel-error-map-rank38',(358,428))):
            a = frame(name)
            self.assertTrue(p._map(a))
            old_score = cv2.minMaxLoc(cv2.matchTemplate(a[175:1115,20:700],
                p.assets['map-boss'],cv2.TM_CCOEFF_NORMED))[1]
            self.assertLess(old_score,.88)
            target = p._match(a,'map-boss',(20,175,700,1115),.88)
            self.assertEqual(target[:2],center)
            self.assertGreater(target[2],.92)
        for name in ('parallel-missing-boss-rank38','observatory-main','android-home'):
            self.assertIsNone(p._match(frame(name),'map-boss',(20,175,700,1115),.88),name)

    def test_rank_change_during_map_search_closes_old_map_before_new_task(self):
        p = self.world
        p.phase,p.rank,p.target_rank = 'find_boss',37,37
        p.training.state = dict(mode='trial',rank=37,start_level=1331,boss_level=1421,started_at=0)
        self.reader.value = 38
        for _ in range(3):
            self.step('parallel-missing-boss-rank38')
        self.assertEqual(p.rank,38)
        self.assertEqual(p.phase,'await_boss_refresh')
        self.assertEqual(self.actions[-1][1:3],(668,1187))
        self.assertIsNone(p.target_rank)
        self.assertEqual(p.training.state['mode'],'idle')
        self.step('parallel-main-rank38')
        self.assertEqual(self.actions[-1][1:3],(351,96))
        self.assertEqual(p.phase,'find_boss')
        self.step('parallel-next-task-rank38')
        self.assertEqual(self.actions[-1][1:3],(163,428))
        self.assertEqual(p.target_rank,38)

    def test_missing_marker_refreshes_once_then_stops_without_guessing(self):
        p = self.world
        p.phase,p.rank,p.map_swipes = 'find_boss',38,24
        self.reader.value = 38
        self.step('parallel-missing-boss-rank38')
        self.assertEqual(p.phase,'refresh_boss_map')
        self.assertEqual(p.map_refreshes,1)
        self.assertEqual(self.actions,[])
        self.step('parallel-missing-boss-rank38')
        self.assertEqual(self.actions[-1][1:3],(668,1187))
        self.step('parallel-main-rank38')
        self.assertEqual(self.actions[-1][1:3],(351,96))
        self.assertEqual(p.map_swipes,0)
        p.map_swipes = 24
        count = len(self.actions)
        with self.assertRaisesRegex(RuntimeError,'重新打开任务地图后仍未找到'):
            self.step('parallel-missing-boss-rank38')
        self.assertEqual(len(self.actions),count)

    def test_rank40_preempts_both_map_refresh_phases(self):
        for phase in ('refresh_boss_map','await_boss_refresh'):
            p = self.world
            p.phase,p.rank,p.completed_rank_40 = phase,39,False
            p.rank_readings.clear()
            p.rank_readings.extend((40,40))
            self.reader.value = 40
            self.actions.clear()
            self.step('parallel-main-rank38')
            self.assertEqual(p.phase,'return_start')
            self.assertTrue(p.completed_rank_40)
            self.assertEqual(self.actions,[])

    def test_refresh_waits_for_map_close_and_ends_on_missing_arrow(self):
        p = self.world
        p.phase,p.rank,p.phase_since = 'await_boss_refresh',38,100
        self.reader.value = 38
        with patch('parallel_world.time.monotonic',return_value=120):
            self.step('parallel-missing-boss-rank38')
        self.assertEqual(self.actions,[])
        with patch('parallel_world.time.monotonic',return_value=131):
            with self.assertRaisesRegex(RuntimeError,'下一任务入口.*30 秒'):
                self.step('parallel-missing-boss-rank38')
        self.assertEqual(self.actions,[])

    def test_real_rank_five_ocr_and_earlier_ranks(self):
        self.world.reader = DigitReader(ROOT)
        for name, value in (('parallel-main', 0), ('parallel-map-rank2', 2),
                            ('parallel-rank3-check', 3), ('parallel-rank5', 5)):
            with self.subTest(name=name):
                self.world.rank = None
                self.world.rank_readings.clear()
                for _ in range(3):
                    self.world._rank(frame(name))
                self.assertEqual(self.world.rank, value)

    def test_conflicting_rank_crops_cannot_trigger_portal_close(self):
        class ConflictingReader:
            def read(self, crop):
                value = 40 if crop.shape[1] >= 50 else 5
                return NumberReading(str(value), value, .999)
        self.world.reader = ConflictingReader()
        for _ in range(3):
            self.step('parallel-rank5')
        self.assertIsNone(self.world.rank)
        self.assertFalse(self.world.completed_rank_40)
        self.assertEqual(self.actions, [])

    def test_portal_close_requires_three_confident_rank_40_readings(self):
        self.reader.value = 40
        self.step('parallel-rank3-check')
        self.step('parallel-rank3-check')
        self.assertFalse(self.world.completed_rank_40)
        self.assertEqual(self.world.phase, 'play')
        self.step('parallel-rank3-check')
        self.assertTrue(self.world.completed_rank_40)
        self.assertEqual(self.world.phase, 'return_start')
        self.step('parallel-rank3-check')
        self.assertEqual(self.world.phase, 'return_confirm')
        self.step('parallel-return-entry')
        self.assertEqual(self.world.phase, 'await_original')
        self.step('portal-active-main')
        self.step('portal-active-main')
        self.step('portal-open-confirmed')
        self.assertEqual(self.world.phase, 'close_click')
        self.step('portal-open-confirmed')
        self.assertEqual(self.world.phase, 'close_confirm')
        self.assertTrue(any('关闭传送门' in action[-1] for action in self.actions))

    def test_forced_close_phase_cannot_bypass_guard(self):
        self.world.phase = 'close_click'
        self.world.rank = 39
        self.world.completed_rank_40 = False
        with self.assertRaisesRegex(RuntimeError, '拒绝关闭传送门'):
            self.step('portal-open-confirmed')
        self.assertEqual(self.actions, [])

    def test_completed_round_reopens_and_clears_old_rank_guard(self):
        self.reader.value = 40
        for _ in range(4):
            self.step('parallel-rank3-check')
        self.step('parallel-return-entry')
        self.step('portal-active-main')
        self.step('portal-active-main')
        self.step('portal-open-confirmed')
        self.step('portal-open-confirmed')
        self.step('portal-tap')  # The close confirmation uses the shared continue button.
        self.step('portal-cat')
        self.step('portal-cat')
        self.step('portal-tap')
        self.step('portal-open-confirmed')
        self.step('portal-active-main')
        self.step('parallel-enter')
        self.step('parallel-main')
        self.assertEqual(self.world.phase, 'play')
        self.assertEqual(self.world.rounds, 1)
        self.assertFalse(self.world.completed_rank_40)
        self.assertIsNone(self.world.rank)

    def test_unknown_screen_timeout_starts_at_disappearance_after_long_play(self):
        p = self.world
        p.phase, p.phase_since, p.rank, p.target_rank = 'play', 0, 37, 37
        home = frame('android-home')  # The real 04:10 stop frame.
        with patch('parallel_world.time.monotonic', return_value=3600):
            self.assertTrue(p.step(home))
        self.assertEqual(p.missing_parallel_since, 3600)
        with patch('parallel_world.time.monotonic', return_value=3639.9):
            self.assertTrue(p.step(home))
        with patch('parallel_world.time.monotonic', return_value=3640.1):
            with self.assertRaisesRegex(RuntimeError, '平行世界主画面.*40 秒'):
                p.step(home)
        self.assertEqual(self.actions, [])
        self.assertFalse(p.completed_rank_40)

    def test_separate_screen_interruptions_do_not_accumulate(self):
        p = self.world
        p.phase, p.phase_since, p.rank, p.target_rank = 'play', 0, 37, 37
        self.reader.value = 37
        with patch('parallel_world.time.monotonic', return_value=3600):
            self.assertTrue(p.step(frame('android-home')))
        with patch('parallel_world.time.monotonic', return_value=3620):
            self.assertFalse(p.step(frame('parallel-rank3-check')))
        self.assertIsNone(p.missing_parallel_since)
        with patch('parallel_world.time.monotonic', return_value=7200):
            self.assertTrue(p.step(frame('android-home')))
        self.assertEqual(p.missing_parallel_since, 7200)
        self.assertEqual(p.phase_since, 0)
        self.assertEqual(self.actions, [])

    def test_passive_visible_world_resets_missing_time_during_other_workflows(self):
        p = self.world
        p.phase, p.rank, p.target_rank = 'play', 37, 37
        p.missing_parallel_since = 10
        p.observe(frame('android-home'))
        self.assertEqual(p.missing_parallel_since, 10)
        p.observe(frame('parallel-rank3-check'))
        self.assertIsNone(p.missing_parallel_since)
        self.assertEqual(self.actions, [])

    def test_phase_changes_clear_missing_time_without_extending_other_deadlines(self):
        p = self.world
        p.phase, p.phase_since, p.missing_parallel_since = 'play', 0, 10
        with patch('parallel_world.time.monotonic', return_value=100):
            p._set('return_confirm')
        self.assertIsNone(p.missing_parallel_since)
        self.assertEqual(p.phase_since, 100)
        with patch('parallel_world.time.monotonic', return_value=130.1):
            with self.assertRaisesRegex(RuntimeError, '返回确认.*30 秒'):
                p._timeout(30, '返回确认')
        self.assertEqual(self.actions, [])


if __name__ == '__main__':
    unittest.main()
