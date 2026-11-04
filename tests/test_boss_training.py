"""Fishing failures govern ordinary quests; the final boss trains to its own target."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Clicker
from boss_training import BossTraining, training_options
from digit_ocr import DigitReader, NumberReading
from parallel_world import ParallelWorld

ROOT = Path(__file__).resolve().parents[1]


def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/training'/f'{name}.png').convert('RGB'))


class Reader:
    def read(self, _):
        return NumberReading('34', 34, .999)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'training.json'
        self.events = []
        self.training = BossTraining(ROOT, Reader(), self.events.append, state_path=self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_level_ocr_excludes_the_xp_bar_and_accepts_changing_levels(self):
        self.training.reader = DigitReader(ROOT)
        for name, expected in (('boss-main', 650), ('fishing-main', 664), ('fishing-691', 691), ('failed', 746)):
            with self.subTest(name=name):
                self.training.readings.clear()
                self.assertIsNone(self.training.levels(frame(name)))
                self.assertEqual(self.training.levels(frame(name))[0], expected)

    def test_level_gap_and_elapsed_time_alone_never_start_training(self):
        t = self.training
        with patch.object(t, 'levels', return_value=(650, 868)):
            for _ in range(6):
                self.assertIsNone(t.evaluate(frame('boss-main'), 33))
            t.state = {'mode': 'trial', 'rank': 33, 'start_level': 650,
                       'boss_level': 868, 'started_at': time.time()-600}
            self.assertIsNone(t.evaluate(frame('boss-main'), 33))
        self.assertEqual(t.state['mode'], 'idle')
        self.assertEqual(t.failures, 0)

    def test_actual_failure_cross_is_distinct_from_battle_weapons_and_success(self):
        t = self.training
        for name in ('failed', 'failed-faded', 'failed-boss', 'failed-rank35'):
            self.assertIsNotNone(t._match(frame(name), 'failure-cross', (290, 955, 432, 1060), .93))
        for name in ('battle', 'waiting', 'success', 'boss-main', 'fishing-main',
                     'battle-rank35', 'waiting-rank35', 'success-rank35'):
            self.assertIsNone(t._match(frame(name), 'failure-cross', (290, 955, 432, 1060), .93))
        self.assertIsNotNone(t._match(frame('success'), 'success-check', (290, 955, 432, 1060), .92))

    def test_one_visible_failure_counts_once_and_success_resets_the_streak(self):
        t = self.training
        with patch.object(t, 'boss_mode', return_value=True):
            for _ in range(5):
                t.observe_outcome(frame('failed'), 34)
            self.assertEqual(t.failures, 1)
            t.observe_outcome(frame('waiting'), 34)
            t.observe_outcome(frame('failed'), 34)
            self.assertEqual(t.failures, 2)
            t.observe_outcome(frame('success'), 34)
            self.assertEqual(t.failures, 0)

    def test_actual_boss_failure_with_a_holder_bar_starts_training(self):
        t = self.training
        t.reader = DigitReader(ROOT)
        self.assertTrue(t.boss_mode(frame('failed-boss')))
        for _ in range(3):
            self.assertIsNone(t.evaluate(frame('failed-boss'), 34))
        self.assertEqual(t.failures, 1)
        t.evaluate(frame('waiting-boss'), 34)
        self.assertEqual(t.evaluate(frame('failed-boss'), 34), 'train')
        self.assertEqual(t.state['mode'], 'farming')
        self.assertEqual(t.state['start_level'], 817)
        self.assertEqual(t.state['boss_level'], 983)

    def test_current_rank35_failures_with_a_holder_bar_start_training(self):
        t = self.training
        t.reader = DigitReader(ROOT)
        for _ in range(3):
            self.assertIsNone(t.evaluate(frame('failed-rank35'), 35))
        self.assertEqual(t.failures, 1)
        t.evaluate(frame('waiting-rank35'), 35)
        self.assertEqual(t.evaluate(frame('failed-rank35'), 35), 'train')
        self.assertEqual(t.state['start_level'], 852)
        self.assertEqual(t.state['boss_level'], 1112)

    def test_minutes_of_other_workflows_do_not_erase_confirmed_failures(self):
        t = self.training
        with patch('boss_training.time.monotonic', return_value=100):
            t.observe_outcome(frame('failed-rank35'), 35)
        self.assertEqual(t.failures, 1)
        with patch('boss_training.time.monotonic', return_value=1000):
            t.observe_outcome(frame('waiting-rank35'), 35)
            self.assertEqual(t.failures, 1)
            t.observe_outcome(frame('failed-rank35'), 35)
        self.assertEqual(t.failures, 2)
        t.observe_outcome(frame('waiting-rank35'), 36)
        self.assertEqual(t.failures, 0)

    def test_passive_outcome_observation_is_guarded_and_never_clicks(self):
        actions = []
        world = ParallelWorld(ROOT, lambda *a: actions.append(a), lambda *a: actions.append(a),
                              self.events.append, Reader(), training_state_path=self.path)
        world.phase, world.rank = 'play', 35
        world.next_action_at = time.monotonic()+60
        world.observe(frame('failed-rank35'))
        world.observe(frame('failed-rank35'))
        self.assertEqual(world.training.failures, 1)
        world.observe(frame('boss-map'))
        world.observe((frame('failed-rank35')*.5).astype('uint8'))
        self.assertEqual(world.training.failures, 1)
        world.observe(frame('waiting-rank35'))
        world.phase = 'find_boss'
        world.observe(frame('failed-rank35'))
        self.assertEqual(world.training.failures, 1)
        world.phase, world.rank = 'play', 40
        world.observe(frame('failed-rank35'))
        self.assertEqual(world.training.failures, 1)
        world.rank = 35
        world.observe(frame('failed-rank35'))
        self.assertEqual(world.training.failures, 2)
        self.assertEqual(actions, [])

    def test_clicker_counts_brief_outcomes_while_equipment_owns_the_ui(self):
        actions = []
        world = ParallelWorld(ROOT, lambda *a: actions.append(a), lambda *a: actions.append(a),
                              self.events.append, Reader(), training_state_path=self.path)
        world.phase, world.rank = 'play', 35
        world.next_action_at = time.monotonic()+60
        equipment = SimpleNamespace(active=True, cycles=0, phase='await_main',
                                    step=lambda _: True, status=lambda: {'phase': 'await_main'},
                                    panel=lambda _: False)
        frames = [frame(n) for n in ('failed-rank35', 'failed-rank35', 'waiting-rank35',
                                     'failed-rank35', 'failed-rank35')]

        class Device:
            def __init__(self, _, stop):
                self.stop, self.index = stop, 0
                self.io_lock = threading.RLock()
                self.image = None

            def screenshot(self):
                if self.index == len(frames):
                    self.stop.set()
                self.image = frames[min(self.index, len(frames)-1)].copy()
                self.index += 1
                return self.image

            def stuck_record_add(self, _):
                pass

        with tempfile.TemporaryDirectory(prefix='brief-outcome-replay-') as directory:
            root = Path(directory)
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules'] = [deepcopy(r) for r in config['rules']
                               if r['id'] in ('driftinggirl_parallel_world', 'driftinggirl_equipment_upgrade')]
            config['poll_interval'] = .1
            for rule in config['rules']:
                rule['enabled'] = True
                (root/rule['asset']).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/rule['asset'], root/rule['asset'])
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            app.digit_reader = Reader()
            try:
                with patch('app.ParallelWorld', return_value=world), \
                     patch('app.EquipmentUpgrade', return_value=equipment):
                    app.start(preview=False)
                    app.thread.join(10)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(world.training.failures, 2)
                self.assertEqual(app.state['parallel']['training']['failures'], 2)
                self.assertEqual(actions, [])
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_ordinary_failures_after_quest_success_do_not_block_the_next_boss(self):
        t = self.training
        self.assertFalse(t.boss_mode(frame('failed')))
        t.observe_outcome(frame('failed'), 34)
        t.observe_outcome(frame('waiting'), 34)
        t.observe_outcome(frame('failed'), 34)
        self.assertEqual(t.failures, 0)
        t.failures = 1
        t.observe_outcome(frame('waiting'), 35)
        self.assertEqual(t.failures, 0)

    def test_dimmed_or_obscured_outcomes_never_count(self):
        with patch.object(self.training, 'boss_mode', return_value=True):
            self.training.observe_outcome((frame('failed')*.5).astype('uint8'), 34)
        self.assertEqual(self.training.failures, 0)
        self.assertIsNone(self.training.levels(frame('boss-map')))

    def test_map_chooses_a_red_fishing_pin_away_from_player_and_boss(self):
        for name, boss in (('boss-map', None), ('fishing-map', (377, 1009))):
            with self.subTest(name=name):
                self.assertEqual(self.training.training_target(frame(name), boss), (317, 727))
        self.assertIsNone(self.training.training_target(frame('boss-main')))

    def test_rank38_map_ignores_changed_labels_and_terrain(self):
        image = frame('training-map-rank38')
        self.assertEqual(self.training.training_target(image), (556, 462))
        self.assertEqual(self.training.training_target(image, (358, 432)), (556, 462))
        # A close quest marker must still exclude an otherwise valid red pin.
        self.assertEqual(self.training.training_target(image, (550, 460)), (571, 988))

    def test_red_pin_fallback_needs_shape_color_and_a_current_location(self):
        image = frame('training-map-rank38').copy()
        image[380:445, 310:380] = 0  # Hide the player and its overlapping boss.
        self.assertIsNone(self.training.training_target(image))
        self.assertEqual(self.training.training_target(image, (358, 432)), (556, 462))
        blank = np.zeros_like(image)
        blank[449:478, 540:573] = (220, 60, 20)  # Red rectangle is not a pin.
        self.assertIsNone(self.training.training_target(blank, (358, 432)))
        self.assertIsNone(self.training.training_target(frame('boss-main'), (358, 432)))

    def test_missing_training_pin_preserves_farming_and_never_retries_early(self):
        actions = []
        world = ParallelWorld(ROOT, lambda *a: actions.append(a), lambda *a: actions.append(a),
                              self.events.append, Reader(), training_state_path=self.path)
        world.phase, world.rank, world.map_swipes = 'find_training', 34, 4
        world.training.begin(34, 650, 868)
        saved = self.path.read_bytes()
        with patch.object(world.training, 'training_target', return_value=None):
            with self.assertRaisesRegex(RuntimeError, '已保留练级进度'):
                world.step(frame('boss-map'))
        self.assertEqual(world.training.state['mode'], 'farming')
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(actions, [])
        restarted = BossTraining(ROOT, Reader(), self.events.append, state_path=self.path)
        with patch.object(restarted, 'levels', return_value=(847, None)):
            self.assertEqual(restarted.evaluate(frame('fishing-main'), 34), 'farming')

    def test_restart_preserves_farming_and_does_not_react_to_fish_level_changes(self):
        t = self.training
        t.begin(33, 650, 868)
        restarted = BossTraining(ROOT, Reader(), self.events.append, state_path=self.path)
        self.assertEqual(restarted.status()['boss_level'], 868)
        with patch.object(restarted, 'levels', return_value=(664, 840)):
            self.assertEqual(restarted.evaluate(frame('fishing-main'), 33), 'farming')
        self.assertEqual(restarted.state['start_level'], 650)
        self.assertEqual(restarted.state['boss_level'], 868)
        self.assertEqual(json.loads(self.path.read_text())['mode'], 'farming')

    def test_gain_and_elapsed_time_do_not_return_until_near_boss(self):
        t = self.training
        t.begin(33, 650, 868)
        with patch.object(t, 'levels', return_value=(670, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'farming')
        t.begin(33, 650, 868)
        t.state['started_at'] -= 601
        with patch.object(t, 'levels', return_value=(650, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'farming')
        with patch.object(t, 'levels', return_value=(847, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'farming')
        with patch.object(t, 'levels', return_value=(848, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'retry')
        t.begin(33, 650, 868)
        t.options['enabled'] = False
        with patch.object(t, 'levels', return_value=(660, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'retry')
        self.assertEqual(t.state['mode'], 'idle')

    def test_unknown_boss_waits_and_exact_level_option_is_supported(self):
        t = self.training
        t.begin(33, 650, None)
        t.state['started_at'] -= 36000
        with patch.object(t, 'levels', return_value=(900, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'farming')
        t.begin(33, 650, 868)
        t.options['retry_level_gap'] = 0
        with patch.object(t, 'levels', return_value=(867, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'farming')
        with patch.object(t, 'levels', return_value=(868, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 33), 'retry')

    def test_real_battle_header_survives_the_gold_meter_covering_the_range(self):
        actions = []
        world = ParallelWorld(ROOT,lambda *a:actions.append(a),lambda *a:actions.append(a),
                              self.events.append,DigitReader(ROOT),training_state_path=self.path,
                              defense_action=lambda *a:actions.append(a))
        world.phase,world.rank = 'play',39
        t = world.training
        # These native frames really cover the former bottom-label OCR region.
        self.assertIsNone(t._number(frame('sea-god-meter-high'),
                                     ((238,929,285,952),(241,930,281,950))))
        world.observe(frame('sea-god-meter-high'))
        self.assertIsNone(t._cached_boss(39))
        world.observe(frame('sea-god-meter-high-2'))
        self.assertEqual(t._cached_boss(39),1810)
        self.assertEqual(world.target_rank,39)
        self.assertEqual(actions,[])
        self.assertFalse(world.completed_rank_40)
        self.assertEqual(world.rounds,0)
        t.begin(39,1827,None)
        self.assertEqual(t.state['boss_level'],1810)
        self.assertEqual(t.status()['retry_level'],1810)

    def test_header_cache_is_saved_during_idle_and_survives_a_restart(self):
        t = self.training
        t.reader = DigitReader(ROOT)
        self.assertIsNone(t.observe_boss_header(frame('sea-god-meter-high'),39,selected=True))
        self.assertEqual(t.observe_boss_header(frame('sea-god-meter-high-2'),39,selected=True),1810)
        self.assertEqual(t.state['mode'],'idle')
        restarted = BossTraining(ROOT,Reader(),self.events.append,state_path=self.path)
        restarted.begin(39,1809,None)
        self.assertEqual(restarted.state['boss_level'],1810)
        with patch.object(restarted,'levels',return_value=(1809,None)):
            self.assertEqual(restarted.evaluate(frame('fishing-main'),39),'farming')
        with patch.object(restarted,'levels',return_value=(1810,None)):
            self.assertEqual(restarted.evaluate(frame('fishing-main'),39),'retry')

    def test_unknown_farming_target_is_filled_without_resetting_progress_or_sailing(self):
        t = self.training
        t.reader = DigitReader(ROOT)
        t.begin(39,1780,None)
        t.state.update(sailing_enabled=True,sailing_verified=True,
                       fishing_meter_limits={'native-context':{'low':1524,'high':1810}})
        t._save()
        before = deepcopy(t.state)
        t.observe_boss_header(frame('sea-god-meter-high'),39,selected=True)
        t.observe_boss_header(frame('sea-god-meter-high-2'),39,selected=True)
        self.assertEqual(t.state['boss_level'],1810)
        for key,value in before.items():
            if key != 'boss_level':
                self.assertEqual(t.state[key],value)

    def test_another_rank_and_disabled_tapping_cannot_poison_the_cached_quest(self):
        actions = []
        world = ParallelWorld(ROOT,lambda *a:actions.append(a),lambda *a:actions.append(a),
                              self.events.append,DigitReader(ROOT),training_state_path=self.path,
                              defense_action=lambda *a:actions.append(a))
        world.phase,world.rank = 'play',39
        world.training.options['boss_interrupt'] = False
        for name in ('sea-god-meter-high','sea-god-meter-high-2'):
            world.observe(frame(name))
        self.assertEqual(world.training._cached_boss(39),1810)
        self.assertFalse(world.defense.owns(frame('sea-god-meter-high'),39))
        self.assertEqual(actions,[])
        world.training.observe_outcome(frame('fishing-main'),38)
        world.training.begin(38,1800,None)
        self.assertEqual(world.training.state['boss_level'],0)
        self.assertNotIn('quest_boss_level',world.training.state)

    def test_fresh_charging_core_defers_header_ocr_but_not_outcome_observation(self):
        world = ParallelWorld(ROOT,lambda *a:None,lambda *a:None,self.events.append,
                              Reader(),training_state_path=self.path,
                              defense_action=lambda *a:None)
        world.phase,world.rank = 'play',39
        charging = np.asarray(Image.open(ROOT/'tests/fixtures/boss-defense/native-right-charge-low.png').convert('RGB'))
        idle = np.asarray(Image.open(ROOT/'tests/fixtures/boss-defense/ship-fire.png').convert('RGB'))
        world.training.boss_header_readings.append((39,1810))
        with patch.object(world.training,'observe_outcome') as outcome, \
             patch.object(world.training,'observe_boss_header') as header:
            world.observe(charging)
            outcome.assert_called_once_with(charging,39)
            header.assert_not_called()
            self.assertEqual(len(world.training.boss_header_readings),0)
            world.observe(idle)
            header.assert_called_once_with(idle,39,selected=True)

    def test_disabled_counter_still_reads_a_charging_boss_level(self):
        world = ParallelWorld(ROOT,lambda *a:None,lambda *a:None,self.events.append,
                              Reader(),training_state_path=self.path,
                              defense_action=lambda *a:None)
        world.phase,world.rank = 'play',39
        world.training.options['boss_interrupt'] = False
        charging = np.asarray(Image.open(ROOT/'tests/fixtures/boss-defense/native-right-charge-low.png').convert('RGB'))
        with patch.object(world.training,'observe_boss_header') as header:
            world.observe(charging)
            header.assert_called_once_with(charging,39,selected=True)

    def test_unreadable_boss_stays_at_the_quest_with_its_failure_count(self):
        t = self.training
        t.boss_level,t.boss_rank = 1000,38
        t.failures,t.outcome_rank = 2,39
        with patch.object(t,'levels',return_value=(1827,None)), \
             patch.object(t,'boss_mode',return_value=True), \
             patch.object(t,'observe_outcome'):
            self.assertIsNone(t.evaluate(frame('boss-main'),39))
        self.assertEqual(t.state['mode'],'idle')
        self.assertEqual(t.failures,2)
        self.assertFalse(self.path.exists())

    def test_ordinary_dark_and_unselected_frames_never_save_a_header_target(self):
        world = ParallelWorld(ROOT,lambda *a:None,lambda *a:None,self.events.append,
                              DigitReader(ROOT),training_state_path=self.path,
                              defense_action=lambda *a:None)
        world.phase,world.rank = 'play',39
        ordinary = np.asarray(Image.open(ROOT/'tests/fixtures/boss-defense/task-ordinary-fish.png').convert('RGB'))
        for sample in (ordinary,(frame('sea-god-meter-high')*.5).astype('uint8')):
            for _ in range(3):
                world.observe(sample)
        self.assertIsNone(world.training._cached_boss(39))
        self.assertIsNone(world.training.observe_boss_header(frame('sea-god-meter-high'),39))
        world.rank = 38
        for _ in range(3):
            world.observe(frame('sea-god-meter-high'))
        self.assertFalse(self.path.exists())

    def test_battle_header_disagreement_or_missing_prefix_is_not_a_target(self):
        t = self.training
        for readings in ((NumberReading('Lv.1810',None,.999),NumberReading('Lv.1820',None,.999)),
                         (NumberReading('1810',1810,.999),NumberReading('1810',1810,.999))):
            with patch.object(t.reader,'read',side_effect=readings):
                self.assertIsNone(t.observe_boss_header(frame('sea-god-meter-high'),39,selected=True))
        self.assertFalse(self.path.exists())

    def test_final_boss_trains_before_more_attempts_and_waits_for_its_level(self):
        t = self.training
        with patch.object(t, 'levels', return_value=(1622, 1810)), patch.object(t, 'boss_mode', return_value=True):
            self.assertEqual(t.evaluate(frame('boss-main'), 39), 'train')
        self.assertEqual(t.failures, 0)
        self.assertEqual(t.state['mode'], 'farming')
        self.assertEqual(t.status()['retry_level'], 1810)
        self.assertTrue(t.status()['final_boss'])
        self.assertEqual(t.options['retry_level_gap'], 20)
        started = t.state['started_at']
        with patch.object(t, 'levels', return_value=(1790, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 39), 'farming')
        with patch.object(t, 'levels', return_value=(1809, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 39), 'farming')
        self.assertEqual(t.state['started_at'], started)
        with patch.object(t, 'levels', return_value=(1810, None)):
            self.assertEqual(t.evaluate(frame('fishing-main'), 39), 'retry')

    def test_new_final_boss_goal_applies_to_saved_training_without_resetting_progress(self):
        old = {**training_options(), 'final_boss_retry_level_gap': 20}
        t = BossTraining(ROOT, Reader(), self.events.append, old, state_path=self.path)
        t.begin(39, 1622, 1810)
        saved = json.loads(self.path.read_text())
        restarted = BossTraining(ROOT, Reader(), self.events.append, state_path=self.path)
        with patch.object(restarted, 'levels', return_value=(1790, 1604)):
            self.assertEqual(restarted.evaluate(frame('fishing-main'), 39), 'farming')
        self.assertEqual(restarted.state, saved)
        self.assertEqual(restarted.status()['retry_level'], 1810)
        self.assertEqual(restarted.state['boss_level'], 1810)

    def test_final_boss_preparation_requires_a_known_boss_and_enabled_training(self):
        t = self.training
        with patch.object(t, 'boss_mode', return_value=True):
            with patch.object(t, 'levels', return_value=(1622, None)):
                self.assertIsNone(t.evaluate(frame('boss-main'), 39))
            t.options['enabled'] = False
            with patch.object(t, 'levels', return_value=(1622, 1810)):
                self.assertIsNone(t.evaluate(frame('boss-main'), 39))
            t.options['enabled'] = True
            with patch.object(t, 'levels', return_value=(1810, 1810)):
                self.assertIsNone(t.evaluate(frame('boss-main'), 39))
            with patch.object(t, 'levels', return_value=(1622, 1810)):
                self.assertIsNone(t.evaluate(frame('boss-main'), 40))

    def test_long_training_restart_keeps_the_original_boss_target(self):
        t = self.training
        t.begin(33, 650, 868)
        t.state['started_at'] -= 90000
        t._save()
        restarted = BossTraining(ROOT, Reader(), self.events.append, state_path=self.path)
        self.assertEqual(restarted.state['mode'], 'farming')
        self.assertEqual(restarted.status()['retry_level'], 848)
        with patch.object(restarted, 'levels', return_value=(840, 700)):
            self.assertEqual(restarted.evaluate(frame('fishing-main'), 33), 'farming')

    def test_invalid_state_and_parameters_are_rejected(self):
        self.path.write_text('{invalid')
        self.assertEqual(BossTraining(ROOT, Reader(), self.events.append, state_path=self.path).state['mode'], 'idle')
        for value in ({'enabled': 'yes'}, {'failure_limit': 0}, {'failure_limit': True},
                      {'minutes': 1.5}, {'level_gain': 201}, {'level_gap': 80},
                      {'retry_level_gap': -1}, {'retry_level_gap': True}, {'retry_level_gap': 201},
                      {'final_boss_retry_level_gap': -1}, {'final_boss_retry_level_gap': True},
                      {'final_boss_retry_level_gap': 201}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                training_options(value)

    def test_controller_moves_only_after_actual_failures_and_keeps_farming_unowned(self):
        actions = []
        world = ParallelWorld(ROOT, lambda *a: actions.append(a), lambda *a: actions.append(a),
                              self.events.append, Reader(), training_state_path=self.path)
        world.phase, world.rank, world.target_rank = 'play', 34, 34
        t = world.training

        def step(name):
            world.next_action_at = 0
            return world.step(frame(name))

        with patch.object(t, 'levels', return_value=(650, 868)), patch.object(t, 'boss_mode', return_value=True):
            self.assertFalse(step('boss-main'))
            self.assertFalse(step('failed'))
            self.assertFalse(step('failed'))
            self.assertFalse(step('waiting'))
            self.assertTrue(step('failed'))
        self.assertEqual(world.phase, 'find_training')
        self.assertEqual(actions[-1][:2], (350, 30))
        step('boss-map')
        self.assertEqual(actions[-1][:2], (317, 727))
        step('boss-map')
        self.assertEqual(actions[-1][:2], (668, 1187))
        count = len(actions)
        with patch.object(t, 'levels', return_value=(664, None)):
            self.assertFalse(step('fishing-main'))
        self.assertEqual(len(actions), count)
        with patch.object(t, 'levels', return_value=(848, None)):
            self.assertTrue(step('fishing-691'))
        self.assertEqual(world.phase, 'find_boss')
        step('fishing-map')
        self.assertEqual(actions[-1][:2], (377, 1009))

    def test_rank_40_preempts_training_and_still_needs_the_confirmed_guard(self):
        class Rank40Reader:
            def read(self, _):
                return NumberReading('40', 40, .999)
        actions = []
        world = ParallelWorld(ROOT, lambda *a: actions.append(a), lambda *a: None,
                              self.events.append, Rank40Reader(), training_state_path=self.path)
        world.phase = 'find_training'
        world.training.begin(34, 650, 868)
        world.rank = 34
        for _ in range(3):
            world.next_action_at = 0
            world.step(frame('boss-map'))
        self.assertTrue(world.completed_rank_40)
        self.assertEqual(world.training.state['mode'], 'idle')
        self.assertEqual(actions[-1][:2], (668, 1187))
        self.assertFalse(any('次元门' in a[-1] for a in actions))


if __name__ == '__main__':
    unittest.main()
