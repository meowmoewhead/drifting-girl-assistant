"""Replay both production crown-exit failures and guarded return paths."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import Clicker
from features import RECOVERY_RULE_IDS
from tools.configure_achievement_return import CHANGED_FIELDS, CLOSE, FOLD, RETURN, update


class AchievementReturnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {r['id']: r for r in cls.config['rules']}
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        cls.legacy = {r['id']: r for r in json.loads(
            (ROOT/'tests/fixtures/achievements-return/legacy-rules.json').read_text(encoding='utf-8'))}
        cls.frames = {name: cls.image(path) for name, path in {
            'main': 'assets/navigation/main.png',
            'menu': 'assets/achievements-menu-native.png',
            'claimable': 'assets/achievements-list-native.png',
            'empty': 'tests/fixtures/legacy-captures/achievements-cycle-result.png',
            'after-close': 'tests/fixtures/achievements-return/battle-after-close.png',
            'after-fold': 'tests/fixtures/achievements-return/battle-after-fold.png',
            'popup': 'tests/fixtures/achievements-return/task-popup.png',
        }.items()}

    @staticmethod
    def image(path):
        with Image.open(ROOT/path) as source:
            return np.asarray(source.convert('RGB'))

    def visible(self, rule, frame, use_absence=True):
        _, button, offset = Clicker.load_rule(self.context, rule)
        matched = button.match(frame, offset=offset, similarity=rule['threshold'])
        if matched and use_absence and rule.get('absent_rule'):
            guard = self.rules[rule['absent_rule']]
            matched = not self.visible(guard, frame, use_absence=False)
        return matched

    def test_both_recorded_battle_frames_reproduce_old_return_failure(self):
        for name in ('after-close', 'after-fold'):
            with self.subTest(name=name):
                self.assertFalse(self.visible(self.legacy[RETURN], self.frames[name]))
                self.assertTrue(self.visible(self.rules[RETURN], self.frames[name]))

    def test_clear_normal_game_passes_but_expanded_menu_still_requires_folding(self):
        self.assertTrue(self.visible(self.rules[RETURN], self.frames['main']))
        self.assertTrue(self.visible(self.rules[FOLD], self.frames['menu']))
        self.assertFalse(self.visible(self.rules[RETURN], self.frames['menu']))

    def test_modals_and_unfinished_rewards_never_count_as_return(self):
        for name in ('claimable', 'empty', 'popup'):
            with self.subTest(name=name):
                self.assertFalse(self.visible(self.rules[RETURN], self.frames[name]))
        for path in ('assets/raid/result.png', 'assets/raid/waiting.png',
                     'assets/raid/prepare.png', 'assets/mailbox-native.png',
                     'assets/mailbox-loading-native.png', 'assets/mana-confirm-native.png',
                     'assets/guild-shop/potion-confirm.png', 'assets/event-potion/confirm.png',
                     'assets/daily-reward/premium-ready.png', 'assets/ad-confirm-source.png'):
            with self.subTest(path=path):
                self.assertFalse(self.visible(self.rules[RETURN], self.image(path)))

    def test_all_three_controls_and_brightness_are_required(self):
        frame = self.frames['after-fold']
        for box in ([211, 1140, 266, 1189], [542, 1140, 597, 1183], [437, 1144, 486, 1193]):
            with self.subTest(box=box):
                obscured = frame.copy()
                x1, y1, x2, y2 = box
                obscured[y1:y2, x1:x2] = 0
                self.assertFalse(self.visible(self.rules[RETURN], obscured))
        self.assertFalse(self.visible(self.rules[RETURN], (frame*.6).astype(np.uint8)))
        self.assertFalse(self.visible(self.rules[RETURN], np.zeros_like(frame)))

    def test_current_task_notice_is_recognized_before_any_background_action(self):
        self.assertTrue(self.visible(self.rules[RECOVERY_RULE_IDS[0]], self.frames['popup']))
        self.assertFalse(self.visible(self.rules[RECOVERY_RULE_IDS[1]], self.frames['popup']))
        for name in ('after-close', 'after-fold'):
            self.assertTrue(self.visible(self.rules[RECOVERY_RULE_IDS[1]], self.frames[name]))

    def test_migration_preserves_switches_intervals_other_rules_and_stricter_checks(self):
        before = deepcopy(self.config)
        for rule in before['rules']:
            if rule['id'] in CHANGED_FIELDS:
                rule.update(enabled=False, interval=47)
            if rule['id'] == RETURN:
                rule.update(threshold=.97, stable_frames=4)
        after = update(before)
        self.assertEqual(update(after), after)
        after_return = next(r for r in after['rules'] if r['id'] == RETURN)
        self.assertEqual(after_return['threshold'], .97)
        self.assertEqual(after_return['stable_frames'], 4)
        self.assertEqual(len(after['rules']), len(before['rules']))
        for old, new in zip(before['rules'], after['rules']):
            excluded = CHANGED_FIELDS.get(old['id'], ())
            self.assertEqual({k:v for k,v in old.items() if k not in excluded},
                             {k:v for k,v in new.items() if k not in excluded})
        self.assertEqual({k:v for k,v in before.items() if k != 'rules'},
                         {k:v for k,v in after.items() if k != 'rules'})

    def replay(self, scenario, legacy=False):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [r for r in config['rules'] if '_achievements_' in r['id']]
        if legacy:
            config['rules'] = [deepcopy(self.legacy.get(r['id'], r)) for r in config['rules']]
        for rule in config['rules']:
            rule.update(enabled=True, interval=.2, next_delay=0)
        frames, clock = self.frames, [0]
        class Device:
            def __init__(self, cfg, stop):
                self.stop, self.image, self.taps, self.phase = stop, None, [], 0
                self.final_frames = 0
            def screenshot(self):
                clock[0] += 1
                self.phase += 1
                count = len(self.taps)
                if count == 0:
                    name = 'main'
                elif count == 1:
                    name = 'menu'
                elif count == 2:
                    name = 'empty'
                elif count == 3 and scenario not in ('after-close', 'already-main'):
                    name = 'menu'
                else:
                    self.final_frames += 1
                    name = 'main' if scenario == 'already-main' else (
                        'after-close' if scenario == 'after-close' else 'after-fold')
                    if scenario == 'flicker' and self.final_frames == 3:
                        self.image = np.zeros_like(frames['main'])
                        return self.image
                    if scenario == 'unknown':
                        self.image = np.zeros_like(frames['main'])
                        return self.image
                self.image = frames[name].copy()
                return self.image
            def click(self, button):
                expected = ('open', 'crown', 'close', 'fold')
                suffix = button.name.removeprefix('driftinggirl_achievements_')
                assert len(self.taps) < len(expected), self.taps
                assert suffix == expected[len(self.taps)], (suffix, self.taps)
                assert self.final_frames == 0, 'No clicks allowed after returning or during combat'
                self.taps.append(suffix)
                self.phase = 0
            def stuck_record_add(self, button):
                pass
        with tempfile.TemporaryDirectory(prefix='achievement-return-') as folder:
            root = Path(folder)
            for path in {r['asset'] for r in config['rules']}:
                (root/path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/path, root/path)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])):
                    app.start(preview=False, stop_on_main=True, max_clicks=8, timeout=180)
                    app.thread.join(15)
                    self.assertFalse(app.running(), 'Replay did not finish')
                return deepcopy(app.state), list(app.device.taps), app.device.final_frames
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_recorded_failure_after_list_close_now_finishes_without_extra_click(self):
        old, taps, _ = self.replay('after-close', legacy=True)
        self.assertIn('未识别到后续界面', old['last_error'])
        state, fixed_taps, _ = self.replay('after-close')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(fixed_taps, ['open', 'crown', 'close'])
        self.assertEqual(taps, fixed_taps)

    def test_recorded_failure_after_menu_fold_now_finishes_without_extra_click(self):
        old, taps, _ = self.replay('after-fold', legacy=True)
        self.assertIn('未识别到后续界面', old['last_error'])
        state, fixed_taps, _ = self.replay('after-fold')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(fixed_taps, ['open', 'crown', 'close', 'fold'])
        self.assertEqual(taps, fixed_taps)

    def test_already_collapsed_menu_does_not_get_opened_again(self):
        state, taps, _ = self.replay('already-main')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(taps, ['open', 'crown', 'close'])

    def test_three_consecutive_clear_frames_required_after_interruption(self):
        state, _, final_frames = self.replay('flicker')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(final_frames, 6)

    def test_unknown_screen_still_stops_at_original_timeout(self):
        state, taps, _ = self.replay('unknown')
        self.assertIn('未识别到后续界面', state['last_error'])
        self.assertIsNone(state['stop_reason'])
        self.assertEqual(taps, ['open', 'crown', 'close', 'fold'])


if __name__ == '__main__':
    unittest.main()
