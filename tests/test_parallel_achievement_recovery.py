"""Replay the real parallel boss interruption of the crown workflow."""
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
from tools.configure_parallel_achievement_recovery import ENTRY_ID, update


class ParallelAchievementRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {rule['id']: rule for rule in cls.config['rules']}
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        names = {
            'battle': 'assets/navigation/parallel-boss-battle.png',
            'menu': 'assets/achievements-menu-parallel.png',
            'old_menu': 'assets/achievements-menu-native.png',
            'list': 'assets/achievements-list-native.png',
            'main': 'assets/navigation/main.png',
            'whale': 'assets/navigation/world-battle.png',
            'mailbox': 'assets/mailbox-limit-native.png',
            'entry_failure': 'tests/fixtures/navigation/achievements-entry-rank37.png',
            'active_boss': 'tests/fixtures/navigation/mailbox-parallel-boss-active.png',
            'equipment': 'tests/fixtures/equipment/top.png',
            'loading': 'assets/mailbox-loading-native.png',
        }
        cls.frames = {key: np.asarray(Image.open(ROOT / path).convert('RGB'))
                      for key, path in names.items()}

    def match(self, identifier, frame):
        rule, button, offset = Clicker.load_rule(self.context, self.rules[identifier])
        return button.match(frame, offset=offset, similarity=rule['threshold'])

    def test_real_battle_and_night_menu_are_distinct_from_other_screens(self):
        battle = 'driftinggirl_achievements_parallel_battle_recover'
        menu = 'driftinggirl_achievements_claim_menu_recover'
        self.assertTrue(self.match(battle, self.frames['battle']))
        self.assertTrue(self.match(menu, self.frames['menu']))
        self.assertTrue(self.match('driftinggirl_achievements_resume', self.frames['menu']))
        self.assertTrue(self.match('driftinggirl_achievements_crown', self.frames['menu']))
        for name in ('old_menu', 'list', 'main', 'whale', 'mailbox'):
            with self.subTest(name=name):
                self.assertFalse(self.match(battle, self.frames[name]))
        for name in ('battle', 'list', 'main', 'whale', 'mailbox'):
            with self.subTest(name=name):
                self.assertFalse(self.match(menu, self.frames[name]))

    def replay(self, outcome):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [rule for rule in config['rules']
                           if rule['id'].startswith('driftinggirl_achievements_')]
        for rule in config['rules']:
            rule['enabled'] = True
            rule['interval'] = .2
            rule['next_delay'] = 0
        frames, clock = self.frames, [0]

        class Device:
            def __init__(self, _config, stop):
                self.stop, self.image, self.taps, self.phase_frames = stop, None, [], 0

            def screenshot(self):
                self.phase_frames += 1
                clock[0] += 1
                if not self.taps:
                    name = 'menu'
                elif outcome == 'battle' and len(self.taps) == 1 and self.phase_frames <= 15:
                    name = 'battle'
                else:
                    name = 'menu'
                self.image = frames[name].copy()
                return self.image

            def click(self, button):
                assert self.image is not None and not np.array_equal(self.image, frames['battle'])
                assert button.name == 'driftinggirl_achievements_crown'
                self.taps.append(button.name)
                self.phase_frames = 0

            def stuck_record_add(self, _button):
                pass

        with tempfile.TemporaryDirectory(prefix='parallel-crown-') as folder:
            root = Path(folder)
            assets = {rule['asset'] for rule in config['rules']}
            assets.update(path for rule in config['rules'] for path in rule.get('asset_variants', []))
            for path in assets:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, root / path)
            (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])):
                    app.start(preview=False, max_clicks=2 if outcome == 'battle' else None,
                              timeout=120)
                    app.thread.join(12)
                    self.assertFalse(app.running(), 'Replay timed out')
                return deepcopy(app.state), list(app.device.taps), list(app.events)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(2)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_battle_waits_without_tapping_and_retries_crown_after_return(self):
        state, taps, events = self.replay('battle')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'click_limit')
        self.assertEqual(taps, ['driftinggirl_achievements_crown'] * 2)
        self.assertTrue(any('平行世界战斗打断' in event['message']
                            for event in events))

    def test_unchanged_menu_retries_are_bounded(self):
        state, taps, _ = self.replay('ignored')
        self.assertIn('已重试 3 次', state['last_error'])
        self.assertEqual(taps, ['driftinggirl_achievements_crown'] * 4)

    def test_entry_failure_and_active_boss_use_clear_static_controls(self):
        for name in ('entry_failure', 'active_boss', 'battle'):
            with self.subTest(frame=name):
                self.assertTrue(self.match(ENTRY_ID, self.frames[name]))
        for name in ('main', 'menu', 'old_menu', 'list', 'whale', 'mailbox', 'equipment', 'loading'):
            with self.subTest(frame=name):
                self.assertFalse(self.match(ENTRY_ID, self.frames[name]))
        for box in ([254, 38, 289, 75], [279, 884, 307, 910],
                    [211, 1140, 266, 1189], [542, 1140, 597, 1183]):
            obscured = self.frames['entry_failure'].copy()
            x1, y1, x2, y2 = box
            obscured[y1:y2, x1:x2] = 0
            self.assertFalse(self.match(ENTRY_ID, obscured))
        self.assertFalse(self.match(ENTRY_ID, (self.frames['entry_failure']*.45).astype(np.uint8)))

    def test_entry_migration_retains_switches_intervals_and_longer_deadline(self):
        before = deepcopy(self.config)
        before['rules'] = [r for r in before['rules'] if r['id'] != ENTRY_ID]
        entry = next(r for r in before['rules'] if r['id'] == 'driftinggirl_achievements_open')
        entry.update(enabled=False, interval=137, stage_timeout=20)
        after = update(before)
        self.assertEqual(update(after), after)
        by_id = {r['id']: r for r in after['rules']}
        self.assertEqual(by_id[entry['id']], {**entry, 'stage_timeout': 45})
        self.assertFalse(by_id[ENTRY_ID]['enabled'])
        for rule in before['rules']:
            if rule['id'] != entry['id']:
                self.assertEqual(by_id[rule['id']], rule)
        by_id[ENTRY_ID]['interval'] = 9
        by_id[entry['id']]['stage_timeout'] = 90
        again = {r['id']: r for r in update(after)['rules']}
        self.assertEqual(again[ENTRY_ID]['interval'], 9)
        self.assertEqual(again[entry['id']]['stage_timeout'], 90)

    def replay_entry(self, scenario):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [r for r in config['rules']
                           if r['id'].startswith('driftinggirl_achievements_')]
        for rule in config['rules']:
            rule.update(enabled=True, interval=.2, next_delay=0)
        frames, clock = self.frames, [0]

        class Device:
            def __init__(self, _config, stop):
                self.stop, self.image, self.taps, self.phase_frames = stop, None, [], 0

            def screenshot(self):
                clock[0] += 3
                self.phase_frames += 1
                if not self.taps:
                    name = 'entry_failure'
                elif scenario == 'unknown':
                    self.image = np.zeros_like(frames['entry_failure'])
                    return self.image
                elif len(self.taps) == 1 or scenario == 'repeat':
                    name = 'active_boss' if self.phase_frames <= 4 else 'entry_failure'
                    if self.phase_frames == 1:
                        # The real first return frame arrived after the old 20s
                        # deadline. A capture beyond the new deadline must still
                        # stop, even if the current picture is recognizable.
                        clock[0] += 42 if scenario == 'expired' else 19
                else:
                    name = 'menu'
                self.image = frames[name].copy()
                return self.image

            def click(self, button):
                assert not np.array_equal(self.image, frames['active_boss']), 'Do not tap during recovery'
                self.taps.append(button.name.removeprefix('driftinggirl_achievements_'))
                self.phase_frames = 0

            def stuck_record_add(self, _button):
                pass

        with tempfile.TemporaryDirectory(prefix='parallel-crown-entry-') as folder:
            root = Path(folder)
            assets = {r['asset'] for r in config['rules']}
            assets.update(p for r in config['rules'] for p in r.get('asset_variants', []))
            for path in assets:
                (root/path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/path, root/path)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])):
                    app.start(preview=False, max_clicks=3 if scenario == 'retry' else None,
                              timeout=300)
                    app.thread.join(12)
                    self.assertFalse(app.running(), 'Entry replay did not finish')
                return deepcopy(app.state), list(app.device.taps), list(app.events)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(2)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_interrupted_entry_recovers_without_tapping_boss_then_retries_menu(self):
        state, taps, events = self.replay_entry('retry')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'click_limit')
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(taps, ['open', 'open', 'crown'])
        self.assertTrue(any('平行世界入口被打断后恢复主画面' in e['message'] for e in events))

    def test_entry_recovery_is_bounded_and_unknown_screen_still_stops(self):
        state, taps, _ = self.replay_entry('repeat')
        self.assertIn('五分钟内已恢复三次', state['last_error'])
        self.assertEqual(state['recoveries'], 3)
        self.assertEqual(taps, ['open']*4)
        for scenario in ('unknown', 'expired'):
            with self.subTest(scenario=scenario):
                state, taps, _ = self.replay_entry(scenario)
                expected = '安全关闭超过40秒' if scenario == 'unknown' else '五分钟内已重试 3 次'
                self.assertIn(expected, state['last_error'])
                self.assertEqual(state['recoveries'], 0)
                # Unknown pixels receive no cleanup input. The expired main
                # scene can retry the visible crown, under its 3-retry cap.
                self.assertEqual(taps, ['open'] if scenario == 'unknown' else ['open']*2 + ['crown']*4)


if __name__ == '__main__':
    unittest.main()
