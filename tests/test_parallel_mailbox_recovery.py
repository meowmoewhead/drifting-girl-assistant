"""Replay mailbox failures during real parallel-world boss transitions."""
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
from tools.configure_parallel_mailbox_recovery import NEW_IDS, update


class ParallelMailboxRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {rule['id']: rule for rule in cls.config['rules']}
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        paths = {
            'main': 'assets/navigation/main.png',
            'transition': 'tests/fixtures/navigation/mailbox-parallel-interruption.png',
            'battle': 'tests/fixtures/navigation/mailbox-parallel-boss-active.png',
            'mail': 'assets/mailbox-native.png',
            'limit': 'assets/mailbox-limit-native.png',
            'loading': 'assets/mailbox-loading-native.png',
            'equipment_top': 'tests/fixtures/equipment/top.png',
            'equipment_bottom': 'tests/fixtures/equipment/bottom.png',
            'crown': 'assets/achievements-list-native.png',
            'menu': 'assets/achievements-menu-parallel.png',
        }
        cls.frames = {key: np.asarray(Image.open(ROOT / path).convert('RGB'))
                      for key, path in paths.items()}

    def matches(self, identifier, frame):
        rule, button, offset = Clicker.load_rule(self.context, self.rules[identifier])
        return button.match(frame, offset=offset, similarity=rule['threshold'])

    def test_real_failure_and_active_battle_match_without_animated_button(self):
        for identifier in NEW_IDS:
            for name in ('transition', 'battle'):
                with self.subTest(rule=identifier, frame=name):
                    self.assertTrue(self.matches(identifier, self.frames[name]))
            for name in ('main', 'mail', 'limit', 'loading', 'equipment_top',
                         'equipment_bottom', 'crown', 'menu'):
                with self.subTest(rule=identifier, frame=name):
                    self.assertFalse(self.matches(identifier, self.frames[name]))

    def test_clear_controls_are_required_and_dimmed_underlays_are_rejected(self):
        for identifier in NEW_IDS:
            for box in ([254, 38, 289, 75], [279, 884, 307, 910],
                        [211, 1140, 266, 1189], [542, 1140, 597, 1183]):
                with self.subTest(rule=identifier, box=box):
                    frame = self.frames['transition'].copy()
                    x1, y1, x2, y2 = box
                    frame[y1:y2, x1:x2] = 0
                    self.assertFalse(self.matches(identifier, frame))
            dimmed = (self.frames['transition'] * .45).astype(np.uint8)
            self.assertFalse(self.matches(identifier, dimmed))

    def test_migration_preserves_user_settings_and_disabled_mailbox(self):
        before = deepcopy(self.config)
        before['poll_interval'] = .9
        before['rules'] = [r for r in before['rules'] if r['id'] not in NEW_IDS]
        entry = next(r for r in before['rules'] if r['id'] == 'driftinggirl_mailbox_open')
        entry.update(enabled=False, interval=123, stage_timeout=20)
        after = update(before)
        self.assertEqual(update(after), after)
        self.assertEqual(after['poll_interval'], .9)
        by_id = {r['id']: r for r in after['rules']}
        self.assertEqual(by_id[entry['id']], {**entry, 'stage_timeout': 45})
        for rule in before['rules']:
            if rule['id'] != entry['id']:
                self.assertEqual(by_id[rule['id']], rule)
        for identifier in NEW_IDS:
            self.assertFalse(by_id[identifier]['enabled'])
        by_id[NEW_IDS[0]]['interval'] = 7
        by_id[entry['id']]['stage_timeout'] = 90
        again = update(after)
        self.assertEqual(next(r for r in again['rules'] if r['id'] == NEW_IDS[0])['interval'], 7)
        self.assertEqual(next(r for r in again['rules'] if r['id'] == entry['id'])['stage_timeout'], 90)

    def replay(self, scenario):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [r for r in config['rules'] if r['id'].startswith('driftinggirl_mailbox_')]
        for rule in config['rules']:
            rule.update(enabled=True, interval=.2, next_delay=0)
        frames, clock = self.frames, [0]
        expected = ['open', 'open', 'claim_all', 'limit_close']

        class Device:
            def __init__(self, _config, stop):
                self.stop, self.image, self.taps, self.phase_frames = stop, None, [], 0

            def screenshot(self):
                self.phase_frames += 1
                clock[0] += 3
                count = len(self.taps)
                if count == 0:
                    name = 'main'
                elif scenario == 'unknown':
                    self.image = np.zeros_like(frames['main'])
                    return self.image
                elif scenario == 'repeat' or count == 1:
                    name = 'transition' if self.phase_frames <= 4 else 'main'
                    if self.phase_frames == 1:
                        clock[0] += 19  # The first post-click capture exceeded the old 20s deadline.
                elif count == 2:
                    name = 'mail'
                elif count == 3:
                    name = 'limit'
                else:
                    name = 'loading' if scenario == 'slow_return' and self.phase_frames <= 14 else 'battle'
                self.image = frames[name].copy()
                return self.image

            def click(self, button):
                assert not np.array_equal(self.image, frames['transition'])
                assert not np.array_equal(self.image, frames['battle'])
                suffix = button.name.removeprefix('driftinggirl_mailbox_')
                if scenario in ('repeat', 'unknown'):
                    assert suffix == 'open'
                else:
                    assert suffix == expected[len(self.taps)], (self.taps, suffix)
                self.taps.append(suffix)
                self.phase_frames = 0

            def stuck_record_add(self, _button):
                pass

        with tempfile.TemporaryDirectory(prefix='parallel-mailbox-') as folder:
            root = Path(folder)
            for path in {r['asset'] for r in config['rules']}:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / path, root / path)
            (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])):
                    app.start(preview=False, stop_on_main=True, timeout=600)
                    app.thread.join(12)
                    self.assertFalse(app.running(), 'Replay did not finish')
                return deepcopy(app.state), list(app.device.taps), list(app.events)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(2)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_interrupted_open_retries_and_return_accepts_boss_without_extra_taps(self):
        state, taps, events = self.replay('battle')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(state['recoveries'], 1)
        self.assertEqual(taps, ['open', 'open', 'claim_all', 'limit_close'])
        self.assertTrue(any('关闭后确认平行世界战斗主画面' in e['message'] for e in events))

    def test_return_at_timeout_boundary_finishes_consecutive_observations(self):
        state, taps, _ = self.replay('slow_return')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(taps, ['open', 'open', 'claim_all', 'limit_close'])

    def test_repeated_interruptions_are_bounded_and_unknown_screen_still_stops(self):
        state, taps, _ = self.replay('repeat')
        self.assertIn('五分钟内已恢复三次', state['last_error'])
        self.assertEqual(state['recoveries'], 3)
        self.assertEqual(taps, ['open'] * 4)
        state, taps, _ = self.replay('unknown')
        self.assertIn('安全关闭超过40秒', state['last_error'])
        self.assertIn('保留未完成记录，拒绝未知画面点击', state['last_error'])
        self.assertEqual(state['recoveries'], 0)
        self.assertEqual(taps, ['open'])


if __name__ == '__main__':
    unittest.main()
