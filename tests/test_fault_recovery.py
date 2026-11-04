"""Replay production failures without connecting to, or tapping, a device."""
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
from tools.configure_fault_recovery import update


class FaultRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {r['id']: r for r in cls.config['rules']}
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        cls.raid = {name: cls.image('assets/raid/'+name+'.png') for name in (
            'main', 'main-octopus', 'prepare', 'waiting', 'ready', 'battle',
            'result', 'return', 'task-popup', 'returned-without-confirm')}
        cls.mail = {name: cls.image('tests/fixtures/legacy-captures/'+file) for name, file in (
            ('main', 'other-functions-start.png'), ('claim', 'mailbox-before-claim.png'),
            ('empty', 'mailbox-after-claim.png'), ('return', 'mailbox-return.png'))}
        cls.mail['loading'] = cls.image('assets/mailbox-loading-native.png')
        cls.mail['limit'] = cls.image('assets/mailbox-limit-native.png')
        cls.mail['actual_return'] = cls.image('tests/fixtures/recent-stops/mailbox-return-rank30.png')

    @staticmethod
    def image(path):
        with Image.open(ROOT/path) as source:
            return np.asarray(source.convert('RGB'))

    def matches(self, suffix, frame):
        rule, button, offset = Clicker.load_rule(self.context, self.rules['driftinggirl_raid_'+suffix])
        return button.match(frame, offset=offset, similarity=rule['threshold'])

    def test_raid_entry_accepts_old_and_current_boss_but_rejects_overlays(self):
        for name in ('main', 'main-octopus'):
            with self.subTest(name=name):
                self.assertTrue(self.matches('open', self.raid[name]))
                no_badge = self.raid[name].copy()
                no_badge[815:842, 690:712] = 0
                self.assertFalse(self.matches('open', no_badge))
        for name in ('battle', 'waiting', 'prepare', 'result', 'task-popup'):
            with self.subTest(name=name):
                self.assertFalse(self.matches('open', self.raid[name]))

    def test_raid_return_accepts_failure_frame_but_requires_clear_main_controls(self):
        for name in ('main', 'main-octopus', 'return', 'returned-without-confirm'):
            with self.subTest(name=name):
                self.assertTrue(self.matches('recover', self.raid[name]))
        for name in ('battle', 'waiting', 'ready', 'prepare', 'result', 'task-popup'):
            with self.subTest(name=name):
                self.assertFalse(self.matches('recover', self.raid[name]))
        for box in ([227,1146,271,1183], [528,1148,577,1184]):
            incomplete = self.raid['returned-without-confirm'].copy()
            x1,y1,x2,y2 = box
            incomplete[y1:y2,x1:x2] = 0
            self.assertFalse(self.matches('recover', incomplete))

    def test_invalid_variants_and_stage_ages_are_rejected(self):
        for variants in ('assets/raid/main.png', [None], [''], ['../config.json'],
                         ['assets/raid/main.png']*6):
            rule = deepcopy(self.rules['driftinggirl_raid_open'])
            rule['asset_variants'] = variants
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                Clicker.load_rule(self.context, rule)
        for seconds in (-1, float('nan'), float('inf'), 3601):
            rule = deepcopy(self.rules['driftinggirl_raid_recover'])
            rule['min_stage_seconds'] = seconds
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                Clicker.load_rule(self.context, rule)

    def test_migration_preserves_user_settings_and_is_idempotent(self):
        old = deepcopy(self.config)
        old['poll_interval'] = .9
        old['rules'] = [r for r in old['rules'] if r['id'] != 'driftinggirl_raid_recover']
        for rule in old['rules']:
            if rule['id'].startswith('driftinggirl_raid_'):
                rule['enabled'] = False
            if rule['id'] == 'driftinggirl_raid_open':
                rule['interval'] = 123
                rule.pop('asset_variants', None)
                rule.pop('color_condition', None)
            if rule['id'] == 'driftinggirl_mailbox_claim_all':
                rule['stage_timeout'] = 180
        new = update(old)
        self.assertEqual(update(new), new)
        self.assertEqual(new['poll_interval'], .9)
        by_id = {r['id']: r for r in new['rules']}
        self.assertEqual(by_id['driftinggirl_raid_open']['interval'], 123)
        self.assertFalse(by_id['driftinggirl_raid_recover']['enabled'])
        self.assertEqual(by_id['driftinggirl_mailbox_claim_all']['stage_timeout'], 180)
        for rule in old['rules']:
            if rule['id'] not in ('driftinggirl_raid_open', 'driftinggirl_mailbox_claim_all'):
                self.assertEqual(rule, by_id[rule['id']])

    def replay(self, feature, device_class, clock):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [r for r in config['rules'] if r['id'].startswith('driftinggirl_'+feature+'_')]
        for rule in config['rules']:
            rule.update(enabled=True, interval=.2, next_delay=0)
        with tempfile.TemporaryDirectory(prefix='fault-replay-') as directory:
            root = Path(directory)
            paths = {r['asset'] for r in config['rules']}
            paths.update(p for r in config['rules'] for p in r.get('asset_variants', []))
            for path in paths:
                (root/path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/path, root/path)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, device_class)
            try:
                # Replace only the application's clock. Thread waits and ALAS cooldowns
                # retain real time, while simulated long network waits finish quickly.
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])):
                    app.start(preview=False, stop_on_main=True, max_clicks=5, timeout=1200)
                    app.thread.join(12)
                    self.assertFalse(app.running(), 'Replay did not finish')
                    return deepcopy(app.state), list(app.device.taps), app.device.phase_frames
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_raid_auto_return_waits_for_age_and_five_consecutive_main_frames(self):
        frames, clock = self.raid, [0]
        expected = ['open', 'prepare', 'start']
        class Device:
            def __init__(self, config, stop):
                self.stop, self.image, self.taps, self.phase_frames = stop, None, [], 0
            def screenshot(self):
                self.phase_frames += 1
                clock[0] += 1
                count, n = len(self.taps), self.phase_frames
                # A ten-frame main-screen flash before combat must not finish the raid.
                # Four main frames later also must not finish it; combat resets stability.
                name = ('main-octopus' if count == 0 else 'prepare' if count == 1 else
                        'ready' if count == 2 else 'returned-without-confirm'
                        if n <= 10 or 17 <= n <= 20 or n >= 22 else 'battle')
                self.image = frames[name].copy()
                return self.image
            def click(self, button):
                assert len(self.taps) < 3, 'Recovery must never click'
                assert button.name == 'driftinggirl_raid_'+expected[len(self.taps)]
                self.taps.append(button.name)
                self.phase_frames = 0
            def stuck_record_add(self, button):
                pass
        state, taps, frames_seen = self.replay('raid', Device, clock)
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(state['clicks'], 3)
        self.assertEqual(frames_seen, 26)
        self.assertEqual(len(taps), 3)

    def mailbox_replay(self, outcome):
        frames, clock = self.mail, [0]
        expected = ['open', 'claim_all',
                    'limit_close' if outcome in ('limit', 'delayed_return') else 'close']
        class Device:
            def __init__(self, config, stop):
                self.stop, self.image, self.taps, self.phase_frames = stop, None, [], 0
            def screenshot(self):
                self.phase_frames += 1
                clock[0] += 5
                count = len(self.taps)
                name = ('main' if count == 0 else 'claim' if count == 1 else
                        ('empty' if outcome == 'success' and self.phase_frames > 10 else
                         'limit' if outcome in ('limit', 'delayed_return') else 'loading')
                        if count == 2 else
                        ('loading' if outcome == 'delayed_return' and self.phase_frames <= 4
                         else 'return'))
                if count == 2 and outcome == 'stop' and self.phase_frames == 3:
                    self.stop.set()
                self.image = frames[name].copy()
                return self.image
            def click(self, button):
                assert len(self.taps) < 3, 'Unexpected mailbox click'
                assert button.name == 'driftinggirl_mailbox_'+expected[len(self.taps)]
                if len(self.taps) == 2 and outcome == 'success':
                    assert self.phase_frames >= 12, 'Closed before response completed'
                self.taps.append(button.name)
                self.phase_frames = 0
            def stuck_record_add(self, button):
                pass
        return self.replay('mailbox', Device, clock)

    def test_slow_mailbox_response_completes_without_repeated_claims(self):
        state, taps, _ = self.mailbox_replay('success')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(state['clicks'], 3)
        self.assertEqual(taps.count('driftinggirl_mailbox_claim_all'), 1)

    def test_full_mailbox_closes_without_discarding_the_unclaimed_mail(self):
        rule, button, offset = Clicker.load_rule(self.context, self.rules['driftinggirl_mailbox_limit_close'])
        self.assertTrue(button.match(self.mail['limit'], offset=offset, similarity=rule['threshold']))
        for name in ('claim', 'empty', 'loading', 'main'):
            self.assertFalse(button.match(self.mail[name], offset=offset, similarity=rule['threshold']))
        state, taps, _ = self.mailbox_replay('limit')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(taps, ['driftinggirl_mailbox_open',
                                'driftinggirl_mailbox_claim_all',
                                'driftinggirl_mailbox_limit_close'])

    def test_full_mailbox_accepts_main_screen_after_legacy_return_deadline(self):
        rule, button, offset = Clicker.load_rule(self.context, self.rules['driftinggirl_mailbox_return'])
        for name in ('return', 'actual_return'):
            self.assertTrue(button.match(self.mail[name], offset=offset, similarity=rule['threshold']), name)
        for name in ('claim', 'empty', 'loading', 'limit'):
            self.assertFalse(button.match(self.mail[name], offset=offset, similarity=rule['threshold']), name)
        state, taps, frames_seen = self.mailbox_replay('delayed_return')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(taps, ['driftinggirl_mailbox_open',
                                'driftinggirl_mailbox_claim_all',
                                'driftinggirl_mailbox_limit_close'])
        self.assertGreaterEqual(frames_seen, 6)

    def test_mailbox_unknown_loading_stops_after_bounded_optional_cleanup(self):
        state, taps, frames_seen = self.mailbox_replay('timeout')
        self.assertIn('安全关闭超过40秒', state['last_error'])
        self.assertIn('保留未完成记录', state['last_error'])
        self.assertEqual(taps, ['driftinggirl_mailbox_open', 'driftinggirl_mailbox_claim_all'])
        self.assertGreaterEqual(frames_seen, 24)
        self.assertLessEqual(frames_seen, 34)  # 5s/frame, original 120s plus bounded cleanup.

    def test_mailbox_stop_interrupts_loading_without_more_taps(self):
        state, taps, frames_seen = self.mailbox_replay('stop')
        self.assertIsNone(state['last_error'])
        self.assertEqual(len(taps), 2)
        self.assertEqual(frames_seen, 3)


if __name__ == '__main__':
    unittest.main()
