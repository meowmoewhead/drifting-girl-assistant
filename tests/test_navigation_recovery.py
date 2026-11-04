"""Replay the three observed entry failures; never connect to the emulator."""
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
from tools.configure_navigation_recovery import update, NEW_IDS, CHANGED_IDS


class NavigationRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules = {r['id']: r for r in cls.config['rules']}
        cls.context = SimpleNamespace(root=ROOT, config=cls.config)
        cls.frames = {name: cls.image(path) for name, path in {
            'main': 'assets/navigation/main.png',
            'battle': 'assets/navigation/world-battle.png',
            'mailbox-missed': 'tests/fixtures/navigation/mailbox-missed.png',
            'crown-missed': 'tests/fixtures/navigation/crown-missed.png',
            'menu': 'assets/achievements-menu-native.png',
            'list': 'tests/fixtures/legacy-captures/achievements-cycle-result.png',
            'mail': 'tests/fixtures/legacy-captures/mailbox-before-claim.png',
            'empty': 'tests/fixtures/legacy-captures/mailbox-after-claim.png',
        }.items()}

    @staticmethod
    def image(path):
        with Image.open(ROOT/path) as source:
            return np.asarray(source.convert('RGB'))

    def matches(self, suffix, frame, presence=True):
        r, b, o = Clicker.load_rule(self.context, self.rules['driftinggirl_'+suffix])
        matched = b.match(frame, offset=o, similarity=r['threshold'])
        if matched and presence and r.get('absent_rule'):
            guard, gb, go = Clicker.load_rule(self.context, self.rules[r['absent_rule']])
            matched = not gb.match(frame, offset=go, similarity=guard['threshold'])
        return matched

    def test_production_frames_choose_the_correct_recovery(self):
        self.assertTrue(self.matches('achievements_recover', self.frames['crown-missed']))
        self.assertTrue(self.matches('mailbox_recover', self.frames['mailbox-missed']))
        self.assertTrue(self.matches('achievements_resume', self.frames['mailbox-missed']))
        self.assertFalse(self.matches('achievements_open', self.frames['mailbox-missed']))
        for feature in ('mailbox', 'achievements'):
            self.assertTrue(self.matches(feature+'_battle_recover', self.frames['battle']))
            self.assertFalse(self.matches(feature+'_recover', self.frames['battle']))
            self.assertFalse(self.matches(feature+'_battle_recover', self.frames['main']))
            self.assertFalse(self.matches(feature+'_open', self.frames['battle']))

    def test_modal_and_dark_frames_never_trigger_entry_or_recovery(self):
        names = ('mailbox-native', 'mailbox-loading-native', 'mana-confirm-native',
                 'ad-confirm-source', 'task-popup/notice', 'guild-shop/potion-confirm',
                 'event-potion/confirm', 'achievements-list-native',
                 'raid/battle', 'raid/result', 'raid/waiting', 'raid/prepare',
                 'daily-reward/premium-ready')
        rules = list(NEW_IDS)+['driftinggirl_mailbox_open', 'driftinggirl_achievements_open']
        frames = [(name, self.image('assets/'+name+'.png')) for name in names]
        frames.append(('dark', (self.frames['main']*.6).astype(np.uint8)))
        frames.append(('dark battle', (self.frames['battle']*.6).astype(np.uint8)))
        for name, frame in frames:
            for key in rules:
                with self.subTest(frame=name, rule=key):
                    self.assertFalse(self.matches(key.removeprefix('driftinggirl_'), frame))

    def test_recovery_ignores_counters_but_requires_all_main_controls(self):
        changed = self.frames['main'].copy()
        changed[1201:1237,10:700] = 0
        self.assertTrue(self.matches('mailbox_recover', changed))
        for box in ([211,1140,266,1189], [542,1140,597,1183], [650,550,690,587], [659,479,701,505]):
            incomplete = self.frames['main'].copy()
            x1,y1,x2,y2 = box
            incomplete[y1:y2,x1:x2] = 0
            self.assertFalse(self.matches('mailbox_recover', incomplete))
        battle = self.frames['battle'].copy()
        battle[130:158,357:396] = 0
        battle[165:201,430:553] = 0
        self.assertTrue(self.matches('achievements_battle_recover', battle))

    def test_migration_preserves_switches_intervals_and_other_rules(self):
        before = deepcopy(self.config)
        before['poll_interval'] = .9
        before['rules'] = [r for r in before['rules'] if r['id'] not in NEW_IDS]
        for rule in before['rules']:
            if rule['id'].startswith(('driftinggirl_mailbox_', 'driftinggirl_achievements_')):
                rule['enabled'] = False
            if rule['id'] in CHANGED_IDS:
                rule['interval'] = 123
        after = update(before)
        self.assertEqual(update(after), after)
        self.assertEqual(after['poll_interval'], .9)
        by_id = {r['id']: r for r in after['rules']}
        for rule in before['rules']:
            if rule['id'] not in CHANGED_IDS:
                self.assertEqual(rule, by_id[rule['id']])
            else:
                self.assertEqual(by_id[rule['id']]['interval'], 123)
        for key in NEW_IDS:
            self.assertFalse(by_id[key]['enabled'])

    def test_recovery_configuration_requires_observation_and_entry(self):
        original = self.rules['driftinggirl_mailbox_recover']
        for changes in ({'action':'click'}, {'recovery_entry':'missing'},
                        {'next_stage':'mailbox'}, {'stage':'main'}, {'stop_after':True},
                        {'required_source_areas':[]}, {'color_condition':None},
                        {'stable_frames':1}, {'min_stage_seconds':0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Clicker.load_rule(self.context, {**original, **changes})

    def replay(self, feature, scenario):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        config['rules'] = [r for r in config['rules'] if r['id'].startswith('driftinggirl_'+feature+'_')]
        for rule in config['rules']:
            # The replay explicitly selects its workflow; saved UI switches
            # must not disable the test or be changed in production to run it.
            rule.update(enabled=True, interval=.2, next_delay=0)
        frames, clock = self.frames, [0]
        starts_expanded = scenario == 'expanded'
        normal = ['open', 'claim_all', 'close'] if feature == 'mailbox' else ['open', 'crown', 'close', 'fold']
        expected = normal[1:] if starts_expanded else (['open']+normal if scenario in ('missed','battle') else normal)
        class Device:
            def __init__(self, cfg, stop):
                self.stop, self.image, self.taps, self.phase, self.battle_taps = stop, None, [], 0, 0
            def screenshot(self):
                self.phase += 1
                clock[0] += 1
                count = len(self.taps)
                if count == 0:
                    name = 'menu' if starts_expanded else 'main'
                elif scenario == 'repeat':
                    name = 'crown-missed' if feature == 'achievements' else 'mailbox-missed'
                elif count == 1 and scenario in ('missed','battle','unknown','stop'):
                    name = ('battle' if scenario == 'battle' and self.phase <= 12 else
                            'crown-missed' if feature == 'achievements' else 'mailbox-missed')
                    if scenario == 'unknown':
                        self.image = np.zeros_like(frames['main'])
                        return self.image
                    if scenario == 'stop' and self.phase == 3:
                        self.stop.set()
                else:
                    progress = count + int(starts_expanded) - int(scenario in ('missed','battle'))
                    states = ['main','mail','empty','main'] if feature == 'mailbox' else ['main','menu','list','menu','main']
                    name = states[min(progress, len(states)-1)]
                self.image = frames[name].copy()
                return self.image
            def click(self, button):
                if self.stop.is_set():
                    raise InterruptedError()
                if np.array_equal(self.image, frames['battle']):
                    self.battle_taps += 1
                suffix = button.name.removeprefix('driftinggirl_'+feature+'_')
                if scenario != 'repeat':
                    assert len(self.taps) < len(expected), (scenario, self.taps, suffix)
                    assert suffix == expected[len(self.taps)], (scenario, self.taps, suffix)
                else:
                    assert suffix == 'open'
                self.taps.append(suffix)
                self.phase = 0
            def stuck_record_add(self, button):
                pass
        with tempfile.TemporaryDirectory(prefix='navigation-replay-') as folder:
            root = Path(folder)
            for path in {r['asset'] for r in config['rules']}:
                (root/path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/path, root/path)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda:clock[0])):
                    app.start(preview=scenario == 'preview', stop_on_main=True,
                              max_clicks=8, timeout=18 if scenario == 'preview' else 180)
                    app.thread.join(18)
                    self.assertFalse(app.running(), 'Replay timed out')
                self.assertEqual(app.device.battle_taps, 0)
                return deepcopy(app.state), list(app.device.taps)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_missed_entries_retry_and_complete_the_actual_workflow(self):
        for feature in ('mailbox','achievements'):
            with self.subTest(feature=feature):
                state,taps = self.replay(feature, 'missed')
                self.assertIsNone(state['last_error'])
                self.assertEqual(state['recoveries'], 1)
                self.assertEqual(state['stop_reason'], 'cycle_complete')
                self.assertEqual(taps.count('open'), 2)
                self.assertIn('close', taps)

    def test_battle_interrupt_waits_for_visible_entry_without_battle_taps(self):
        for feature in ('mailbox','achievements'):
            with self.subTest(feature=feature):
                state,taps = self.replay(feature, 'battle')
                self.assertIsNone(state['last_error'])
                self.assertEqual(state['recoveries'], 1)
                self.assertEqual(state['stop_reason'], 'cycle_complete')
                self.assertEqual(taps.count('open'), 2)

    def test_expanded_menu_is_adopted_without_toggling_entry(self):
        state,taps = self.replay('achievements', 'expanded')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(taps, ['crown','close','fold'])

    def test_repeated_failed_entries_are_bounded(self):
        state,taps = self.replay('achievements', 'repeat')
        self.assertIn('五分钟内已恢复三次', state['last_error'])
        self.assertEqual(state['recoveries'], 3)
        self.assertEqual(taps, ['open']*4)

    def test_unknown_screen_still_stops_without_retry_clicks(self):
        state,taps = self.replay('mailbox', 'unknown')
        self.assertIn('安全关闭超过40秒', state['last_error'])
        self.assertIn('保留未完成记录，拒绝未知画面点击', state['last_error'])
        self.assertEqual(taps, ['open'])
        self.assertEqual(state['recoveries'], 0)

    def test_stop_and_preview_do_not_execute_retries(self):
        for scenario in ('stop','preview'):
            with self.subTest(scenario=scenario):
                state,taps = self.replay('mailbox', scenario)
                self.assertIsNone(state['last_error'])
                self.assertEqual(taps, ['open'] if scenario == 'stop' else [])
                self.assertEqual(state['recoveries'], 0)


if __name__ == '__main__':
    unittest.main()
