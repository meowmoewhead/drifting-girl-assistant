"""Replay real fish-cake panels, price changes, overlays and controller handoff."""
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
from equipment_upgrade import EquipmentUpgrade, WEAPONS
from boss_training import BossTraining
from training_sailing import TrainingSailing
from dev_tools.configure_equipment_upgrade import add, RULE

ROOT = Path(__file__).resolve().parents[1]


def image(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/equipment'/(name+'.png')).convert('RGB'))


class EquipmentUpgradeTests(unittest.TestCase):
    def test_completion_folds_interrupted_panel_without_holds_or_completed_cycle(self):
        self.u.phase='to_top'
        self.u.yielded_for_completion()
        self.u.next_action_at=0
        self.u.step(image('top'))
        self.assertEqual(self.actions,[('tap',360,1166,'武器强化：关闭鱼饼面板')])
        self.u.next_action_at=0
        for _ in range(3):self.u.step(image('main'))
        self.assertFalse(self.u.active)
        self.assertEqual(self.u.cycles,0)

    def test_battle_interrupted_upgrade_is_not_counted_as_complete(self):
        self.u.phase='upgrade'
        self.u.yielded_for_battle()
        self.u.next_action_at=0
        for _ in range(3):self.u.step(image('main'))
        self.assertEqual(self.actions,[])
        self.assertFalse(self.u.active)
        self.assertEqual(self.u.cycles,0)

    def setUp(self):
        self.actions = []
        self.u = EquipmentUpgrade(ROOT, lambda *a: self.actions.append(('tap', *a)),
                                  lambda *a: self.actions.append(('swipe', *a)),
                                  lambda *a: self.actions.append(('hold', *a)) or True,
                                  lambda _: None)

    def test_real_frames_and_changing_levels_and_prices(self):
        self.assertTrue(self.u.main(image('main')))
        self.assertFalse(self.u.panel(image('main')))
        self.assertTrue(self.u.main(image('main-glow')))
        for name, page in (('top', 'top'), ('top-upgraded', 'top'), ('bottom', 'bottom'),
                           ('bottom-upgraded', 'bottom'), ('bottom-wide-header', 'bottom'),
                           ('top-wide-prices', 'top'), ('top-weather-header', 'top')):
            frame = image(name)
            self.assertTrue(self.u.page_visible(frame, page), name)
            self.assertFalse(self.u.main(frame), name)
            for key in ('rod', 'harpoon', 'cannon') if page == 'top' else ('turret', 'magic'):
                self.assertTrue(self.u.affordable(frame, key), (name, key))
                self.assertTrue(self.u.validate_hold(frame, key, 620, WEAPONS[key][3]))

    def test_new_round_panel_title_still_requires_currency_attack_and_brightness(self):
        frame = image('top-start-round')
        self.assertFalse(self.u._within(frame, 'header-tap', (285,797,440,831), .95, text=True))
        self.assertTrue(self.u.page_visible(frame, 'top'))
        self.assertFalse(self.u.main(frame))
        for key in ('rod', 'harpoon', 'cannon'):
            self.assertTrue(self.u.validate_hold(frame, key, 620, WEAPONS[key][3]))
        self.assertFalse(self.u.panel((frame*.4).astype(np.uint8)))
        for box in ((285,797,440,831), (470,797,595,831), (27,797,63,832), (335,784,410,799)):
            missing = frame.copy()
            x1,y1,x2,y2 = box
            missing[y1:y2,x1:x2] = 0
            self.assertFalse(self.u.panel(missing))
            self.assertFalse(self.u.validate_hold(missing,'rod',620,886))
        self.u.step(frame)
        self.assertTrue(self.u.active)
        self.assertFalse(any(a[0]=='tap' for a in self.actions))
        self.u.glyphs.pop('header-tap-low-level')
        self.assertFalse(self.u.panel(frame))

    def test_modal_dimming_wrong_currency_and_gray_arrows_block_holds(self):
        frame = image('top')
        dim = (frame*.4).astype(np.uint8)
        self.assertFalse(self.u.panel(dim))
        self.assertFalse(self.u.main((image('main')*.4).astype(np.uint8)))
        self.assertFalse(self.u.affordable(dim, 'rod'))
        gray = frame.copy()
        gray[850:905, 535:575] = gray[850:905, 535:575].mean(axis=2, keepdims=True)
        self.assertFalse(self.u.affordable(gray, 'rod'))
        wrong_currency = frame.copy()
        wrong_currency[864:910, 555:650] = 0
        self.assertFalse(self.u.affordable(wrong_currency, 'rod'))
        self.assertFalse(self.u.validate_hold(frame, 'rod', 620, 974))
        for path in ('assets/mailbox-loading-native.png', 'assets/mana-panel-native.png',
                     'assets/raid/prepare.png', 'tests/fixtures/parallel/parallel-map-rank2.png'):
            other = np.asarray(Image.open(ROOT/path).convert('RGB'))
            self.assertFalse(self.u.panel(other), path)
            self.assertFalse(self.u.main(other), path)

    def test_round_robin_is_bounded_and_closes_before_releasing_ui(self):
        screen, clock = ['main'], [0.]
        def tap(x, y, label):
            self.actions.append(('tap', label))
            screen[0] = 'top' if '打开' in label else 'main'
        def swipe(*args):
            self.actions.append(('swipe', args[-1]))
            screen[0] = 'top' if args[1] < args[3] else 'bottom'
        def hold(*args):
            self.actions.append(('hold', args[3]))
            clock[0] += args[2]
            return True
        self.u.click, self.u.swipe, self.u.hold = tap, swipe, hold
        self.u.MAX_HOLDS = 5
        with patch('equipment_upgrade.time', SimpleNamespace(monotonic=lambda: clock[0])):
            for _ in range(65):
                clock[0] += 1
                self.u.step(image(screen[0]))
                if self.u.cycles:
                    break
            self.assertEqual(self.u.cycles, 1)
            self.assertFalse(self.u.active)
            self.assertEqual([a[1] for a in self.actions if a[0] == 'hold'],
                             ['rod', 'harpoon', 'cannon', 'turret', 'magic'])
            self.assertIn(('tap', '武器强化：关闭鱼饼面板'), self.actions)
            count = len(self.actions)
            self.assertFalse(self.u.step(image('main')))
            self.assertEqual(len(self.actions), count)

    def test_larger_damage_header_is_adopted_and_remains_guarded_through_cycle(self):
        screen, clock = ['bottom-wide-header'], [0.]
        def tap(x, y, label):
            self.actions.append(('tap', label))
            screen[0] = 'main'
        def swipe(*args):
            self.actions.append(('swipe', args[-1]))
            screen[0] = 'top' if args[1] < args[3] else 'bottom-wide-header'
        def hold(*args):
            self.actions.append(('hold', args[3]))
            clock[0] += args[2]
            return True
        self.u.click, self.u.swipe, self.u.hold = tap, swipe, hold
        self.u.MAX_HOLDS = 5
        with patch('equipment_upgrade.time', SimpleNamespace(monotonic=lambda: clock[0])):
            for _ in range(65):
                clock[0] += 1
                self.u.step(image(screen[0]))
                if self.u.cycles:
                    break
        self.assertEqual(self.u.cycles, 1)
        self.assertEqual([a[1] for a in self.actions if a[0] == 'hold'],
                         ['rod', 'harpoon', 'cannon', 'turret', 'magic'])
        self.assertEqual([a[1] for a in self.actions if a[0] == 'tap'],
                         ['武器强化：关闭鱼饼面板'])
        for box in ((285, 797, 440, 831), (470, 797, 595, 831)):
            missing = image('bottom-wide-header').copy()
            x1, y1, x2, y2 = box
            missing[y1:y2, x1:x2] = 0
            self.assertFalse(self.u.panel(missing))
            self.assertFalse(self.u.validate_hold(missing, 'turret', 620, 892))
        dim = (image('bottom-wide-header') * .4).astype(np.uint8)
        self.assertFalse(self.u.panel(dim))
        self.assertFalse(self.u.validate_hold(dim, 'turret', 620, 892))

    def test_failed_fresh_recheck_never_advances_or_counts_a_hold(self):
        self.u.phase = 'upgrade'
        self.u.started_at = self.u.phase_since = 0
        self.u.hold = lambda *a: False
        with patch('equipment_upgrade.time', SimpleNamespace(monotonic=lambda: 1)):
            self.u.step(image('top'))
            self.u.step(image('top'))
        self.assertEqual(self.u.holds, 0)
        self.assertEqual(self.u.index, 0)

    def test_real_elastic_scroll_is_rechecked_after_settling_before_attempt_limit(self):
        moving, settled = image('top-scrolling-rank34'), image('top-settled-rank35')
        self.assertTrue(self.u.panel(moving))
        self.assertFalse(self.u.page_visible(moving,'top'))
        self.assertTrue(self.u.page_visible(settled,'top'))
        self.u.phase = 'to_top'
        self.u.started_at = 0
        self.u.swipes = 3
        captures = []
        self.u.capture = lambda: captures.append(True) or settled
        self.u.scroll_guard = lambda frame: True
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:10)):
            self.u.step(moving)
        self.assertEqual(captures,[True])
        self.assertEqual(self.u.phase,'upgrade')
        self.assertEqual(self.u.swipes,0)
        self.assertEqual(self.actions,[])
        self.assertEqual(self.u.holds,0)

    def test_new_scroll_frame_must_pass_context_and_panel_before_any_scroll(self):
        for fresh, allowed in ((image('top-settled-rank35'),False),
                               ((image('top')*.4).astype(np.uint8),True),
                               (image('main'),True)):
            self.u.phase = 'to_top'
            self.u.started_at = 0
            self.u.swipes = 1
            self.u.capture = lambda: fresh
            self.u.scroll_guard = lambda frame: allowed
            with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:10)):
                self.u.step(image('top-scrolling-rank34'))
            self.assertEqual(self.u.phase,'to_top')
            self.assertEqual(self.u.swipes,1)
            self.assertEqual(self.actions,[])

    def test_still_wrong_page_in_new_frame_keeps_three_scroll_limit(self):
        self.u.phase = 'to_top'
        self.u.started_at = 0
        self.u.swipes = 3
        self.u.capture = lambda:image('bottom')
        self.u.scroll_guard = lambda frame:True
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:10)):
            with self.assertRaisesRegex(RuntimeError,'滚动后未确认'):
                self.u.step(image('top'))
        self.assertEqual(self.actions,[])

    def test_scroll_settling_interval_starts_after_gesture_completion(self):
        clock, captures = [10.],[]
        self.u.phase = 'to_top'
        self.u.started_at = 0
        self.u.capture = lambda:captures.append(clock[0]) or image('bottom')
        def swipe(*args):
            self.actions.append(('swipe',args[-1]))
            clock[0] += .48
        self.u.swipe = swipe
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:clock[0])):
            self.u.step(image('bottom'))
            self.assertAlmostEqual(self.u.next_action_at,11.48)
            clock[0] = 11.1
            self.u.step(image('bottom'))
        self.assertEqual(captures,[10.])
        self.assertEqual(self.u.swipes,1)

    def test_fresh_scroll_observation_does_not_extend_cycle_deadline(self):
        # Begin before the new close reserve, then let capture itself cross
        # the original deadline. The fresh-frame guard must still reject input.
        clock = [169.9]
        self.u.phase = 'to_top'
        self.u.started_at = 0
        self.u.swipes = 1
        def capture():
            clock[0] = 180.1
            return image('bottom')
        self.u.capture = capture
        with patch('equipment_upgrade.time',SimpleNamespace(monotonic=lambda:clock[0])):
            self.u.step(image('bottom'))
        self.assertEqual(self.u.phase,'close')
        self.assertEqual(self.actions,[])
        self.assertEqual(self.u.started_at,0)
        self.assertTrue(self.u.interrupted)

    def test_shifted_title_strokes_still_require_both_headers_and_currency(self):
        frame = image('top-weather-header')
        self.assertTrue(self.u.panel(frame))
        self.assertFalse(self.u.panel((frame*.4).astype(np.uint8)))
        for box in ((285,797,440,831), (470,797,595,831), (27,797,63,832)):
            missing = frame.copy()
            x1,y1,x2,y2 = box
            missing[y1:y2,x1:x2] = 0
            self.assertFalse(self.u.panel(missing))
            self.assertFalse(self.u.validate_hold(missing,'rod',620,886))

    def test_restart_adopts_real_stop_panel_and_closes_after_verified_holds(self):
        screen, clock = ['top-weather-header'], [0.]
        self.u.click = lambda *a: (self.actions.append(('tap',a[-1])),screen.__setitem__(0,'main'))
        def swipe(*args):
            screen[0] = 'top-weather-header' if args[1] < args[3] else 'bottom'
        def hold(*args):
            self.actions.append(('hold',args[3]))
            clock[0] += args[2]
            return True
        self.u.swipe, self.u.hold = swipe, hold
        self.u.MAX_HOLDS = 5
        with patch('equipment_upgrade.time', SimpleNamespace(monotonic=lambda:clock[0])):
            for _ in range(65):
                clock[0] += 1
                self.u.step(image(screen[0]))
                if self.u.cycles:
                    break
        self.assertEqual(self.u.cycles,1)
        self.assertEqual([a[1] for a in self.actions if a[0]=='hold'],
                         ['rod','harpoon','cannon','turret','magic'])
        self.assertEqual([a[1] for a in self.actions if a[0]=='tap'],['武器强化：关闭鱼饼面板'])

    def test_sailing_guard_accepts_floating_fishcake_but_rejects_modal_or_dark_screen(self):
        training = BossTraining(ROOT,SimpleNamespace(),lambda _:None)
        sailing = TrainingSailing(ROOT,training,None,lambda _:None)
        self.u.main_guard = lambda f:sailing.running(f) is not None
        frame = np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/active.png').convert('RGB'))
        self.assertTrue(self.u.main(frame))
        self.assertFalse(self.u.main((frame*.4).astype(np.uint8)))
        for name in ('top','bottom','top-weather-header'):
            self.assertFalse(self.u.main(image(name)))
        self.assertFalse(self.u.main(np.zeros_like(frame)))

    def test_unknown_panel_never_clicks_or_scrolls(self):
        self.u.phase = 'upgrade'
        self.u.phase_since = self.u.started_at = 0
        with patch('equipment_upgrade.time', SimpleNamespace(monotonic=lambda: 36)):
            with self.assertRaisesRegex(RuntimeError, '被遮挡'):
                self.u.step(np.zeros((1280, 720, 3), dtype=np.uint8))
        self.assertEqual(self.actions, [])

    def test_migration_keeps_existing_settings_and_is_idempotent(self):
        before = {'poll_interval': .9, 'rules': [{'id': 'custom', 'enabled': False}]}
        updated = add(before)
        self.assertEqual(add(updated), updated)
        self.assertEqual(before['rules'], [{'id': 'custom', 'enabled': False}])
        updated['rules'][-1]['enabled'] = False
        updated['rules'][-1]['interval'] = 123
        self.assertEqual(add(updated), updated)

    def clicker_replay(self, disable_during_hold=False, saved_route=False):
        frames = {key: image(key) for key in ('main', 'top', 'bottom')}
        if saved_route:
            frames['top'] = image('top-weather-header')
            for name,key in (('active','main'),('stopped','stopped')):
                frames[key] = np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing'/(name+'.png')).convert('RGB'))
        actions, clock = [], [0.]
        applications, pending = [], []
        class Device:
            def __init__(self, config, stop):
                self.stop = stop
                self.io_lock = threading.RLock()
                self.image = None
                self.screen = 'top' if saved_route else 'main'
            def screenshot(self):
                clock[0] += 1
                if saved_route and not json.loads((root/'training_state.json').read_text())['sailing_enabled']:
                    self.stop.set()
                if disable_during_hold and applications and applications[0].state['applied_revision'] == 1:
                    self.stop.set()
                self.image = frames[self.screen].copy()
                return self.image
            def click(self, button):
                label = button.button
                actions.append(('tap', label))
                if saved_route:
                    self.screen = 'stopped' if self.screen == 'main' else 'main'
                else:
                    self.screen = 'top' if self.screen == 'main' else 'main'
            def command(self, *args):
                self.assert_command(args)
                if args[-1] == '12000':
                    actions.append(('hold', int(args[3]), int(args[4])))
                    clock[0] += 12
                    if disable_during_hold and len([a for a in actions if a[0] == 'hold']) == 1:
                        applications[0].change_feature({'id': 'equipment_upgrade', 'enabled': False})
                        pending.append(applications[0].status()['settings_pending'])
                else:
                    self.screen = 'top' if int(args[4]) < int(args[6]) else 'bottom'
            @staticmethod
            def assert_command(args):
                assert args[:3] == ('shell', 'input', 'swipe'), args
                assert args[-1] in ('480', '12000'), args
            def stuck_record_add(self, _):
                pass
        with tempfile.TemporaryDirectory(prefix='equipment-replay-') as directory:
            root = Path(directory)
            # Current runtime main-screen guards also read the parallel/map
            # anchors. Include the real optional source-sprite asset set so
            # this replay exercises the same recognition path as production.
            shutil.copytree(ROOT/'assets',root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            rules = [deepcopy(RULE)]
            if saved_route:
                parallel = deepcopy(next(r for r in config['rules'] if r['id']=='driftinggirl_parallel_world'))
                parallel['enabled'] = False
                rules.append(parallel)
                saved = dict(version=1,mode='farming',rank=38,start_level=1535,boss_level=1604,
                             started_at=time.time(),sailing_enabled=True,sailing_verified=True)
                (root/'training_state.json').write_text(json.dumps(saved),encoding='utf-8')
            config.update(rules=rules, poll_interval=.1)
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            if saved_route:
                app.digit_reader = SimpleNamespace(read=lambda _:SimpleNamespace(value=38,confidence=1.,text='38'))
            applications.append(app)
            try:
                with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                     patch('equipment_upgrade.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                     patch.object(EquipmentUpgrade, 'MAX_HOLDS', 5):
                    app.start(preview=False, rule_ids=None if disable_during_hold or saved_route else [RULE['id']],
                              stop_on_main=not disable_during_hold and not saved_route, timeout=300)
                    app.thread.join(15)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stage'], 'main')
                self.assertEqual([a[2] for a in actions if a[0] == 'hold'], [886, 974, 1062, 892, 980])
                if saved_route:
                    current = json.loads((root/'training_state.json').read_text())
                    self.assertEqual({k:current[k] for k in ('rank','boss_level','start_level','started_at')},
                                     {k:saved[k] for k in ('rank','boss_level','start_level','started_at')})
                    self.assertFalse(current['sailing_enabled'])
                    taps = [(a[1][0]+5,a[1][1]+5) for a in actions if a[0]=='tap']
                    self.assertEqual(taps,[(360,1166),(233,45)])
                    self.assertIsNone(app.state['parallel'])
                elif disable_during_hold:
                    self.assertEqual(pending, [True])
                    self.assertEqual(app.state['applied_revision'], 1)
                    self.assertEqual(app.state['active_rule_ids'], [])
                    self.assertFalse(app.config['rules'][0]['enabled'])
                else:
                    self.assertEqual(app.state['stop_reason'], 'cycle_complete')
                    self.assertEqual(app.state['equipment']['cycles'], 1)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_clicker_scoped_cycle_uses_real_long_holds_and_returns(self):
        self.clicker_replay()

    def test_switch_off_finishes_the_panel_then_applies_without_more_holds(self):
        self.clicker_replay(disable_during_hold=True)

    def test_restart_closes_existing_equipment_before_stopping_saved_sailing(self):
        self.clicker_replay(saved_route=True)


if __name__ == '__main__':
    unittest.main()
