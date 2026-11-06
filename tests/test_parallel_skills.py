"""Replay native skill states and interrupted talent-point transactions."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import shutil
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from digit_ocr import DigitReader
from parallel_skills import ParallelSkills
from app import Clicker
from boss_training import BossTraining
from training_sailing import TrainingSailing
from dev_tools.configure_parallel_skills import RULE, add

ROOT = Path(__file__).resolve().parents[1]

def image(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/parallel-skills'/(name+'.png')).convert('RGB'))


def enable_skill_fixture(config):
    """Exercise skill recovery explicitly, independent of safe release defaults."""
    from drift.tasks.cruise_options import DEFAULTS
    for rule in config['rules']:
        if rule['id'] == 'driftinggirl_parallel_skills':
            rule['cruise_settings'] = deepcopy(DEFAULTS)
            rule['cruise_settings']['cruise']['enabled'] = False


class ParallelSkillsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.actions = []
        self.u = ParallelSkills(ROOT, lambda *a: self.actions.append(a) or True,
                                lambda _: None, self.reader)
        self.u.path = Path(self.temp.name)/'parallel_skill_state.json'

    def tearDown(self):
        self.temp.cleanup()

    def test_native_states_percent_and_five_digit_balance(self):
        for name, enhance, auto, percent, balance, active in (
            ('parallel-skill-panel', False, False, 345, 12436, False),
            ('parallel-fishing-auto', False, True, 345, 12436, False),
            ('parallel-fishing-enhance', True, True, 690, 12436, False),
            ('parallel-growth-confirm', True, True, 690, 12237, True)):
            a = image(name)
            self.assertTrue(self.u.panel(a), name)
            self.assertEqual(self.u.toggle(a, True), enhance, name)
            self.assertEqual(self.u.toggle(a), auto, name)
            self.assertEqual(self.u.effect(a), percent, name)
            self.assertEqual(self.u.points(a), balance, name)
            self.assertEqual(self.u.growth_active(a), active, name)
        self.assertTrue(self.u.main(image('main')))
        self.assertTrue(self.u.modal(image('parallel-growth-points')))

    def test_dimmed_other_panels_and_changed_purchase_are_rejected(self):
        a = image('parallel-skill-panel')
        self.assertFalse(self.u.panel((a*.4).astype(np.uint8)))
        self.assertFalse(self.u.main(a))
        for name in ('tests/fixtures/equipment/top.png', 'tests/fixtures/parallel/parallel-map-rank2.png'):
            other = np.asarray(Image.open(ROOT/name).convert('RGB'))
            self.assertFalse(self.u.panel(other))
            self.assertFalse(self.u.modal(other))
        modal = image('parallel-growth-points')
        for name in ('confirm-currency', 'confirm-cost', 'confirm-growth'):
            changed = modal.copy()
            x, y, X, Y = self.u.BOXES[name]
            changed[y:Y, x:X] = 0
            self.assertFalse(self.u.modal(changed), name)
            self.assertFalse(self.u.validate('confirm', changed, 210, 788), name)

    def test_native_dark_point_outlines_keep_two_strong_balance_reads(self):
        self.assertEqual(self.u.points(image('parallel-sailing-open-timeout')), 11507)
        gear = image('parallel-points-dark-outline')
        self.assertEqual(self.u.points(gear), 11562)
        self.assertFalse(self.u.panel(gear))
        self.assertFalse(self.u.validate('open_growth', gear, 530, 1077))

    def test_point_consensus_requires_two_strong_views_and_no_strong_conflict(self):
        frame = image('parallel-sailing-open-timeout')
        result = lambda text, confidence: SimpleNamespace(text=text, confidence=confidence)
        for readings, expected in (
            ([result('11507', .995), result('11507', .999), result('11507', .83)], 11507),
            ([result('11507', .995), result('11507', .83), result('11507', .82)], None),
            ([result('11507', .995), result('11507', .999), result('11501', .98)], None),
            ([result('11507', .995), result('11507', .999), result('115O7', .98)], None),
        ):
            with patch.object(self.u.reader, 'read', side_effect=readings):
                self.assertEqual(self.u.points(frame), expected)

    def test_toggle_fresh_validation_never_turns_off_enabled_fishing(self):
        off, on = image('parallel-skill-panel'), image('parallel-growth-confirm')
        self.assertTrue(self.u.validate('enhance', off, 500, 886))
        self.assertTrue(self.u.validate('auto1', off, 558, 886))
        self.assertFalse(self.u.validate('enhance', on, 500, 886))
        self.assertFalse(self.u.validate('auto1', on, 558, 886))
        self.assertFalse(self.u.validate('open_growth', on, 530, 1077))
        self.assertFalse(self.u.validate('open_growth', off, 646, 1077))
        for value in (None, 0, 199):
            with patch.object(self.u, 'points', return_value=value):
                self.assertFalse(self.u.validate('open_growth', off, 530, 1077))
        with patch.object(self.u, 'points', return_value=200):
            self.assertTrue(self.u.validate('open_growth', off, 530, 1077))

    def test_complete_cycle_enhances_then_auto_and_purchases_only_once(self):
        frames = {name: image(name) for name in ('main', 'parallel-skill-panel',
                  'parallel-fishing-enhance', 'parallel-growth-points', 'parallel-growth-confirm')}
        enhanced_off = frames['parallel-fishing-enhance'].copy()
        enhanced_off[858:910, 533:583] = frames['parallel-skill-panel'][858:910, 533:583]
        frames['enhanced-off'] = enhanced_off
        screen, clock = ['main'], [0.]
        next_screen = dict(open='parallel-skill-panel', enhance='enhanced-off',
                           auto1='parallel-fishing-enhance', open_growth='parallel-growth-points',
                           confirm='parallel-growth-confirm', close='main')
        def click(x, y, label, action):
            current = frames[screen[0]]
            self.assertTrue(self.u.validate(action, current, x, y), action)
            self.u.before_click(action, current)
            self.actions.append(action)
            screen[0] = next_screen[action]
            return True
        self.u.click = click
        with patch('parallel_skills.time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 100000+clock[0])):
            for _ in range(45):
                clock[0] += 1
                self.u.step(frames[screen[0]])
                if self.u.cycles:
                    break
            self.assertEqual(self.u.cycles, 1)
            self.assertEqual(self.actions, ['open', 'enhance', 'auto1', 'open_growth', 'confirm', 'close'])
            saved = json.loads(self.u.path.read_text())
            self.assertTrue(saved['confirmed'])
            self.assertFalse(saved['pending'])
            self.assertGreater(saved['growth_until'], 103600)
            self.assertFalse(self.u.validate('open_growth', frames['parallel-skill-panel'], 530, 1077))
            self.assertFalse(self.u.validate('confirm', frames['parallel-growth-points'], 210, 788))

    def test_existing_buff_skips_purchase_and_already_enabled_toggles(self):
        screen, clock = ['parallel-growth-confirm'], [0.]
        def click(x, y, label, action):
            self.assertEqual(action, 'close')
            self.actions.append(action)
            screen[0] = 'main'
            return True
        self.u.click = click
        with patch('parallel_skills.time.monotonic', side_effect=lambda: clock[0]):
            for _ in range(12):
                clock[0] += 1
                self.u.step(image(screen[0]))
        self.assertEqual(self.actions, ['close'])
        self.assertEqual(self.u.cycles, 1)
        self.assertFalse(self.u.path.exists())

    def test_native_opening_at_timeout_finishes_observations_without_buying_again(self):
        panel = image('parallel-sailing-open-timeout')
        self.assertTrue(self.u.panel(panel))
        self.assertTrue(self.u.growth_active(panel))
        self.u.journal = dict(pending=False, confirmed=True, growth_until=200000)
        self.u._save()
        saved = self.u.path.read_bytes()
        clock = [0.]
        with patch('parallel_skills.time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 100000+clock[0])):
            self.u._set('opening')
            clock[0] = 31
            self.assertTrue(self.u.step(panel))
            self.assertEqual(self.u.phase, 'opening')
            self.assertEqual(self.actions, [])
            clock[0] = 32
            self.u.step(panel)
            self.assertEqual(self.u.phase, 'configure')
            for second in (33, 34):
                clock[0] = second
                self.u.step(panel)
            self.assertEqual(self.u.phase, 'closing')
            self.assertEqual(self.actions, [])
            self.assertEqual(self.u.path.read_bytes(), saved)

    def assert_expired_navigation_or_purchase(self,frame,second):
        phase=self.u.phase;cycles=self.u.cycles
        saved=self.u.path.read_bytes() if self.u.path.exists() else None
        try:
            self.u.step(frame)
        except RuntimeError as exc:
            self.assertIn('超时',str(exc))
        else:
            # Current DS navigation policy abandons an unopened menu, while
            # older supported controllers stop. Purchase phases must still stop.
            self.assertIn(phase,('opening','await_main','closing','configure'))
            self.assertEqual(self.u.phase,'idle')
            self.assertEqual(self.u.next_check_at,second+30)
            self.assertEqual(self.u.cycles,cycles)
            self.assertEqual(self.u.path.read_bytes() if self.u.path.exists() else None,saved)
        self.assertEqual(self.actions,[])

    def test_timeout_grace_requires_visible_expected_page_and_has_a_fixed_limit(self):
        panel = image('parallel-sailing-open-timeout')
        for phase, frame, second in (
            ('opening', (panel*.4).astype(np.uint8), 31),
            ('opening', panel, 39), ('confirm', panel, 31),
            ('verify', panel, 31), ('await_main', panel, 31)):
            self.u.phase, self.u.since = phase, 0
            with patch('parallel_skills.time.monotonic', return_value=second):
                self.assert_expired_navigation_or_purchase(frame,second)
        self.assertEqual(self.actions, [])

    def test_opening_observations_must_be_consecutive(self):
        panel = image('parallel-sailing-open-timeout')
        self.u.phase = 'opening'
        self.u.step(panel)
        self.u.step((panel*.4).astype(np.uint8))
        self.u.step(panel)
        self.assertEqual(self.u.phase, 'opening')
        self.u.step(panel)
        self.assertEqual(self.u.phase, 'configure')
        self.assertEqual(self.actions, [])

    def interrupted_main(self):
        training = BossTraining(ROOT, self.reader, lambda _: None,
                                state_path=Path(self.temp.name)/'training.json')
        sailing = TrainingSailing(ROOT, training, None, lambda _: None)
        self.u.main_guard = lambda frame: sailing.running(frame) is not None
        frame = image('opening-interrupted-main')
        self.assertTrue(self.u.main(frame))
        return frame

    def test_native_lost_opening_after_verified_battle_defers_without_completed_cycle_or_purchase(self):
        frame = self.interrupted_main()
        self.u.journal = dict(pending=False, confirmed=True, growth_until=200000,
                              opened_at=90000, balance_before=500)
        self.u._save()
        saved = self.u.path.read_bytes()
        clock = [0.]
        with patch('parallel_skills.time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 100000+clock[0])):
            self.u._set('opening')
            self.u.suspend(60)
            for second in (91,92):
                clock[0] = second
                self.u.step(frame)
                self.assertEqual(self.u.phase, 'opening' if second == 91 else 'idle')
            self.assertEqual(self.u.cycles,0)
            self.assertEqual(self.actions,[])
            self.assertEqual(self.u.next_check_at,122)
            self.assertFalse(self.u.step(frame))
            self.assertEqual(self.u.path.read_bytes(),saved)

    def test_interrupted_opening_main_recovery_requires_new_consecutive_valid_frames(self):
        frame = self.interrupted_main()
        clock = [0.]
        with patch('parallel_skills.time.monotonic',side_effect=lambda:clock[0]):
            self.u._set('opening')
            self.u.suspend(60)
            for second,current in ((61,frame),(62,(frame*.4).astype(np.uint8)),(63,frame)):
                clock[0] = second
                self.u.step(current)
                self.assertEqual(self.u.phase,'opening')
            clock[0] = 64
            self.u.step(frame)
            self.assertEqual(self.u.phase,'idle')
        self.assertEqual(self.actions,[])
        self.assertEqual(self.u.cycles,0)

    def test_lost_opening_keeps_unknown_purchase_and_absolute_grace_timeouts(self):
        frame = self.interrupted_main()
        for phase,current,suspended,second in (
            ('opening',frame,False,31),
            ('opening',frame,True,99),
            ('opening',(frame*.4).astype(np.uint8),True,91),
            ('opening',image('parallel-growth-points'),True,91),
            ('confirm',frame,True,91),('verify',frame,True,91)):
            with self.subTest(phase=phase,suspended=suspended,second=second):
                with patch('parallel_skills.time.monotonic',return_value=0):
                    self.u._set(phase)
                    if suspended:
                        self.u.suspend(60)
                with patch('parallel_skills.time.monotonic',return_value=second):
                    self.assert_expired_navigation_or_purchase(current,second)
        self.assertEqual(self.actions,[])

    def test_interrupted_opening_still_adopts_a_real_panel(self):
        panel = image('parallel-sailing-open-timeout')
        with patch('parallel_skills.time.monotonic',return_value=0):
            self.u._set('opening')
            self.u.suspend(60)
        for second in (91,92):
            with patch('parallel_skills.time.monotonic',return_value=second):
                self.u.step(panel)
        self.assertEqual(self.u.phase,'configure')
        self.assertFalse(self.u.opening_interrupted)
        self.assertEqual(self.actions,[])

    def test_visible_panel_is_adopted_during_entry_cooldown(self):
        self.u.next_check_at = float('inf')
        self.assertTrue(self.u.step(image('parallel-sailing-open-timeout')))
        self.assertEqual(self.u.phase, 'configure')
        self.assertEqual(self.actions, [])

    def test_sailing_main_guard_handles_obscured_footer_but_rejects_panels(self):
        training = BossTraining(ROOT, self.reader, lambda _: None,
                                state_path=Path(self.temp.name)/'training.json')
        sailing = TrainingSailing(ROOT, training, None, lambda _: None)
        self.u.main_guard = lambda frame: sailing.running(frame) is not None
        a = np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/active-flying-cake.png').convert('RGB'))
        self.assertFalse(self.u.screen.main(a))
        self.assertTrue(self.u.main(a))
        self.assertTrue(self.u.validate('open', a, 45, 1166))
        self.assertFalse(self.u.main((a*.4).astype(np.uint8)))
        self.assertFalse(self.u.main(image('parallel-sailing-open-timeout')))
        self.assertFalse(self.u.main(image('parallel-growth-points')))
        self.u.phase, self.u.since = 'await_main', 0
        with patch('parallel_skills.time.monotonic', return_value=31):
            self.u.step(a)
        with patch('parallel_skills.time.monotonic', return_value=32):
            self.u.step(a)
        self.assertEqual(self.u.cycles, 1)
        self.assertEqual(self.actions, [])

    def test_worker_resumes_native_open_panel_without_restarting_verified_route(self):
        root = Path(self.temp.name)
        shutil.copytree(ROOT/'assets', root/'assets')
        config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        enable_skill_fixture(config)
        config.update(rules=[deepcopy(r) for r in config['rules']
                            if r['id'] in ('driftinggirl_parallel_world', 'driftinggirl_parallel_skills')],
                      poll_interval=.1)
        for rule in config['rules']:
            rule['enabled'] = True
        (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
        training = BossTraining(root, self.reader, lambda _: None)
        training.begin(38, 1535, 1604)
        training.set_sailing(True, verified=True)
        record = (root/'training_state.json').read_bytes()
        (root/'parallel_skill_state.json').write_text(json.dumps(dict(
            pending=False, confirmed=True, growth_until=200000)), encoding='utf-8')
        journal = (root/'parallel_skill_state.json').read_bytes()
        actions, clock = [], [0.]
        frames = {'panel': image('parallel-sailing-open-timeout'),
                  'main': np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/active-flying-cake.png').convert('RGB'))}
        class Device:
            def __init__(self, config, stop):
                self.stop, self.io_lock = stop, threading.RLock()
                self.screen, self.image = 'panel', None
            def screenshot(self):
                clock[0] += 1
                self.image = frames[self.screen].copy()
                return self.image
            def click(self, button):
                point = ((button.button[0]+button.button[2])//2,
                         (button.button[1]+button.button[3])//2)
                actions.append(point)
                if point != (360, 700):
                    raise AssertionError(point)
                self.screen = 'main'
            def stuck_record_add(self, _):
                pass
        app = Clicker(root, Device)
        app.digit_reader = self.reader
        try:
            with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                 patch('parallel_skills.time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 100000+clock[0])):
                app.start(preview=False, stop_on_main=True, timeout=100)
                app.thread.join(12)
            self.assertFalse(app.running())
            self.assertIsNone(app.state['last_error'])
            self.assertEqual(app.state['stop_reason'], 'cycle_complete')
            self.assertEqual(actions, [(360, 700)])
            self.assertEqual((root/'training_state.json').read_bytes(), record)
            self.assertEqual((root/'parallel_skill_state.json').read_bytes(), journal)
        finally:
            app.stop()
            if app.thread:
                app.thread.join(3)
            for handler in list(app.log.handlers):
                handler.close()
                app.log.removeHandler(handler)

    def test_restart_during_confirm_uses_reservation_to_cancel_duplicate(self):
        modal, panel = image('parallel-growth-points'), image('parallel-skill-panel')
        self.u.before_click('open_growth', panel)
        self.assertTrue(self.u.validate('confirm', modal, 210, 788))
        self.u.before_click('confirm', modal)
        reopened = ParallelSkills(ROOT, lambda *a: True, lambda _: None, self.reader)
        reopened.journal = json.loads(self.u.path.read_text())
        self.assertFalse(reopened.validate('confirm', modal, 210, 788))
        self.assertTrue(reopened.validate('cancel', modal, 510, 788))
        self.assertFalse(reopened.validate('open_growth', panel, 530, 1077))

    def test_worker_adopts_owned_occluded_prompt_before_training_reads(self):
        root = Path(self.temp.name)
        shutil.copytree(ROOT/'assets',root/'assets')
        config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        enable_skill_fixture(config)
        config.update(rules=[deepcopy(r) for r in config['rules']
                            if r['id'] in ('driftinggirl_parallel_world','driftinggirl_parallel_skills')],poll_interval=.1)
        for rule in config['rules']:
            rule['enabled'] = True
        (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
        training = BossTraining(root,self.reader,lambda _:None)
        training.begin(38,1535,1810)
        training.set_sailing(True,verified=True)
        record = (root/'training_state.json').read_bytes()
        (root/'parallel_skill_state.json').write_text(json.dumps(dict(
            pending=True,opened_at=1,balance_before=10772,growth_until=0)),encoding='utf-8')
        actions,clock = [],[0.]
        frames = {'modal':image('parallel-growth-occluded'),'panel':image('parallel-skill-panel'),
                  'main':np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/active-flying-cake.png').convert('RGB'))}
        class Device:
            def __init__(self,config,stop):
                self.stop,self.io_lock = stop,threading.RLock()
                self.screen,self.image = 'modal',None
            def screenshot(self):
                clock[0] += 1
                self.image = frames[self.screen].copy()
                return self.image
            def click(self,button):
                point = ((button.button[0]+button.button[2])//2,(button.button[1]+button.button[3])//2)
                actions.append(point)
                if point == (510,788):
                    self.screen = 'panel'
                elif point == (360,700):
                    self.screen = 'main'
                else:
                    raise AssertionError(point)
            def stuck_record_add(self,_):
                pass
        app = Clicker(root,Device)
        app.digit_reader = self.reader
        try:
            with patch('app.time',SimpleNamespace(monotonic=lambda:clock[0])), \
                 patch('parallel_skills.time',SimpleNamespace(monotonic=lambda:clock[0],time=lambda:100000+clock[0])):
                app.start(preview=False,stop_on_main=True,timeout=100)
                app.thread.join(12)
            self.assertFalse(app.running())
            self.assertIsNone(app.state['last_error'])
            self.assertEqual(app.state['stop_reason'],'cycle_complete')
            self.assertEqual(actions,[(510,788),(360,700)])
            self.assertEqual((root/'training_state.json').read_bytes(),record)
            saved = json.loads((root/'parallel_skill_state.json').read_text())
            self.assertFalse(saved['pending'])
            self.assertEqual(saved['growth_until'],0)
        finally:
            app.stop()
            if app.thread:
                app.thread.join(3)
            for handler in list(app.log.handlers):
                handler.close()
                app.log.removeHandler(handler)

    def test_occluded_purchase_can_only_be_cancelled_from_its_saved_record(self):
        a = image('parallel-growth-occluded')
        self.assertFalse(self.u.modal(a))
        self.assertTrue(self.u.purchase_prompt(a))
        self.u.main_guard = lambda _: True
        self.assertFalse(self.u.main(a))
        self.assertFalse(self.u.validate('confirm',a,210,788))
        self.assertFalse(self.u.validate('cancel_recovery',a,510,788))
        self.u.journal = dict(pending=True,opened_at=1,balance_before=10772,growth_until=2)
        self.u.step(a)
        self.assertEqual(self.u.phase,'cancel_recovery')
        self.assertTrue(self.u.validate('cancel_recovery',a,510,788))
        self.assertFalse(self.u.validate('confirm',a,210,788))
        self.u.before_click('cancel_recovery',a)
        self.assertFalse(self.u.journal['pending'])
        self.assertEqual(self.u.journal['growth_until'],2)
        for name in ('confirm-currency','confirm-cost','continue'):
            changed = a.copy()
            x,y,X,Y = self.u.BOXES[name]
            changed[y:Y,x:X] = 0
            self.assertFalse(self.u.validate('cancel_recovery',changed,510,788),name)
        self.assertFalse(self.u.validate('cancel_recovery',image('parallel-growth-transient-world'),510,788))

    def test_confirm_timeout_cancels_safely_then_closes_and_defers_retry(self):
        clock = [0.]
        modal,panel = image('parallel-growth-occluded'),image('parallel-skill-panel')
        self.u.journal = dict(pending=True,opened_at=99990,balance_before=10772,growth_until=0)
        def click(x,y,label,action):
            a = modal if action == 'cancel_recovery' else panel
            self.assertTrue(self.u.validate(action,a,x,y))
            self.u.before_click(action,a)
            self.actions.append(action)
            return True
        self.u.click = click
        self.u.phase,self.u.since = 'confirm',0
        with patch('parallel_skills.time.monotonic',side_effect=lambda:clock[0]), \
             patch('parallel_skills.time.time',return_value=100000):
            clock[0] = 31
            self.u.step(image('parallel-growth-transient-world'))
            self.assertEqual(self.u.phase,'cancel_recovery')
            clock[0] = 32
            self.u.step(modal)
            clock[0] = 33
            self.u.step(modal)
            clock[0] = 35
            self.u.step(panel)
            clock[0] = 36
            self.u.step(panel)
            clock[0] = 38
            self.u.step(image('main'))
            clock[0] = 39
            self.u.step(image('main'))
        self.assertEqual(self.actions,['cancel_recovery','close'])
        self.assertEqual(self.u.phase,'idle')
        self.assertGreater(self.u.next_check_at,300)
        self.assertFalse(self.u.journal['pending'])
        self.assertEqual(self.u.journal['growth_until'],0)

    def test_confirmation_and_recovery_require_consecutive_valid_frames(self):
        modal = image('parallel-growth-points')
        bad = image('parallel-growth-transient-world')
        self.u.phase = 'confirm'
        self.u.step(modal)
        self.u.step(bad)
        self.u.step(modal)
        self.assertEqual(self.actions,[])
        self.u.phase,self.u.since = 'cancel_recovery',0
        self.u.journal = dict(pending=True,opened_at=1,balance_before=10772)
        with patch('parallel_skills.time.monotonic',return_value=31):
            with self.assertRaisesRegex(RuntimeError,'超时'):
                self.u.step(bad)
        self.assertEqual(self.actions,[])

    def test_disabled_points_option_and_stale_pending_cannot_confirm(self):
        modal, panel = image('parallel-growth-points'), image('parallel-skill-panel')
        self.u.growth_points = False
        self.assertFalse(self.u.validate('open_growth', panel, 530, 1077))
        self.u.journal = dict(pending=True, opened_at=1)
        self.assertFalse(self.u.validate('confirm', modal, 210, 788))
        self.assertTrue(self.u.validate('cancel', modal, 510, 788))

    def test_other_auto_skills_are_only_switched_off_when_enabled(self):
        a = image('parallel-growth-confirm').copy()
        for row in (2, 3):
            y = (951, 1015)[row-2]
            a[y-21:y+22, 539:544] = a[865:908, 539:544]
            self.assertTrue(self.u.validate('auto'+str(row), a, 558, y))
            self.assertFalse(self.u.validate('auto'+str(row), image('parallel-growth-confirm'), 558, y))

    def test_clicker_owns_panel_until_return_and_applies_disable_after_close(self):
        actions, clock, apps, pending = [], [0.], [], []
        frames = {name: image(name) for name in ('main', 'parallel-growth-confirm')}
        class Device:
            def __init__(self, config, stop):
                self.stop, self.io_lock = stop, threading.RLock()
                self.screen = 'main'
                self.image = None
            def screenshot(self):
                clock[0] += 1
                if apps and apps[0].state['applied_revision'] == 1:
                    self.stop.set()
                self.image = frames[self.screen].copy()
                return self.image
            def click(self, button):
                point = ((button.button[0]+button.button[2])//2,
                         (button.button[1]+button.button[3])//2)
                actions.append(point)
                if point == (45, 1166):
                    self.screen = 'parallel-growth-confirm'
                    apps[0].change_feature({'id': 'parallel_skills', 'enabled': False})
                    pending.append(apps[0].status()['settings_pending'])
                elif point == (360, 700):
                    self.screen = 'main'
                else:
                    raise AssertionError(point)
            def stuck_record_add(self, _):
                pass
        root = Path(self.temp.name)
        for folder in ('parallel-skills', 'parallel', 'equipment'):
            shutil.copytree(ROOT/'assets'/folder, root/'assets'/folder)
        config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        config.update(rules=[dict(RULE)], poll_interval=.1)
        (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
        app = Clicker(root, Device)
        app.digit_reader = self.reader
        apps.append(app)
        try:
            with patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                 patch('parallel_skills.time', SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 100000+clock[0])):
                app.start(preview=False, timeout=200)
                app.thread.join(12)
            self.assertFalse(app.running())
            self.assertIsNone(app.state['last_error'])
            self.assertEqual(actions, [(45, 1166), (360, 700)])
            self.assertEqual(pending, [True])
            self.assertEqual(app.state['applied_revision'], 1)
            self.assertEqual(app.state['active_rule_ids'], [])
        finally:
            app.stop()
            if app.thread:
                app.thread.join(3)
            for handler in list(app.log.handlers):
                handler.close()
                app.log.removeHandler(handler)

    def test_migration_preserves_existing_disabled_feature(self):
        config = {'rules': [dict(RULE, enabled=False, interval=55, growth_points=False)]}
        self.assertEqual(add(config), config)


if __name__ == '__main__':
    unittest.main()
