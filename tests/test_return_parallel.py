"""Offline return-to-parallel contract tests; every input is a mock."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from drift.tasks.features import FEATURE_BY_ID, feature_states
from tests import test_clicker as helpers


ROOT = Path(__file__).resolve().parents[1]


def load_frame(name):
    raw = cv2.imread(str(ROOT / 'tests/fixtures/parallel' / (name + '.png')))
    if raw is None:
        raise AssertionError('Missing local fixture: ' + name)
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


class ReturnParallelTests(unittest.TestCase):
    def setUp(self):
        # Keep each red case visible when the implementation does not exist yet.
        self.cls = importlib.import_module('drift.tasks.return_parallel').ReturnParallel
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.now = 100.
        self.path = Path(self.temp.name) / 'return.json'
        self.actions, self.events = [], []
        self.task = self.new_task()
        self.frame = np.zeros((1280, 720, 3), np.uint8)
        for name, value in dict(_parallel=False, _map=False, main=True, moving=False,
                                _battle=False, _off_icon=True, _observatory=False,
                                _paid_prompt=False, _gate_manager=None, _gate_active=True,
                                _npc_ready=True, _enter_question=False,
                                _search_question=False, _map_target=None).items():
            setattr(self.task, name, Mock(return_value=value))

    def new_task(self):
        return self.cls(ROOT, lambda a: self.actions.append(deepcopy(a)) or True,
                        self.events.append, state_path=self.path, clock=lambda: self.now)

    def step(self, frame=None, advance=2):
        self.now += advance
        before = len(self.actions)
        owned = self.task.step(self.frame.copy() if frame is None else frame)
        self.assertLessEqual(len(self.actions) - before, 1, 'At most one input per frame')
        return owned

    def phase(self, name):
        self.task._set(name)

    def kinds(self):
        return [a['kind'] for a in self.actions]

    def test_already_parallel_finishes_without_input(self):
        self.task._parallel.return_value = True
        self.step()
        self.assertIs(self.task.done, True)
        self.assertFalse(self.task.active)
        self.assertEqual(self.actions, [])
        self.assertIn('返回平行世界已完成', self.events)
        self.step()
        self.assertEqual(self.events.count('返回平行世界已完成'), 1)

    def test_moving_waits_without_opening_map_and_logs_at_most_every_30_seconds(self):
        self.task.moving.return_value = True
        for _ in range(12):
            self.assertFalse(self.step())
        self.assertEqual(self.actions, [])
        self.assertEqual(len(self.events), 1)
        self.step(advance=30)
        self.assertEqual(len(self.events), 2)
        self.assertEqual(self.task.phase, 'idle')

    def test_unknown_battle_and_running_auto_do_not_open_map(self):
        for name, value in [('main', False), ('_battle', True), ('_off_icon', False)]:
            with self.subTest(guard=name):
                probe = getattr(self.task, name)
                previous = probe.return_value
                probe.return_value = value
                self.assertFalse(self.step())
                probe.return_value = previous
        self.assertEqual(self.actions, [])

    def test_paid_prompt_only_cancels_in_every_live_phase(self):
        for phase in self.cls.PHASE_TIMEOUTS:
            with self.subTest(phase=phase):
                self.task = self.new_task()
                self.task._paid_prompt = Mock(return_value=True)
                self.task._parallel = Mock(return_value=False)
                self.task._set(phase)
                self.actions.clear()
                self.step()
                self.assertEqual(self.kinds(), ['teleport-cancel'])
                self.assertEqual((self.actions[0]['x'], self.actions[0]['y']), (540, 833))
                self.assertFalse(any('confirm' in k for k in self.task.status()['attempts']))
                self.assertTrue(any('放弃了代价传送' in text for text in self.events))
                self.path.unlink()

    def test_map_found_after_second_pan_selects_once_then_closes_once(self):
        self.step()  # idle -> map_open, possibly the opening input
        self.task._map.return_value = True
        self.step()
        self.assertEqual(self.task.phase, 'map_find')
        self.task._map_target.side_effect = [None, None, (210, 765, .98)]
        self.step(); self.step(); self.step(); self.step()
        self.assertEqual(self.kinds().count('map-pan-left'), 2)
        self.assertEqual(self.kinds().count('map-select'), 1)
        self.assertEqual(self.kinds().count('map-close'), 1)
        pans = [a for a in self.actions if 'swipe' in a]
        self.assertTrue(all((a['x'], a['y'], *a['swipe'][:2]) == (300, 640, 690, 640)
                            for a in pans))
        self.assertEqual(self.task.phase, 'travel')

    def test_full_six_plus_six_scan_stops_without_selecting(self):
        self.phase('map_find'); self.task._map.return_value = True
        for _ in range(13):
            self.step()
        self.assertIs(self.task.done, False)
        self.assertFalse(self.task.active)
        self.assertEqual(self.task.status()['failure'], '地图上找不到传送观测所')
        self.assertEqual(self.kinds().count('map-pan-left'), 6)
        self.assertEqual(self.kinds().count('map-pan-right'), 6)
        self.assertTrue(all('swipe' in a for a in self.actions))
        right = [a for a in self.actions if a['kind'] == 'map-pan-right']
        self.assertTrue(all((a['x'], a['y'], *a['swipe'][:2]) == (690, 640, 300, 640)
                            for a in right))
        self.assertEqual(json.loads(self.path.read_text(encoding='utf-8'))['done'], False)
        old = len(self.actions)
        for _ in range(10): self.step()
        self.assertEqual(len(self.actions), old)

    def test_closed_map_reopens_without_resetting_scan_budget(self):
        self.phase('map_find'); self.task._map.return_value = True
        self.step()
        self.task._map.return_value = False
        self.step()
        self.assertEqual(self.task.phase, 'map_open')
        self.assertEqual(self.kinds(), ['map-pan-left'])
        self.step()
        self.assertEqual(self.kinds()[-1], 'map-open')
        self.task._map.return_value = True
        self.step(); self.step()
        self.assertEqual(self.task.status()['attempts']['map-pan-left'], 2)

    def test_arrival_then_npc_enter_and_exact_confirmation(self):
        self.phase('travel'); self.task._observatory.return_value = True
        for _ in range(3): self.step()
        self.assertEqual(self.task.phase, 'npc')
        self.step()
        self.assertEqual(self.task.phase, 'enter')
        self.step()
        self.assertEqual(self.kinds(), ['enter'])
        self.assertEqual((self.actions[-1]['x'], self.actions[-1]['y']), (354, 482))
        self.assertEqual(self.task.phase, 'enter_confirm')
        self.step()  # No exact question, no generic green-button confirmation.
        self.assertNotIn('enter-confirm', self.kinds())
        self.task._enter_question.return_value = True
        self.step()
        self.assertEqual(self.kinds()[-1], 'enter-confirm')
        self.assertEqual((self.actions[-1]['x'], self.actions[-1]['y']), (208, 787))
        self.assertEqual(self.task.phase, 'arrive')

    def test_travel_requires_stopped_main_and_three_distinct_arrival_frames(self):
        self.phase('travel'); self.task._observatory.return_value = True
        self.task.moving.return_value = True
        for _ in range(4): self.step()
        self.assertEqual(self.task.phase, 'travel')
        self.task.moving.return_value = False
        for _ in range(4): self.step(frame=self.frame)
        self.assertEqual(self.task.phase, 'travel')
        self.step(); self.step()
        self.assertEqual(self.task.phase, 'npc')

    def test_travel_diagnostics_are_throttled(self):
        self.phase('travel')
        self.events.clear()
        for _ in range(12): self.step()
        self.assertEqual(len(self.events), 1)
        self.assertTrue(all(word in self.events[0] for word in ('主界面=', '移动中=', '地名=', '确认帧数=')))
        self.step(advance=30)
        self.assertEqual(len(self.events), 2)

    def test_active_manager_preserves_existing_gate_and_closes_panel(self):
        self.phase('npc'); self.task._observatory.return_value = True
        self.task._gate_manager.return_value = True
        self.step(); self.step(); self.step()
        self.assertEqual(self.kinds(), ['panel-close'])
        self.assertEqual(self.task.phase, 'enter')
        self.assertNotIn('close-gate', self.task.status()['attempts'])

    def test_inactive_manager_activates_searches_and_enters_new_gate(self):
        self.phase('npc'); self.task._observatory.return_value = True
        self.task._gate_manager.return_value = False
        self.task._gate_active.return_value = False
        self.step(); self.step()
        self.assertEqual(self.kinds(), ['activate'])
        self.task._search_question.return_value = True
        self.step()
        self.assertEqual(self.kinds()[-1], 'search-confirm')
        self.assertEqual(self.task.phase, 'new_gate')
        self.task._search_question.return_value = False
        self.task._gate_manager.return_value = True
        self.step(); self.step()
        self.assertEqual(self.kinds()[-1], 'panel-close')
        self.assertEqual(self.task.phase, 'enter')

    def test_parallel_appearance_completes_pending_journal(self):
        self.phase('arrive'); self.task._parallel.return_value = True
        self.step()
        self.assertIs(self.task.done, True)
        self.assertEqual(self.actions, [])
        saved = json.loads(self.path.read_text(encoding='utf-8'))
        self.assertFalse(saved['pending'])
        self.assertTrue(saved['done'])

    def test_every_phase_times_out_and_retains_failure_without_throwing(self):
        for phase, timeout in self.cls.PHASE_TIMEOUTS.items():
            with self.subTest(phase=phase):
                self.task = self.new_task()
                self.task._parallel = Mock(return_value=False)
                self.task._paid_prompt = Mock(return_value=False)
                self.task._set(phase)
                self.now = self.task.status()['phase_deadline'] + 1
                self.actions.clear()
                self.step()
                self.assertIs(self.task.done, False)
                self.assertFalse(self.task.active)
                self.assertIn(phase, self.task.status()['failure'])
                self.assertEqual(self.actions, [])
                self.assertEqual(self.new_task().done, False)
                self.path.unlink()

    def test_reentering_phase_and_restarting_do_not_renew_deadlines_or_attempts(self):
        self.phase('map_find'); self.task._map.return_value = True
        self.step()
        before = self.task.status()
        self.now += 10
        self.task._set('map_find')
        resumed = self.new_task()
        self.assertEqual(resumed.status()['phase_deadline'], before['phase_deadline'])
        self.assertEqual(resumed.status()['deadline'], before['deadline'])
        self.assertEqual(resumed.status()['attempts'], before['attempts'])
        self.assertTrue(resumed.active)

    def test_enter_inputs_are_bounded_even_if_confirmation_repeats(self):
        self.phase('enter'); self.task._observatory.return_value = True
        for _ in range(12): self.step()
        self.assertLessEqual(self.kinds().count('enter'), 2)
        self.assertNotIn('enter-confirm', self.kinds())
        self.assertIs(self.task.done, False)

    def test_disabled_task_is_interruptible_and_preserves_pending_record(self):
        self.phase('map_find'); self.task._map.return_value = True
        self.task.enabled = lambda: False
        before = self.path.read_bytes()
        self.assertFalse(self.step())
        self.assertEqual(self.actions, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_fresh_paid_dialog_vetoes_stale_enter_confirmation(self):
        self.phase('enter_confirm')
        self.task._enter_question.return_value = True
        self.task._paid_prompt.return_value = True
        action = dict(kind='enter-confirm', x=208, y=787, phase='enter_confirm')
        self.assertFalse(self.task.validate(action, self.frame))
        self.step()
        self.assertEqual(self.kinds(), ['teleport-cancel'])

    def test_paid_popup_keeps_frame_ownership_during_input_cooldown(self):
        self.task._paid_prompt.return_value = True
        self.assertTrue(self.step())
        self.assertTrue(self.step(advance=.1))
        self.assertEqual(self.kinds(), ['teleport-cancel'])

    def test_expired_arrival_cannot_be_marked_success(self):
        self.phase('arrive')
        self.task._parallel.return_value = True
        self.now = self.task.status()['phase_deadline'] + 1
        self.step()
        self.assertIs(self.task.done, False)
        self.assertEqual(self.actions, [])

    def test_actual_templates_detect_world_map_location_and_enter_question(self):
        task = self.new_task()
        self.assertTrue(task._parallel(load_frame('parallel-main')))
        self.assertTrue(task._map(load_frame('parallel-map-rank2')))
        self.assertTrue(task._observatory(load_frame('portal-active-main')))
        self.assertTrue(task._enter_question(load_frame('parallel-enter')))
        self.assertFalse(task._paid_prompt(load_frame('parallel-enter')))
        self.assertFalse(task.normal_idle(load_frame('android-home')))
        self.assertFalse(task.normal_idle(load_frame('quest-battle-rank3')))

    def test_full_marker_required_at_094_without_edge_or_blank_false_match(self):
        task = self.new_task()
        marker = task.assets['map-observatory']
        h, w = marker.shape[:2]
        frame = np.zeros_like(self.frame)
        self.assertIsNone(task._map_target(frame))
        frame[740:740+h, :w//2] = marker[:, w-w//2:]
        self.assertIsNone(task._map_target(frame))
        frame[740:740+h, 200:200+w] = marker
        found = task._map_target(frame)
        self.assertEqual(found[:2], (200+w//2, 740+h//2))
        self.assertGreaterEqual(found[2], .94)

    def test_real_paid_prompt_glyphs_override_any_enter_question(self):
        task = self.new_task()
        # Existing currency/cancel/continue assets; no cost numeral is assumed.
        frame = np.full_like(self.frame, 190)
        for name, (x, y) in {'teleport-currency': (145, 430),
                              'teleport-continue': (140, 805),
                              'teleport-cancel': (460, 805)}.items():
            raw = cv2.imread(str(ROOT / 'assets/guild-tasks' / (name + '.png')))
            crop = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            h, w = crop.shape[:2]
            frame[y:y+h, x:x+w] = crop
        self.assertTrue(task._paid_prompt(frame))
        task.step(frame)
        self.assertEqual(self.kinds(), ['teleport-cancel'])


class ReturnParallelFeatureTests(unittest.TestCase):
    setUp = helpers.FrameworkTests.setUp
    tearDown = helpers.FrameworkTests.tearDown
    rule = helpers.FrameworkTests.rule
    wait_for = helpers.FrameworkTests.wait_for

    def test_feature_defaults_off_even_with_no_rules(self):
        self.assertIn('return_parallel', FEATURE_BY_ID)
        feature = next(f for f in feature_states({'rules': []}) if f['id'] == 'return_parallel')
        self.assertEqual(feature['rules'], ())
        self.assertTrue(feature['available'])
        self.assertFalse(feature['enabled'])
        self.assertEqual(next(f for f in self.app.feature_status() if f['id'] == 'return_parallel')
                         ['execution']['status'], 'disabled')

    def test_explicit_switch_persists_and_preserves_other_rules(self):
        self.rule()
        before = deepcopy(self.app.config['rules'])
        self.assertTrue(self.app.change_feature({'id': 'return_parallel', 'enabled': True})['enabled'])
        saved = json.loads((self.root / 'config.json').read_text(encoding='utf-8'))
        self.assertIs(saved['return_parallel_enabled'], True)
        self.assertEqual(saved['rules'], before)
        self.assertFalse(self.app.change_feature({'id': 'return_parallel', 'enabled': False})['enabled'])
        self.assertEqual(self.app.config['rules'], before)

    def test_switch_validation_and_failed_persistence_are_atomic(self):
        before = deepcopy(self.app.config)
        with self.assertRaises(ValueError):
            self.app.change_feature({'id': 'return_parallel', 'enabled': 'true'})
        with patch.object(self.app, 'persist', side_effect=OSError('full')):
            with self.assertRaises(OSError):
                self.app.change_feature({'id': 'return_parallel', 'enabled': True})
        self.assertEqual(self.app.config, before)

    def test_disabled_and_preview_workers_never_construct_controller(self):
        rule = self.rule(); rule['threshold'] = 1
        with patch('drift.runtime.ReturnParallel') as constructor:
            self.app.start(preview=False)
            self.wait_for(lambda: self.app.state['frames'] >= 2)
            self.app.stop(); self.app.thread.join(3)
            constructor.assert_not_called()
            self.app.change_feature({'id': 'return_parallel', 'enabled': True})
            self.app.start(preview=True)
            self.wait_for(lambda: self.app.state['frames'] >= 2)
            self.app.stop(); self.app.thread.join(3)
            constructor.assert_not_called()

    def test_worker_runs_with_only_return_feature_and_publishes_status(self):
        self.app.change_feature({'id': 'return_parallel', 'enabled': True})
        task = Mock(active=True, done=None, phase='travel')
        task.step.return_value = True
        task.status.return_value = dict(phase='travel', hold='等待航行', target='传送观测所',
                                        deadline=700, attempts={})
        with patch('drift.runtime.ReturnParallel', return_value=task) as constructor:
            self.app.start(preview=False)
            self.wait_for(lambda: task.step.call_count >= 2)
            self.assertEqual(self.app.status()['return_parallel']['phase'], 'travel')
            self.assertEqual(self.app.state['stage'], 'return_parallel_travel')
            self.app.change_feature({'id': 'return_parallel', 'enabled': False})
            self.wait_for(lambda: self.app.state['applied_revision'] == self.app.config_revision)
            self.assertEqual(constructor.call_count, 1)
            self.assertEqual(self.app.device.taps, [])
            self.assertIsNone(self.app.state['last_error'])


if __name__ == '__main__':
    unittest.main()
