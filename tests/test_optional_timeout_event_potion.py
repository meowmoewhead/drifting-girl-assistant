"""Offline event-potion abandonment: real templates, fake device and clock."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from drift.core.matcher import scan_rules
from drift.core.session import LoopState
from drift.runtime import Clicker
from drift.tasks.frame_runtime import FrameRuntime
from drift.tasks.optional_timeout import OptionalCleanup, apply_policy


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'driftinggirl_event_potion_'
CROSS = [583, 390, 650, 446]


class EventPotionTimeoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'config.json').read_text('utf-8-sig'))
        cls.config['rules'] = [r for r in cls.config['rules'] if r['id'].startswith(PREFIX)]
        for rule in cls.config['rules']:
            rule['enabled'] = True
        cls.rules = {r['id']: r for r in cls.config['rules']}
        cls.frames = {}
        for suffix in ('', '-candy', '-ghost'):
            for name in ('main', 'shop', 'confirm'):
                with Image.open(ROOT / 'assets/event-potion' / (name + suffix + '.png')) as source:
                    cls.frames[name + suffix] = np.array(source.convert('RGB'))

    def load(self, rule):
        return Clicker.load_rule(SimpleNamespace(root=ROOT, config=self.config), deepcopy(rule))

    def matchers(self):
        return {rid: self.load(rule) for rid, rule in self.rules.items()}

    def prepared(self, stage):
        cleanup = OptionalCleanup(stage, 0, original_deadline=-1, original_trigger='test')
        matchers = self.matchers()
        actions = cleanup.prepare(list(matchers.values()), self.load)
        matchers.update({r['id']: (r, b, o) for r, b, o in actions})
        return cleanup, actions, matchers

    def test_policy_only_enables_open_select_confirm_and_respects_explicit_false(self):
        for step in ('open', 'select', 'confirm', 'close', 'return'):
            with self.subTest(step=step):
                original = deepcopy(self.rules[PREFIX + step])
                self.assertEqual(bool(apply_policy(original).get('skip_on_timeout')),
                                 step in ('open', 'select', 'confirm'))
                self.assertEqual(original, self.rules[PREFIX + step])
                self.assertFalse(apply_policy(dict(original, skip_on_timeout=False))['skip_on_timeout'])
        for field in ('purchase_condition', 'quantity_condition', 'confirmation_condition'):
            rule = dict(self.rules[PREFIX + 'confirm'], **{field: {'cost': 500}})
            self.assertFalse(apply_policy(rule).get('skip_on_timeout', False))

    def test_shop_cleanup_uses_card_evidence_and_only_the_existing_red_cross(self):
        cleanup, actions, _ = self.prepared('event_potion_shop')
        self.assertTrue(actions)
        shop_rule = next(r for r, _, _ in actions if r['asset'] == 'assets/event-potion/shop.png')
        self.assertEqual(shop_rule['source_crop'], [143, 621, 206, 694])
        self.assertEqual(shop_rule['search_area'], [112, 580, 235, 716])
        self.assertEqual(shop_rule['threshold'], .94)
        for suffix in ('', '-candy', '-ghost'):
            matched = [(r, b) for r, b, o in actions
                       if b.match(self.frames['shop' + suffix], offset=o, similarity=r['threshold'])]
            self.assertTrue(matched, suffix)
            for rule, button in matched:
                self.assertEqual(list(button.button), CROSS)
                self.assertEqual(rule['click_source_area'], CROSS)
                self.assertEqual((rule['stage'], rule['next_stage'], rule['action']),
                                 ('optional_cleanup', 'optional_cleanup', 'click'))
                self.assertEqual(rule['cleanup_origin'], PREFIX + 'close')
                self.assertGreaterEqual(rule['stable_frames'], 3)
                self.assertEqual(rule['next_delay'], .8)
                for flag in ('recheck_before_click', 'timeout_grace', 'skip_on_timeout'):
                    self.assertNotIn(flag, rule)
        self.assertEqual(cleanup.original_deadline, -1)
        self.assertEqual(self.rules[PREFIX + 'close']['absent_rule'], PREFIX + 'confirm')

    def test_close_and_confirm_cleanup_keep_close_geometry_and_can_cancel_modal(self):
        for stage in ('event_potion_confirm', 'event_potion_close'):
            cleanup, actions, _ = self.prepared(stage)
            shop_rule = next(r for r, _, _ in actions if r['asset'] == 'assets/event-potion/shop.png')
            self.assertEqual(shop_rule['source_crop'], [557, 521, 597, 564])
            self.assertEqual(shop_rule['search_area'], [517, 485, 627, 591])
            for suffix in ('', '-candy', '-ghost'):
                frame = self.frames['confirm' + suffix]
                matched = [(r, b) for r, b, o in actions
                           if b.match(frame, offset=o, similarity=r['threshold'])]
                self.assertTrue(matched, (stage, suffix))
                rule, button = matched[0]
                self.assertEqual(list(button.button), CROSS)
                self.assertNotIn('absent_rule', rule)
                cleanup.reset_input_proof()
                self.assertFalse(cleanup.ready(rule['id'], button, frame.copy(), 1))
                self.assertFalse(cleanup.ready(rule['id'], button, frame.copy(), 2))
                self.assertTrue(cleanup.ready(rule['id'], button, frame.copy(), 4))

    def test_altered_close_geometry_or_assets_cannot_authorize_cleanup(self):
        for field, value in (
                ('asset', 'assets/event-potion/confirm.png'),
                ('source_crop', [558, 521, 597, 564]),
                ('search_area', [518, 485, 627, 591]),
                ('click_source_area', [95, 761, 321, 821]),
                ('asset_variants', ['assets/mana-confirm-native.png'])):
            with self.subTest(field=field):
                rules = deepcopy(list(self.rules.values()))
                next(r for r in rules if r['id'] == PREFIX + 'close')[field] = value
                cleanup = OptionalCleanup('event_potion_shop', 0)
                loader = Mock(side_effect=lambda r: (r, None, None))
                self.assertEqual(cleanup.prepare([(r, None, None) for r in rules], loader), [])
                loader.assert_not_called()

    def test_unopened_shop_returns_on_three_fresh_frames_and_three_seconds_without_footer(self):
        for suffix in ('', '-candy', '-ghost'):
            cleanup, _, matchers = self.prepared('event_potion_shop')
            frame = self.frames['main' + suffix].copy()
            frame[1140:1200] = 0
            frame[10:62, 72:126] = 0
            self.assertFalse(cleanup.observe_return(frame, matchers, 1))
            self.assertFalse(cleanup.observe_return(frame, matchers, 5))  # cached frame
            self.assertFalse(cleanup.observe_return(frame.copy(), matchers, 5.1))
            self.assertTrue(cleanup.observe_return(frame.copy(), matchers, 5.2))
            cleanup, _, matchers = self.prepared('event_potion_shop')
            for now in (1, 2, 3.9):
                self.assertFalse(cleanup.observe_return(frame.copy(), matchers, now))
            self.assertTrue(cleanup.observe_return(frame.copy(), matchers, 4))

    def test_each_blocker_resets_return_proof_even_when_disabled(self):
        for blocker in ('select', 'confirm', 'close'):
            cleanup, _, matchers = self.prepared('event_potion_shop')
            frame = self.frames['main']
            self.assertFalse(cleanup.observe_return(frame.copy(), matchers, 1))
            rule, _, offset = matchers[PREFIX + blocker]
            button = Mock()
            button.match.return_value = True
            matchers[PREFIX + blocker] = (dict(rule, enabled=False), button, offset)
            self.assertFalse(cleanup.observe_return(frame.copy(), matchers, 4))
            button.match.return_value = False
            self.assertFalse(cleanup.observe_return(frame.copy(), matchers, 5))
            self.assertFalse(cleanup.observe_return(frame.copy(), matchers, 6))
            self.assertTrue(cleanup.observe_return(frame.copy(), matchers, 8))

    def test_missing_positive_or_blocker_unknown_frame_and_deadline_cannot_prove_return(self):
        for missing in ('open', 'select', 'confirm', 'close'):
            cleanup, _, matchers = self.prepared('event_potion_return')
            matchers.pop(PREFIX + missing)
            for now in (1, 2, 4):
                self.assertFalse(cleanup.observe_return(self.frames['main'].copy(), matchers, now))
        cleanup, _, matchers = self.prepared('event_potion_close')
        for now in (1, 2, 4):
            self.assertFalse(cleanup.observe_return(np.zeros_like(self.frames['main']), matchers, now))
        self.assertFalse(cleanup.observe_return(self.frames['main'].copy(), matchers, 40))

    def replay(self, stage='main', scene='main', *, repeated=False):
        config = deepcopy(self.config)
        config['poll_interval'] = .1
        clock = [100.]
        stages, scans = [], []
        frames = self.frames
        state = LoopState(stage=stage, stage_deadline=99. if stage != 'main' else None)
        if stage != 'main':
            trigger = {'event_potion_shop': 'open', 'event_potion_confirm': 'select',
                       'event_potion_close': 'confirm'}[stage]
            state.stage_trigger = self.rules[PREFIX + trigger]['name']
            state.skip_on_timeout = bool(apply_policy(self.rules[PREFIX + trigger]).get('skip_on_timeout'))

        class Device:
            def __init__(self, config, stop):
                self.stop, self.image, self.taps, self.reads = stop, None, [], 0
                self.scene = scene

            def screenshot(self):
                clock[0] += 1
                self.reads += 1
                stages.append(app.state['stage'])
                if self.reads > (230 if repeated else 45):
                    raise InterruptedError('end of bounded offline replay')
                self.image = frames[self.scene].copy()
                return self.image

            def click(self, button):
                self.taps.append((button.name, list(button.button)))
                if button.name == PREFIX + 'open' and repeated:
                    return  # Battle ignores the open click.
                if not button.name.endswith('_cleanup') or list(button.button) != CROSS:
                    raise AssertionError('Cleanup may only tap the existing red cross')
                self.scene = 'shop' if self.scene == 'confirm' else 'main'

            def stuck_record_add(self, _):
                pass

        def scan(frame, candidates, *args, **kwargs):
            scans.append((kwargs['stage'], [r['id'] for r, _, _ in candidates]))
            return scan_rules(frame, candidates, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copytree(ROOT / 'assets/event-potion', root / 'assets/event-potion')
            (root / 'config.json').write_text(json.dumps(config), encoding='utf-8')
            app = Clicker(root, Device)
            app.device = Device(config, app.stop_event)
            try:
                with patch('drift.runtime.LoopState', return_value=state), \
                     patch('app.time', SimpleNamespace(monotonic=lambda: clock[0])), \
                     patch('alas_core.timer.time', side_effect=lambda: clock[0]), \
                     patch.object(FrameRuntime, 'start_engine'), \
                     patch.object(FrameRuntime, 'wait', return_value=False), \
                     patch('drift.runtime.scan_rules', side_effect=scan), \
                     patch.object(app, 'number_condition', side_effect=lambda rule, *_:
                                  (True, 1000 if rule.get('number_condition') else None)):
                    app.run([app.load_rule(r) for r in config['rules']], False,
                            stop_on_main=not repeated)
                return deepcopy(app.state), app.device.taps, stages, scans, list(app.events)
            finally:
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_worker_unopened_shop_timeout_enters_cleanup_then_main_without_error(self):
        state, taps, stages, _, events = self.replay('event_potion_shop')
        self.assertIsNone(state['last_error'])
        self.assertIn('optional_cleanup', stages)
        self.assertEqual(state['stage'], 'main')
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertTrue(state['optional_cleanup']['returned'])
        self.assertEqual(taps, [])
        self.assertTrue(any('跳过本次' in e['message'] for e in events))

    def test_worker_open_shop_confirm_and_close_timeouts_exit_with_only_cross_clicks(self):
        for stage, scene in (('event_potion_shop', 'shop'), ('event_potion_confirm', 'confirm'),
                             ('event_potion_close', 'shop'), ('event_potion_close', 'confirm')):
            with self.subTest(stage=stage, scene=scene):
                state, taps, stages, _, _ = self.replay(stage, scene)
                self.assertIsNone(state['last_error'])
                self.assertIn('optional_cleanup', stages)
                self.assertEqual(state['stage'], 'main')
                self.assertEqual(state['stop_reason'], 'cycle_complete')
                self.assertEqual(len(taps), 2 if scene == 'confirm' else 1)
                self.assertTrue(all(name.endswith('_cleanup') and box == CROSS for name, box in taps))

    def test_worker_three_skips_remove_open_from_scans_for_the_rest_of_each_run(self):
        for _ in range(2):  # Budget resets only on a new run.
            state, taps, _, scans, events = self.replay(repeated=True)
            self.assertIsNone(state['last_error'])
            self.assertEqual([name for name, _ in taps], [PREFIX + 'open'] * 3)
            skipped = [e for e in events if '跳过本次' in e['message']]
            self.assertEqual(len(skipped), 3)
            budget = [e for e in events if '本轮' in e['message'] and '3' in e['message']]
            self.assertEqual(len(budget), 1)
            suppressed = [ids for stage, ids in scans if stage == 'main' and PREFIX + 'open' not in ids]
            self.assertGreater(len(suppressed), 5)
            first = next(i for i, (stage, ids) in enumerate(scans)
                         if stage == 'main' and PREFIX + 'open' not in ids)
            self.assertTrue(all(PREFIX + 'open' not in ids for _, ids in scans[first:]))


if __name__ == '__main__':
    unittest.main()
