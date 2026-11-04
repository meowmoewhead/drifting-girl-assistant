"""Native port routing and once-only fishing-bag sale with locked materials."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from backpack_sale import BackpackSale
from app import Clicker
from digit_ocr import DigitReader, CounterReading
from boss_training import BossTraining
from parallel_world import ParallelWorld
from tools.configure_backpack_sale import update, RULE

ROOT = Path(__file__).resolve().parents[1]


def frame(name):
    return np.asarray(Image.open(ROOT/'tests/fixtures/backpack'/f'{name}.png').convert('RGB'))


class SaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'bag.json'
        self.actions, self.events = [], []
        self.sale = BackpackSale(ROOT, self.click, self.events.append, self.reader, state_path=self.path)
        self.current = None
        self.reject = False

    def tearDown(self):
        self.temp.cleanup()

    def click(self, x, y, label, action):
        if self.reject or not self.sale.validate(action, self.current, x, y):
            return False
        self.actions.append((action, x, y))
        return True

    def step(self, name, count=1, rank=37):
        for _ in range(count):
            self.current = frame(name) if isinstance(name, str) else name
            self.sale.next_action = 0
            result = self.sale.step(self.current, rank)
        return result

    def reach_bag(self):
        self.step('before', 3)
        self.step('port-map')
        self.step('port-selected')
        self.step('port-travel', 3)
        self.assertEqual(self.sale.phase, 'bag')

    def test_real_capacity_full_and_after_sale_need_consistent_confident_views(self):
        self.assertEqual(self.sale.capacity(frame('before')), (30, 30))
        self.assertEqual(self.sale.capacity(frame('sold-stable')), (4, 30))
        class Unreadable:
            def read_counter(self, _):
                return CounterReading('30/30', 30, 30, .7)
        self.sale.reader = Unreadable()
        self.assertIsNone(self.sale.capacity(frame('before')))

    def test_full_bag_waits_for_navigation_handoff_before_any_input(self):
        self.sale.before_route = lambda _: False
        self.step('before', 3)
        self.assertEqual(self.sale.phase, 'prepare')
        self.assertEqual(self.actions, [])
        self.assertTrue(json.loads(self.path.read_text())['pending'])
        self.step('before', 2)
        self.assertEqual(self.actions, [])
        self.sale.before_route = lambda _: True
        self.step('before')
        self.assertEqual(self.actions, [('map_open', 350, 30)])

    def test_resume_waits_for_stop_confirmation_and_retains_checkpoint_on_timeout(self):
        self.step('before', 3)
        saved = self.path.read_bytes()
        self.actions.clear()
        self.sale = BackpackSale(ROOT, self.click, self.events.append, self.reader, state_path=self.path)
        self.sale.before_route = lambda _: False
        self.step('port-map', 2)
        self.assertEqual(self.actions, [])
        self.sale.since -= 46
        with self.assertRaisesRegex(RuntimeError, '等待停航超时'):
            self.step('port-map')
        self.assertEqual(self.path.read_bytes(), saved)

    def test_static_anchor_and_header_allow_port_bag_despite_trade_animation(self):
        port = frame('sailing-stopped-full-port')
        obscured = port.copy()
        obscured[1048:1132, 632:718] = 0
        self.assertTrue(self.sale.port(obscured))
        self.step(obscured, 3)
        self.assertFalse(self.sale.validate('map_open', port, 350, 30))
        self.step(obscured, 2)
        self.assertEqual(self.actions, [('bag_open',674,1174)])

    def test_header_or_trade_icon_without_central_anchor_cannot_open_port_bag(self):
        port=frame('sailing-stopped-full-port')
        self.assertTrue(self.sale.dock(port))
        for box in ((248,33,295,82),(310,954,416,1064),(539,1137,600,1186)):
            altered=port.copy();x,y,X,Y=box;altered[y:Y,x:X]=0
            altered[1048:1132,632:718]=0
            self.assertFalse(self.sale.dock(altered))
            self.assertFalse(self.sale.validate('bag_open',altered,674,1174))
        self.assertFalse(self.sale.dock((port*.5).astype(np.uint8)))

    def test_port_bag_needs_explicit_top_off_icon_when_lower_auto_is_covered(self):
        bag=frame('port-bag')
        self.sale.parallel=True
        self.assertTrue(self.sale.stopped_panel(bag))
        for box in ((194,4,246,42),(186,34,253,65),(248,33,295,82),(10,43,42,75)):
            altered=bag.copy();x,y,X,Y=box;altered[y:Y,x:X]=0
            self.assertFalse(self.sale.stopped_panel(altered))
        self.sale.parallel=False
        self.assertFalse(self.sale.stopped_panel(bag))

    def test_actual_rank39_port_trade_rotations_keep_static_dock_guard(self):
        for name in ('rank39-rotating-trade-1','rank39-rotating-trade-2','rank39-rotating-trade-3'):
            with self.subTest(name=name):
                current=frame(name)
                self.assertTrue(self.sale.dock(current))
                self.assertTrue(self.sale.validate('bag_open',current,674,1174))

    def test_three_full_frames_and_rank_40_priority(self):
        self.step('before', 2)
        self.assertEqual(self.actions, [])
        self.step('sold-closed')
        self.step('before', 2)
        self.assertEqual(self.actions, [])
        self.step('before', rank=40)
        self.assertEqual(self.actions, [])
        self.step('before', 3)
        self.assertEqual(self.actions, [('map_open', 350, 30)])

    def test_restart_after_first_map_open_recovers_original_place_before_leaving(self):
        self.step('before', 3)
        saved = json.loads(self.path.read_text())
        self.assertTrue(saved['pending'])
        self.assertIsNone(saved['origin'])
        self.sale = BackpackSale(ROOT, self.click, self.events.append, self.reader, state_path=self.path)
        self.step('port-map', 2)
        self.assertEqual(self.actions[-1][0], 'port_select')
        self.assertIsNotNone(json.loads(self.path.read_text())['origin'])

    def test_real_full_sale_cycle_returns_using_shifted_landmark(self):
        self.reach_bag()
        self.step('port-bag', 3)
        self.assertEqual(self.actions[-1], ('sell', 620, 866))
        self.assertTrue(json.loads(self.path.read_text())['attempted'])
        self.step('sold-stable', 3)
        self.step('sold-closed', 3)
        self.step('return-map')
        self.assertEqual(self.actions[-1][0], 'origin_select')
        self.assertLess(np.hypot(self.actions[-1][1]-426, self.actions[-1][2]-738), 6)
        self.step('return-training-selected')
        self.step('returned-main', 3)
        self.assertEqual(self.sale.phase, 'idle')
        self.assertEqual(self.sale.cycles, 1)
        self.assertEqual(self.sale.capacity_after, 4)
        self.assertFalse(json.loads(self.path.read_text())['pending'])
        self.assertEqual(sum(a[0] == 'sell' for a in self.actions), 1)

    def test_wrong_tab_dark_panel_and_non_port_never_sell(self):
        for name in ('cooking', 'food', 'bag-help', 'bag-open', 'sale-entry'):
            with self.subTest(name=name):
                self.assertFalse(self.sale.validate('sell', frame(name), 620, 866))
        self.assertTrue(self.sale.validate('sell', frame('port-bag'), 620, 866))
        self.assertFalse(self.sale.validate('sell', (frame('port-bag')*.5).astype('uint8'), 620, 866))
        self.assertFalse(self.sale.validate('sell', frame('sold-stable'), 620, 866))

    def test_missing_or_ambiguous_origin_label_never_selects_a_location(self):
        self.reach_bag()
        unknown = frame('return-map').copy()
        unknown[688:724, 385:462] = 0
        self.assertIsNone(self.sale._origin_target(unknown))
        ambiguous = frame('return-map').copy()
        ambiguous[300:332, 300:440] = self.sale.origin
        self.assertIsNone(self.sale._origin_target(ambiguous))

    def test_fresh_validation_failure_does_not_advance_or_duplicate_sale(self):
        self.reach_bag()
        self.reject = True
        self.step('port-bag', 3)
        self.assertFalse(self.sale.attempted)
        self.assertEqual(sum(a[0] == 'sell' for a in self.actions), 0)
        self.reject = False
        self.step('port-bag', 3)
        self.step('port-bag', 6)
        self.assertEqual(sum(a[0] == 'sell' for a in self.actions), 1)

    def test_only_locked_slots_are_preserved_without_committing(self):
        self.reach_bag()
        locked = frame('port-bag').copy()
        for row in range(3):
            for col in range(10):
                x, y = 28+68*col, 915+68*row
                cv2.rectangle(locked, (x, y), (x+55, y+55), (20, 150, 80), 5)
        self.assertFalse(self.sale.unlocked(locked))
        self.step(locked, 3)
        self.assertFalse(any(a[0] == 'sell' for a in self.actions))
        self.assertEqual(self.sale.phase, 'open_return')

    def test_disable_before_commit_cancels_sale_and_returns(self):
        self.reach_bag()
        self.sale.enabled = lambda: False
        self.step('port-bag')
        self.assertFalse(any(a[0] == 'sell' for a in self.actions))
        self.assertEqual(self.sale.phase, 'open_return')
        self.step('sold-closed', 3)
        self.step('return-map')
        self.step('return-training-selected')
        self.step('returned-main', 3)
        self.assertEqual(self.sale.phase, 'idle')

    def test_restart_after_commit_only_closes_and_returns(self):
        self.reach_bag()
        self.step('port-bag', 3)
        self.actions.clear()
        self.sale = BackpackSale(ROOT, self.click, self.events.append, self.reader, state_path=self.path)
        self.assertEqual(self.sale.phase, 'resume')
        self.step('sold-stable')
        self.step('sold-stable', 3)
        self.step('sold-closed', 3)
        self.step('return-map')
        self.assertFalse(any(a[0] == 'sell' for a in self.actions))
        self.assertEqual(self.actions[-1][0], 'origin_select')

    def test_moving_and_unknown_frames_do_not_start_and_wait_is_bounded(self):
        self.step('port-selected', 4)
        self.assertEqual(self.actions, [])
        self.reach_bag()
        self.sale.since -= 46
        with self.assertRaisesRegex(RuntimeError, '等待超时'):
            self.step(np.zeros((1280, 720, 3), dtype='uint8'))

    def test_corrupt_return_checkpoint_stops_and_config_add_is_isolated(self):
        self.path.write_text('{broken')
        with self.assertRaisesRegex(RuntimeError, '返回记录损坏'):
            BackpackSale(ROOT, self.click, self.events.append, self.reader, state_path=self.path)
        old = {'rules': [{'id': 'old', 'enabled': False, 'interval': 123}], 'settings': {'a': 1}}
        result = update(old)
        self.assertEqual(result['rules'][:-1], old['rules'])
        self.assertEqual(update(result), result)
        self.assertEqual(old['rules'], [{'id': 'old', 'enabled': False, 'interval': 123}])
        result['rules'].append(deepcopy(RULE))
        with self.assertRaises(ValueError):
            update(result)

    def prepare_scan(self, phase='find_port'):
        self.sale.origin = cv2.imread(str(ROOT/'tests/fixtures/backpack/origin-mumbai.png'))
        self.sale.parallel, self.sale.capacity_before = True, 30
        self.sale._set(phase)
        self.sale._save()
        def swipe(*args):
            if not self.sale.validate_scan(self.current):
                return False
            self.actions.append(('pan', *args[:4]))
            return True
        self.sale.swipe = swipe

    def test_real_offscreen_port_requires_three_stopped_frames_and_keeps_saved_origin(self):
        self.prepare_scan()
        saved, since = self.path.read_bytes(), self.sale.since
        self.assertEqual(self.sale._ports(frame('port-offscreen-mumbai')), [])
        self.step('port-offscreen-mumbai', 2, rank=11)
        self.assertEqual(self.actions, [])
        self.step('port-offscreen-mumbai', rank=11)
        self.assertEqual(self.actions, [('pan', 600, 600, 120, 600)])
        self.assertEqual(self.sale.since, since)
        self.assertEqual(self.path.read_bytes(), saved)
        self.step('port-map')
        self.assertEqual(self.actions[-1][0], 'port_select')
        self.assertEqual(json.loads(self.path.read_text())['origin'], json.loads(saved)['origin'])

    def test_pan_rejects_fresh_rank_world_tier_location_and_running_auto_changes(self):
        self.prepare_scan()
        self.step('port-offscreen-mumbai', 3, rank=11)
        reference = frame('port-offscreen-mumbai')
        for box in ((71,9,136,59), (10,43,60,75), (300,12,418,36),
                    (192,2,248,44), (9,1136,405,1241)):
            changed = reference.copy()
            x,y,X,Y = box
            changed[y:Y,x:X] = 0
            with self.subTest(box=box):
                self.assertFalse(self.sale.validate_scan(changed))
        green = reference.copy()
        green[2:44,192:248] = frame('sailing-full-port')[2:44,192:248]
        self.assertFalse(self.sale.validate_scan(green))
        self.assertFalse(self.sale.validate_scan((reference*.5).astype('uint8')))

    def test_missing_map_guards_do_not_pan_or_extend_timeout(self):
        self.prepare_scan()
        since, saved = self.sale.since, self.path.read_bytes()
        unknown = frame('port-offscreen-mumbai').copy()
        unknown[1136:1241,9:405] = 0
        self.step(unknown, 5, rank=11)
        self.assertEqual(self.actions, [])
        self.assertEqual(self.sale.since, since)
        self.sale.since -= 46
        with self.assertRaisesRegex(RuntimeError, '等待超时'):
            self.step('port-offscreen-mumbai', rank=11)
        self.assertEqual(self.path.read_bytes(), saved)

    def test_scan_count_bound_searches_both_directions_without_resetting_phase(self):
        self.prepare_scan()
        saved, since = self.path.read_bytes(), self.sale.since
        self.step('port-offscreen-mumbai', 36, rank=11)
        self.assertEqual(len(self.actions), 12)
        self.assertTrue(all(a[1:] == (600,600,120,600) for a in self.actions[:4]))
        self.assertTrue(all(a[1:] == (120,600,600,600) for a in self.actions[4:]))
        self.assertEqual(self.sale.since, since)
        with self.assertRaisesRegex(RuntimeError, '12次上限'):
            self.step('port-offscreen-mumbai', 3, rank=11)
        self.assertEqual(self.path.read_bytes(), saved)

    def test_missing_origin_can_be_found_after_pan_and_restart_keeps_landmark(self):
        self.prepare_scan('find_origin')
        self.sale.attempted = True
        self.sale._save()
        saved = self.path.read_bytes()
        self.step('port-map', 3)
        self.assertEqual(self.actions[-1][0], 'pan')
        self.assertEqual(self.path.read_bytes(), saved)
        self.step('port-offscreen-mumbai', rank=11)
        self.assertEqual(self.actions[-1][0], 'origin_select')
        original = self.sale.origin.copy()
        self.sale = BackpackSale(ROOT, self.click, self.events.append, self.reader, state_path=self.path)
        self.assertTrue(np.array_equal(self.sale.origin, original))
        self.assertTrue(self.sale.attempted)

    def test_fresh_pan_failure_and_disabled_sale_cannot_advance_search(self):
        self.prepare_scan()
        saved, since = self.path.read_bytes(), self.sale.since
        self.sale.swipe = lambda *args: False
        self.step('port-offscreen-mumbai', 6, rank=11)
        self.assertEqual(self.sale.scan_swipes, 0)
        self.assertEqual(self.actions, [])
        self.assertEqual(self.path.read_bytes(), saved)
        self.assertEqual(self.sale.since, since)
        self.sale.enabled = lambda: False
        self.step('port-offscreen-mumbai', rank=11)
        self.assertTrue(self.sale.attempted)
        self.assertEqual(self.actions[-1][0], 'origin_select')
        self.assertFalse(any(a[0] in ('sell','pan') for a in self.actions))

    def test_real_busan_and_sydney_anchors_exclude_city_names_and_scenery(self):
        self.prepare_scan()
        native = frame('ports-busan-sydney')
        ports = self.sale._ports(native)
        self.assertEqual(len(ports), 2)
        self.assertTrue(any(np.hypot(x-101,y-510)<4 for x,y in ports))
        self.assertTrue(any(np.hypot(x-261,y-1013)<4 for x,y in ports))
        self.current = native
        for point in ports:
            self.assertTrue(self.sale.validate('port_select', native, *point))
        missing = native.copy()
        missing[488:533,80:123] = 0
        missing[991:1036,240:283] = 0
        self.assertEqual(self.sale._ports(missing), [])
        self.assertFalse(self.sale.validate('port_select', missing, *ports[0]))
        self.assertEqual(self.sale._ports(frame('port-offscreen-mumbai')), [])

    def test_corrupt_native_map_never_authorizes_a_pan(self):
        self.prepare_scan()
        self.step('port-offscreen-mumbai', 2, rank=11)
        saved, since = self.path.read_bytes(), self.sale.since
        self.step('corrupt-native-map', 3, rank=11)
        self.assertEqual(self.actions, [])
        self.assertEqual(self.sale.since, since)
        self.assertEqual(self.path.read_bytes(), saved)

    def test_origin_scan_first_reverses_the_last_port_search_direction(self):
        self.prepare_scan()
        saved = self.path.read_bytes()
        self.step('port-offscreen-mumbai', 3, rank=11)
        self.sale._set('find_origin')
        self.step('port-map', 3)
        self.assertEqual(self.actions[-1], ('pan',120,600,600,600))
        self.sale._set('find_port')
        self.step('port-offscreen-mumbai', 15, rank=11)
        self.assertEqual(self.actions[-1], ('pan',120,600,600,600))
        self.sale._set('find_origin')
        self.step('port-map', 3)
        self.assertEqual(self.actions[-1], ('pan',600,600,120,600))
        self.assertEqual(self.path.read_bytes(), saved)

    def test_port_origin_and_map_close_recheck_stopped_auto_in_the_new_frame(self):
        self.prepare_scan()
        port = frame('ports-busan-sydney')
        point = self.sale._ports(port)[0]
        self.assertTrue(self.sale.validate('port_select', port, *point))
        origin = frame('port-offscreen-mumbai')
        origin_point = self.sale._origin_target(origin)
        self.assertTrue(self.sale.validate('origin_select', origin, *origin_point))
        self.assertTrue(self.sale.validate('map_close', frame('port-selected'), 668, 1187))
        for action, image, xy in (('port_select',port,point), ('origin_select',origin,origin_point),
                                  ('map_close',frame('port-selected'),(668,1187))):
            for unknown in (False, True):
                changed = image.copy()
                changed[2:44,192:248] = (0 if unknown else frame('sailing-full-port')[2:44,192:248])
                self.assertFalse(self.sale.validate(action, changed, *xy))


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader = DigitReader(ROOT)

    def run_worker(self, preview=False, max_clicks=None, corrupt=False, stale=False,
                   pan=False, pan_fresh=None):
        images = {name: frame(name) for name in (
            'before', 'port-map', 'port-selected', 'port-travel', 'port-bag',
            'sold-stable', 'sold-closed', 'return-map', 'return-training-selected',
            'returned-main', 'cooking', 'port-offscreen-mumbai')}
        transitions = [('before', 'port-map'), ('port-map', 'port-selected'),
                       ('port-selected', 'port-travel'), ('port-travel', 'port-bag'),
                       ('port-bag', 'sold-stable'), ('sold-stable', 'sold-closed'),
                       ('sold-closed', 'return-map'), ('return-map', 'return-training-selected'),
                       ('return-training-selected', 'returned-main')]
        if pan:
            transitions = transitions[1:]
            images['return-map'] = images['port-offscreen-mumbai']
        outer = self
        class Device:
            def __init__(self, config, stop):
                self.stop, self.io_lock = stop, threading.RLock()
                self.image, self.screen, self.shots, self.actions = None, 'port-offscreen-mumbai' if pan else 'before', 0, []
                self.transition_index, self.injected, self.inside_pan = 0, False, False
            def screenshot(self):
                self.shots += 1
                if self.shots > 200 or preview and self.shots >= 4:
                    self.stop.set()
                self.image = images['cooking' if stale and self.shots == 4 else self.screen].copy()
                if pan_fresh and self.inside_pan and not self.injected:
                    self.injected = True
                    if pan_fresh == 'stop':
                        self.stop.set()
                    elif pan_fresh == 'rank':
                        self.image[9:59,71:136] = images['port-map'][9:59,71:136]
                return self.image
            def click(self, button):
                source, target = transitions[self.transition_index]
                outer.assertEqual(self.screen, source)
                b = button.button
                self.actions.append(((b[0]+b[2])//2, (b[1]+b[3])//2))
                self.screen = target
                self.transition_index += 1
            def command(self, *args):
                outer.assertEqual(self.screen, 'port-offscreen-mumbai')
                outer.assertEqual(args, ('shell','input','swipe','600','600','120','600','480'))
                self.actions.append(('pan',600,600,120,600))
                self.screen = 'port-map'
            def stuck_record_add(self, _):
                pass
        class FastSale(BackpackSale):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                original_swipe = self.swipe
                def swipe(*args):
                    app.device.inside_pan = True
                    try:
                        return original_swipe(*args)
                    finally:
                        app.device.inside_pan = False
                self.swipe = swipe
            def step(self, *args, **kwargs):
                self.next_action = 0
                return super().step(*args, **kwargs)
        with tempfile.TemporaryDirectory(prefix='bag-worker-replay-') as folder:
            root = Path(folder)
            shutil.copytree(ROOT/'assets', root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules'] = [r for r in config['rules'] if r['id'] == RULE['id']]
            config['rules'][0]['enabled'] = True
            config['poll_interval'] = .1
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            if corrupt:
                (root/'bag_sale_state.json').write_text('{invalid', encoding='utf-8')
            elif pan:
                saved = BackpackSale(ROOT, None, lambda _:None, self.reader,
                                     state_path=root/'bag_sale_state.json')
                saved.origin = cv2.imread(str(ROOT/'tests/fixtures/backpack/origin-mumbai.png'))
                saved.parallel, saved.capacity_before = True, 30
                saved._save()
            app = Clicker(root, Device)
            app.digit_reader = self.reader
            before = deepcopy(app.config)
            try:
                with patch('app.BackpackSale', FastSale):
                    app.start(preview=preview, rule_ids=[RULE['id']], stop_on_main=True,
                              max_clicks=max_clicks, timeout=20)
                    app.thread.join(25)
                self.assertFalse(app.running())
                self.assertEqual(app.config, before)
                checkpoint = (root/'bag_sale_state.json').read_text() if (root/'bag_sale_state.json').exists() else None
                return {**deepcopy(app.state), 'events': list(app.events)}, list(app.device.actions), checkpoint
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_production_worker_rechecks_stale_frame_and_completes_full_sale(self):
        state, actions, checkpoint = self.run_worker(stale=True)
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual((len(actions), state['clicks']), (9, 9))
        self.assertEqual(actions[4], (620, 866))
        self.assertEqual(state['backpack']['capacity_after'], 4)
        self.assertFalse(json.loads(checkpoint)['pending'])

    def test_worker_click_limit_preserves_route_for_restart(self):
        state, actions, checkpoint = self.run_worker(max_clicks=1)
        self.assertEqual(state['stop_reason'], 'click_limit')
        self.assertEqual(actions, [(350, 30)])
        self.assertTrue(json.loads(checkpoint)['pending'])
        self.assertFalse(json.loads(checkpoint)['attempted'])

    def test_preview_never_opens_map_or_writes_checkpoint(self):
        state, actions, checkpoint = self.run_worker(preview=True)
        self.assertIsNone(state['last_error'])
        self.assertEqual(actions, [])
        self.assertIsNone(checkpoint)

    def test_corrupt_return_checkpoint_is_reported_by_worker_without_clicking(self):
        state, actions, checkpoint = self.run_worker(corrupt=True)
        self.assertIn('返回记录损坏', state['last_error'])
        self.assertEqual(actions, [])
        self.assertEqual(checkpoint, '{invalid')

    def test_worker_pans_real_offscreen_port_then_sells_once_and_returns(self):
        state, actions, checkpoint = self.run_worker(pan=True, pan_fresh='rank')
        self.assertIsNone(state['last_error'])
        self.assertEqual(state['stop_reason'], 'cycle_complete')
        self.assertEqual(actions[0], ('pan',600,600,120,600))
        self.assertEqual((len(actions), state['clicks']), (9,9))
        self.assertEqual(actions.count((620,866)), 1)
        self.assertFalse(json.loads(checkpoint)['pending'])
        self.assertTrue(any('滑动前世界、RANK' in e['message'] for e in state['events']))

    def test_worker_pan_counts_toward_input_limit_and_preserves_return(self):
        state, actions, checkpoint = self.run_worker(pan=True, max_clicks=1)
        self.assertEqual(state['stop_reason'], 'click_limit')
        self.assertEqual(actions, [('pan',600,600,120,600)])
        self.assertTrue(json.loads(checkpoint)['pending'])
        self.assertFalse(json.loads(checkpoint)['attempted'])

    def test_stop_during_new_pan_screenshot_prevents_input(self):
        state, actions, checkpoint = self.run_worker(pan=True, pan_fresh='stop')
        self.assertIsNone(state['last_error'])
        self.assertEqual(actions, [])
        self.assertTrue(json.loads(checkpoint)['pending'])
        self.assertFalse(json.loads(checkpoint)['attempted'])


class SailingSaleWorkerTests(unittest.TestCase):
    """Replay a real full AUTO port, with explicit off/on confirmation."""
    def run_worker(self, restart=False, fresh_restarted=False, fresh_unknown=False,
                   cold_rank=False, starting_bag=False):
        active = frame('sailing-full-port')
        stopped = frame('sailing-stopped-full-port')
        emptied = stopped.copy()
        emptied[1210:1238, 625:706] = frame('sold-stable')[1210:1238, 625:706]
        active_emptied = emptied.copy()
        active_emptied[4:42, 194:246] = active[4:42, 194:246]
        menu = np.asarray(Image.open(ROOT/'tests/fixtures/training-sailing/menu-three.png').convert('RGB'))
        images = {'active': active, 'stopped': stopped, 'bag': frame('port-bag'),
                  'sold': frame('sold-stable'), 'closed': emptied, 'menu': menu,
                  'resumed': active_emptied}
        # The original bag samples were from a different RANK. A single
        # simulated sale must keep the same current world/RANK and OFF header.
        for name in ('bag','sold'):
            images[name]=images[name].copy()
            images[name][:84,:]=stopped[:84,:]
        taps, observations = [], []
        with tempfile.TemporaryDirectory(prefix='sailing-bag-replay-') as directory:
            root = Path(directory)
            shutil.copytree(ROOT/'assets', root/'assets')
            config = json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
            config['rules'] = [r for r in config['rules'] if r['id'] in
                               ('driftinggirl_parallel_world', RULE['id'])]
            for rule in config['rules']:
                rule['enabled'] = True
            config['poll_interval'] = .1
            (root/'config.json').write_text(json.dumps(config), encoding='utf-8')
            training = BossTraining(ROOT, None, lambda _:None, state_path=root/'training_state.json')
            training.begin(38, 1535, 1604)
            training.set_sailing(True, verified=True)
            if starting_bag:
                training.set_sailing(False)
            if restart:
                (root/'bag_sale_state.json').write_text(json.dumps(dict(
                    version=1, pending=True, parallel=True, origin=None, total=30, attempted=False)))
            outer = self

            class Device:
                def __init__(self, _, stop):
                    self.stop, self.io_lock = stop, threading.RLock()
                    self.screen = 'bag' if starting_bag else 'active'
                    self.shots, self.injected = 0, False
                def screenshot(self):
                    self.shots += 1
                    saved = json.loads((root/'training_state.json').read_text())
                    # A route restarts between the off confirmation and the
                    # fresh bag-input screenshot. No input may use the stale off.
                    if (fresh_restarted and not self.injected and self.screen == 'stopped'
                            and not saved['sailing_enabled'] and getattr(self, 'inside_sale', False)):
                        self.screen, self.injected = 'active', True
                    if self.screen == 'resumed' and saved.get('sailing_verified') or self.shots > 150:
                        self.stop.set()
                    self.image = images[self.screen].copy()
                    if (fresh_unknown and not self.injected and self.screen=='stopped'
                            and not saved['sailing_enabled'] and getattr(self,'inside_sale',False)):
                        self.injected=True
                        self.image[4:42,194:246]=0
                    observations.append((self.screen, saved['sailing_enabled']))
                    return self.image
                def click(self, button):
                    b = button.button
                    point = ((b[0]+b[2])//2, (b[1]+b[3])//2)
                    saved = json.loads((root/'training_state.json').read_text())
                    bag = json.loads((root/'bag_sale_state.json').read_text())
                    if point == (233, 45) and self.screen == 'active':
                        outer.assertTrue(bag['pending'])
                        self.screen = 'stopped'
                    elif point == (674, 1174) and self.screen == 'stopped':
                        outer.assertFalse(saved['sailing_enabled'])
                        self.screen = 'bag'
                    elif point == (620, 866) and self.screen == 'bag':
                        outer.assertFalse(saved['sailing_enabled'])
                        self.screen = 'sold'
                    elif point == (674, 1174) and self.screen == 'sold':
                        self.screen = 'closed'
                    elif point == (233, 45) and self.screen == 'closed':
                        outer.assertFalse(bag['pending'])
                        self.screen = 'menu'
                    elif point == (360, 872) and self.screen == 'menu':
                        outer.assertFalse(bag['pending'])
                        self.screen = 'resumed'
                    else:
                        outer.fail(f'Unexpected native input {point} on {self.screen}')
                    taps.append(point)
                def stuck_record_add(self, _):
                    pass

            class FastSale(BackpackSale):
                def step(self, *args, **kwargs):
                    self.next_action = 0
                    app.device.inside_sale = True
                    try:
                        return super().step(*args, **kwargs)
                    finally:
                        app.device.inside_sale = False

            def create_world(*args, **kwargs):
                world = ParallelWorld(*args, **kwargs)
                world.phase, world.rank, world.training.player_level = 'play', 38, 1544
                if cold_rank:
                    world.phase, world.rank = 'detect', None
                else:
                    world._rank = lambda _:38
                world.training.levels = lambda _, **kwargs:(1544, None)
                step = world.sailing.step
                def immediate(image, wanted):
                    world.sailing.next_at = 0
                    return step(image, wanted)
                world.sailing.step = immediate
                return world

            app = Clicker(root, Device)
            app.digit_reader = DigitReader(ROOT)
            try:
                with patch('app.ParallelWorld', side_effect=create_world), patch('app.BackpackSale', FastSale):
                    # A cold restart adds three real OCR frames before the
                    # original stop/sale/start sequence. This test limit is
                    # separate from the unchanged production phase deadlines.
                    app.start(preview=False, timeout=35 if cold_rank else 20)
                    app.thread.join(40 if cold_rank else 25)
                self.assertFalse(app.running())
                self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['backpack']['cycles'], 1)
                self.assertEqual(app.state['backpack']['capacity_after'], 4)
                self.assertFalse(json.loads((root/'bag_sale_state.json').read_text())['pending'])
                saved = json.loads((root/'training_state.json').read_text())
                self.assertEqual((saved['mode'],saved['rank'],saved['boss_level'],saved['start_level']),
                                 ('farming',38,1604,1535))
                self.assertTrue(saved['sailing_enabled'] and saved['sailing_verified'])
                return taps, list(app.events)
            finally:
                app.stop()
                if app.thread:
                    app.thread.join(3)
                for handler in list(app.log.handlers):
                    handler.close()
                    app.log.removeHandler(handler)

    def test_full_auto_bag_stops_once_sells_once_then_restarts_verified_route(self):
        taps, _ = self.run_worker()
        self.assertEqual(taps, [(233,45),(674,1174),(620,866),(674,1174),(233,45),(360,872)])

    def test_resumed_sale_rejects_new_active_route_during_fresh_bag_input(self):
        taps, events = self.run_worker(restart=True, fresh_restarted=True)
        self.assertEqual(taps[:3], [(233,45),(233,45),(674,1174)])
        self.assertEqual(taps.count((620,866)), 1)
        self.assertTrue(any('航行尚未停稳' in e['message'] for e in events))

    def test_unknown_auto_on_new_bag_frame_rejects_input_then_reconfirms_stopped(self):
        taps,events=self.run_worker(restart=True,fresh_unknown=True)
        self.assertEqual(taps.count((620,866)),1)
        self.assertTrue(any('航行状态未知' in e['message'] for e in events))

    def test_saved_sale_cold_restart_establishes_current_rank_before_panel_input(self):
        taps,events=self.run_worker(restart=True,cold_rank=True)
        self.assertEqual(taps.count((620,866)),1)
        self.assertTrue(any('已连续确认当前平行世界 RANK 38' in e['message'] for e in events))

    def test_saved_open_bag_cold_restart_reads_rank_without_reopening_or_stopping(self):
        taps,events=self.run_worker(restart=True,cold_rank=True,starting_bag=True)
        self.assertEqual(taps[:2],[(620,866),(674,1174)])
        self.assertEqual(taps.count((620,866)),1)
        self.assertTrue(any('已连续确认当前平行世界 RANK 38' in e['message'] for e in events))


class ColdBackpackRankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reader=DigitReader(ROOT)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        root=Path(self.temp.name)
        self.actions=[]
        self.world=ParallelWorld(ROOT,lambda *a:self.actions.append(a),
            lambda *a:self.actions.append(a),lambda _:None,self.reader,
            training_state_path=root/'training.json',fishing_action=lambda *a:False,
            sailing_action=lambda *a:False)
        self.world.training.begin(39,1678,1810)
        self.saved=(root/'training.json').read_bytes()
        self.sale=BackpackSale(ROOT,lambda *a:self.actions.append(a),lambda _:None,
            self.reader,state_path=root/'bag.json')
        self.sale.phase,self.sale.parallel='bag',True
        self.frame=frame('rank39-rotating-trade-1')

    def tearDown(self):
        self.temp.cleanup()

    def test_three_real_stopped_port_frames_establish_rank_without_record_or_input(self):
        self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.assertIsNone(self.world.rank)
        self.assertTrue(self.world.observe_backpack_rank(self.frame,self.sale))
        self.assertEqual((self.world.phase,self.world.rank),('play',39))
        self.assertEqual(self.actions,[])
        self.assertEqual(Path(self.world.training.path).read_bytes(),self.saved)

    def test_reading_running_main_establishes_rank_but_never_proves_off_or_inputs(self):
        active=self.frame.copy()
        active[2:44,192:248]=frame('sailing-full-port')[2:44,192:248]
        self.assertIs(self.world.sailing.running(active),True)
        for _ in range(3):
            self.world.observe_backpack_rank(active,self.sale)
        self.assertEqual(self.world.rank,39)
        self.assertIs(self.world.sailing.running(active),True)
        self.assertEqual(self.actions,[])
        self.assertEqual(Path(self.world.training.path).read_bytes(),self.saved)

    def test_unknown_running_dark_world_or_rank_reject_and_reset_votes(self):
        for box in ((192,2,248,44),(10,43,42,75),(71,9,136,59)):
            self.world.backpack_rank_readings.extend((39,39))
            altered=self.frame.copy();x,y,X,Y=box;altered[y:Y,x:X]=0
            self.assertFalse(self.world.observe_backpack_rank(altered,self.sale))
            self.assertIsNone(self.world.rank)
            self.assertEqual(list(self.world.backpack_rank_readings),[])
        for image in (frame('sailing-full-port'),(self.frame*.5).astype(np.uint8)):
            self.assertFalse(self.world.observe_backpack_rank(image,self.sale))
        self.assertEqual(self.actions,[])

    def test_conflicting_saved_rank_and_rank40_cannot_establish_sale_context(self):
        for value in (38,40,None):
            with patch.object(self.world,'_rank_value',return_value=value):
                for _ in range(3):
                    self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
            self.assertIsNone(self.world.rank)
        self.assertEqual(Path(self.world.training.path).read_bytes(),self.saved)

    def test_new_bag_votes_do_not_change_navigation_or_renew_sale_deadline(self):
        before=self.sale.since
        self.world.phase='find_boss'
        self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.assertEqual(self.world.phase,'find_boss')
        self.world.phase='detect'
        for _ in range(3):
            self.world.observe_backpack_rank(self.frame,self.sale)
        self.assertEqual(self.sale.since,before)
        self.assertEqual(self.actions,[])


class StaleBackpackRankTests(ColdBackpackRankTests):
    def setUp(self):
        super().setUp()
        self.world.training.state['mode']='idle'
        self.world.phase='play';self.world.rank=self.world.target_rank=4
        self.world.latest_rank_frame=np.asarray(Image.open(ROOT/'tests/fixtures/parallel/bag-rank4-reference.png').convert('RGB'))
        self.frame=np.asarray(Image.open(ROOT/'tests/fixtures/parallel/bag-rank5-stopped-map.png').convert('RGB'))
        self.sale.phase='find_port';self.sale.capacity_before=30
        self.sale.origin=self.frame[560:592,280:420].copy()
        self.sale._save()
        self.bag_saved=Path(self.sale.path).read_bytes()

    # The inherited tests describe cold RANK39 state; this class exercises the
    # separate real menu-completion case without modifying that farming setup.
    def test_three_real_stopped_port_frames_establish_rank_without_record_or_input(self):
        self.assertFalse(self.world.sailing_context('bag',self.frame))
        before=self.sale.since
        for index in range(3):
            accepted=self.world.observe_backpack_rank(self.frame.copy(),self.sale)
            self.assertEqual(accepted,index==2)
        self.assertEqual(self.world.rank,5)
        self.assertEqual(self.world.target_rank,4)
        self.assertTrue(self.world.sailing_context('bag',self.frame.copy()))
        self.assertEqual(self.sale.since,before)
        self.assertEqual(self.actions,[])
        self.assertEqual(Path(self.world.training.path).read_bytes(),self.saved)
        self.assertEqual(Path(self.sale.path).read_bytes(),self.bag_saved)

    def test_fresh_bag_input_rejects_missing_or_changed_stage_after_rank_confirmation(self):
        for _ in range(3):self.world.observe_backpack_rank(self.frame.copy(),self.sale)
        self.assertEqual(self.world.rank,5)
        for box in ((40,45,68,71),(43,45,60,72)):
            altered=self.frame.copy();x,y,X,Y=box;altered[y:Y,x:X]=0
            self.assertFalse(self.world.sailing_context('bag',altered))
        self.assertTrue(self.world.sailing_context('bag',self.frame.copy()))
        self.assertEqual(self.actions,[])

    def test_reading_running_main_establishes_rank_but_never_proves_off_or_inputs(self):
        for route in (True,None):
            with patch.object(self.sale,'stopped_panel',return_value=False),patch.object(self.sale,'main',return_value=True),patch.object(self.world.sailing,'running',return_value=route):
                for _ in range(3):self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.assertEqual(self.world.rank,4)

    def test_unknown_running_dark_world_or_rank_reject_and_reset_votes(self):
        for box in ((192,2,248,44),(10,43,42,75),(43,45,60,72),(71,9,136,59)):
            altered=self.frame.copy();x,y,X,Y=box;altered[y:Y,x:X]=0
            for _ in range(3):self.assertFalse(self.world.observe_backpack_rank(altered,self.sale))
        for _ in range(3):self.assertFalse(self.world.observe_backpack_rank((self.frame*.5).astype(np.uint8),self.sale))
        self.assertEqual(self.world.rank,4)
        self.assertEqual(self.actions,[])

    def test_conflicting_saved_rank_and_rank40_cannot_establish_sale_context(self):
        for value in (None,3,4,6,39,40):
            with patch.object(self.world,'_rank_value',return_value=value):
                for _ in range(3):self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.world.training.state['mode']='farming'
        for _ in range(3):self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.world.training.state['mode']='idle';self.world.training.state['sailing_enabled']=True
        for _ in range(3):self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.assertEqual(self.world.rank,4)
        self.assertEqual(Path(self.world.training.path).read_bytes(),self.saved)
        self.assertEqual(Path(self.sale.path).read_bytes(),self.bag_saved)

    def test_new_bag_votes_do_not_change_navigation_or_renew_sale_deadline(self):
        self.world.phase='find_boss'
        for _ in range(3):self.assertFalse(self.world.observe_backpack_rank(self.frame,self.sale))
        self.assertEqual(self.world.rank,4)
        self.assertEqual(self.world.phase,'find_boss')


if __name__ == '__main__':
    unittest.main()
